import asyncio
import os
import sys
import time

import aiohttp
from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    Frame,
    InterimTranscriptionFrame,
    LLMFullResponseEndFrame,
    LLMRunFrame,
    TextFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.observers.user_bot_latency_observer import UserBotLatencyObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.services.cartesia.tts import CartesiaTTSService, GenerationConfig
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openrouter.llm import OpenRouterLLMService
from pipecat.transcriptions.language import Language
from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioTransportParams
from pipecat.workers.runner import WorkerRunner

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="INFO")

SHOW_LATENCY = os.getenv("SHOW_LATENCY_METRICS", "false").strip().lower() in ("true", "1", "yes")


class UserTranscriptLogger(FrameProcessor):
    """Prints recognized user speech to the terminal."""

    def __init__(self, bot_logger=None):
        super().__init__()
        self.bot_logger = bot_logger

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            if frame.text and frame.text.strip():
                if self.bot_logger and SHOW_LATENCY:
                    self.bot_logger.mark_user_speech()
                print(f"\n🗣️  [用户/User]: {frame.text.strip()}", flush=True)

        await self.push_frame(frame, direction)


class BotTranscriptLogger(FrameProcessor):
    """Prints streaming bot speech tokens to the terminal and displays latency report underneath."""

    def __init__(self):
        super().__init__()
        self._in_bot_response = False
        self._user_speech_time = None
        self._pending_breakdown = None
        self._pending_total_latency = None

    def mark_user_speech(self):
        self._user_speech_time = time.time()

    def set_latency_breakdown(self, breakdown):
        self._pending_breakdown = breakdown
        # If response has already finished, print immediately
        if not self._in_bot_response and (self._pending_breakdown or self._pending_total_latency):
            self.print_latency_report()

    def set_total_latency(self, total_secs):
        self._pending_total_latency = total_secs
        # If response has already finished, print immediately
        if not self._in_bot_response and (self._pending_breakdown or self._pending_total_latency):
            self.print_latency_report()

    def print_latency_report(self):
        if not SHOW_LATENCY:
            return
        if self._pending_breakdown is not None:
            bd = self._pending_breakdown
            print("\n⏱️  [延迟分析 / Latency Breakdown]:", flush=True)
            if bd.user_turn_secs is not None:
                print(f"   • 🎙️ VAD静音检测 & Cartesia STT语音识别结算: {bd.user_turn_secs * 1000:.0f} ms", flush=True)
            for t in bd.ttfb:
                proc_label = (
                    t.processor
                    .replace("OpenRouterLLMService#0", "🧠 LLM (Cerebras)")
                    .replace("CartesiaTTSService#0", "🔊 TTS (Cartesia)")
                    .replace("CartesiaSTTService#0", "🎙️ STT (Cartesia)")
                )
                print(f"   • ⚡ {proc_label} 首字/首包 (TTFB/TTFT): {t.duration_secs * 1000:.0f} ms", flush=True)
            for fc in bd.function_calls:
                print(f"   • 🛠️ 工具调用 [{fc.function_name}]: {fc.duration_secs * 1000:.0f} ms", flush=True)
            if bd.text_aggregation:
                print(f"   • 📝 文本切分与聚合耗时: {bd.text_aggregation.duration_secs * 1000:.0f} ms", flush=True)
            self._pending_breakdown = None

        if self._pending_total_latency is not None:
            print(f"   🚀 端到端语音响应总延迟: {self._pending_total_latency * 1000:.0f} ms\n", flush=True)
            self._pending_total_latency = None

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and not isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if not self._in_bot_response:
                ttft_info = ""
                if SHOW_LATENCY and self._user_speech_time:
                    ttft_ms = (time.time() - self._user_speech_time) * 1000
                    ttft_info = f" ⚡[首字/TTFT: {ttft_ms:.0f}ms]"
                print(f"\n🤖 [助手/Bot]{ttft_info}: ", end="", flush=True)
                self._in_bot_response = True
            print(frame.text, end="", flush=True)
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._in_bot_response:
                print("", flush=True)
                self._in_bot_response = False
                self._user_speech_time = None
                # Print buffered latency report cleanly at the bottom!
                self.print_latency_report()

        await self.push_frame(frame, direction)


async def search_knowledge_base(params: FunctionCallParams, query: str):
    """Search the internal Misuedi knowledge base for specific questions about the ARK project, ARK whitepaper, or ARK governance.

    CRITICAL: ONLY invoke this tool if the user's query explicitly mentions ARK, ARK whitepaper, ARK tokenomics, or ARK protocol architecture.
    DO NOT invoke for general questions, ordinary blockchain concepts, greetings, calculations, or common knowledge.

    Args:
        query: The specific ARK-related query or topic to look up.
    """
    api_key = os.getenv("MISUEDI_API_KEY", "dataset-nV5sJbUz38MXg92S0VNL35i9")
    base_url = os.getenv("MISUEDI_BASE_URL", "https://misuedi.com/v1")
    host = os.getenv("MISUEDI_HOST", "dify.misuedi.com")
    dataset_id = os.getenv("MISUEDI_DEFAULT_DATASET_ID", "08a9c3f9-4e3b-4cc3-8945-173b003abbf9")

    url = f"{base_url}/datasets/{dataset_id}/retrieve"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Host": host,
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "pipecat-voice-agent/1.0",
    }

    print(f"\n📚 [知识库/KB]: 正在检索知识库 '{query}' ...", flush=True)
    t_start = time.time()

    async def _do_retrieve(method: str):
        payload = {
            "query": query,
            "retrieval_model": {
                "search_method": method,
                "reranking_enable": False,
                "score_threshold_enabled": False,
                "top_k": 3,
            },
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                return resp.status, (await resp.json() if resp.status == 200 else await resp.text())

    try:
        status, data = await _do_retrieve("semantic_search")
        if status != 200:
            # Fallback to keyword_search if semantic search returned 400
            status, data = await _do_retrieve("keyword_search")

        dur_ms = (time.time() - t_start) * 1000
        timing_str = f" (耗时: {dur_ms:.0f}ms)" if SHOW_LATENCY else ""

        if status == 200 and isinstance(data, dict):
            records = data.get("records", [])
            simplified = []
            for r in records[:3]:
                seg = r.get("segment", {})
                doc = seg.get("document", {})
                score = r.get("score", 0.0)
                content = seg.get("content", "")
                simplified.append({
                    "score": round(score, 2),
                    "document": doc.get("name", "Unknown Document"),
                    "content": content,
                })

            print(f"✅ [知识库/KB]: 成功匹配到 {len(simplified)} 条知识库记录{timing_str}:", flush=True)
            for i, item in enumerate(simplified, 1):
                doc_name = item.get("document")
                score = item.get("score")
                content_preview = item.get("content", "").strip().replace("\n", " ")
                preview = (content_preview[:120] + "...") if len(content_preview) > 120 else content_preview
                print(f"   [{i}] 📄 {doc_name} (匹配度: {score})\n       📝 {preview}", flush=True)

            await params.result_callback({"records": simplified})
        else:
            print(f"⚠️ [知识库/KB]: 检索失败 (Status {status}{timing_str}): {data}", flush=True)
            await params.result_callback({"error": f"HTTP {status}: {data}"})
    except Exception as e:
        dur_ms = (time.time() - t_start) * 1000
        timing_str = f" (耗时: {dur_ms:.0f}ms)" if SHOW_LATENCY else ""
        print(f"⚠️ [知识库/KB]: 发生异常{timing_str}: {e}", flush=True)
        await params.result_callback({"error": f"KB search failed: {str(e)}"})


async def search_web(params: FunctionCallParams, query: str):
    """Search the live web using Parallel.ai ONLY for dynamic real-time information (e.g. today's live stock prices, current weather forecast, breaking news today).

    CRITICAL: ONLY invoke this tool if the user explicitly asks to search the web or asks for volatile real-time data that requires live internet (such as today's live weather, today's stock price, or today's breaking news).
    DO NOT invoke for casual conversation, historical facts, explanations, coding, math, greetings, or general knowledge.

    Args:
        query: The concise real-time search query.
    """
    api_key = os.getenv("PARALLEL_API_KEY")
    if not api_key:
        await params.result_callback({"error": "PARALLEL_API_KEY is not set."})
        return

    url = "https://api.parallel.ai/v1/search"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }
    payload = {
        "objective": query,
        "search_queries": [query],
    }

    print(f"\n🌐 [网络搜索/Search]: 正在查询 '{query}' ...", flush=True)
    t_start = time.time()

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=6)) as resp:
                dur_ms = (time.time() - t_start) * 1000
                timing_str = f" (耗时: {dur_ms:.0f}ms)" if SHOW_LATENCY else ""

                if resp.status == 200:
                    data = await resp.json()
                    results = data.get("results", [])
                    simplified = []
                    for r in results[:3]:
                        excerpts = "\n".join(r.get("excerpts", [])[:2])
                        simplified.append({
                            "title": r.get("title", ""),
                            "url": r.get("url", ""),
                            "snippet": excerpts,
                        })
                    print(f"✅ [网络搜索/Search]: 成功获取 {len(simplified)} 条实时结果{timing_str}:", flush=True)
                    for i, item in enumerate(simplified, 1):
                        title = item.get("title", "无标题")
                        url = item.get("url", "")
                        snippet = item.get("snippet", "").strip().replace("\n", " ")
                        snippet_preview = (snippet[:120] + "...") if len(snippet) > 120 else snippet
                        print(f"   [{i}] 📌 {title}\n       🔗 {url}\n       📝 {snippet_preview}", flush=True)
                    await params.result_callback({"results": simplified})
                else:
                    err_msg = await resp.text()
                    print(f"⚠️ [网络搜索/Search]: 失败 (Status {resp.status}{timing_str}): {err_msg}", flush=True)
                    await params.result_callback({"error": f"HTTP {resp.status}: {err_msg}"})
    except Exception as e:
        dur_ms = (time.time() - t_start) * 1000
        timing_str = f" (耗时: {dur_ms:.0f}ms)" if SHOW_LATENCY else ""
        print(f"⚠️ [网络搜索/Search]: 发生异常{timing_str}: {e}", flush=True)
        await params.result_callback({"error": f"Search failed: {str(e)}"})


async def main():
    logger.info(
        f"Initializing Local Voice Agent (STT + OpenRouter Cerebras + Misuedi KB + Parallel Search + TTS)"
        f" [Latency Display: {'ON' if SHOW_LATENCY else 'OFF'}]..."
    )

    # 1. Local Audio Transport (Microphone + Speakers)
    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )

    # 2. Cartesia STT (Chinese Mandarin)
    stt = CartesiaSTTService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaSTTService.Settings(
            language=Language.ZH,
            model="ink-whisper",
        ),
    )

    # 3. OpenRouter LLM (Cerebras Ultra-Fast Llama 3.3 70B)
    llm = OpenRouterLLMService(
        api_key=os.environ["OPENROUTER_API_KEY"],
        settings=OpenRouterLLMService.Settings(
            model="meta-llama/llama-3.3-70b-instruct:cerebras",
            system_instruction=(
                "你是一个极速中文语音助手。"
                "【基本要求】：直接且自然地回答用户，保持口语化、简短，严禁输出 markdown 格式、列表、特殊符号或表情符号。"
                "\n【工具调用严格守则】：\n"
                "1. 默认直接根据自身知识回答，绝不滥用搜索工具；\n"
                "2. 严禁在问候、打招呼、闲聊、自我介绍、常识、概念解释、数学计算或一般问答时调用任何工具；\n"
                "3. 仅当用户明确询问【ARK 项目】、【ARK 白皮书】、【ARK 治理/代币】时，才调用 search_knowledge_base；\n"
                "4. 仅当用户明确询问【实时最新新闻】、【今日实时行情/天气/股价】或明确要求【帮我上网查一下】时，才调用 search_web；\n"
                "5. 若不满足上述严格条件，一律直接回答。"
            ),
            temperature=0.6,
            max_tokens=200,
        ),
        extra_body={
            "provider": {
                "order": ["Cerebras"],
                "allow_fallbacks": True,
            }
        },
    )

    # 4. Cartesia TTS (Chinese Conversational Voice)
    voice_id = os.getenv("CARTESIA_VOICE_ID", "e90c6678-f0d3-4767-9883-5d0ecf5894a8")
    if not voice_id or voice_id == "...":
        voice_id = "e90c6678-f0d3-4767-9883-5d0ecf5894a8"

    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        settings=CartesiaTTSService.Settings(
            model="sonic-3.5",
            language=Language.ZH,
            voice=voice_id,
            generation_config=GenerationConfig(
                speed=1.0,
                emotion="content",
            ),
        ),
    )

    # Spoken notification when search/KB tool is triggered
    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        for call in function_calls:
            fn_name = getattr(getattr(call, "function", None), "name", str(call))
            if "knowledge" in fn_name:
                print("\n🔍 [系统/System]: 稍等，我正在检索专属知识库...", flush=True)
                await tts.queue_frame(TTSSpeakFrame("稍等，我正在查询专属知识库..."))
                return
        print("\n🔍 [系统/System]: 稍等，我正在网上搜索相关信息...", flush=True)
        await tts.queue_frame(TTSSpeakFrame("稍等，我正在网上搜索相关信息..."))

    # 5. Conversation Context with both Misuedi Knowledge Base & Web Search Tools
    context = LLMContext(tools=[search_knowledge_base, search_web])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # 6. Separate Loggers for User and Bot (with TTFT tracking)
    bot_logger = BotTranscriptLogger()
    user_logger = UserTranscriptLogger(bot_logger=bot_logger)

    # 7. Optional Latency Observer for breakdown
    observers = []
    if SHOW_LATENCY:
        latency_observer = UserBotLatencyObserver()

        @latency_observer.event_handler("on_latency_breakdown")
        async def on_latency_breakdown(observer, breakdown):
            bot_logger.set_latency_breakdown(breakdown)

        @latency_observer.event_handler("on_latency_measured")
        async def on_latency_measured(observer, latency_secs):
            bot_logger.set_total_latency(latency_secs)

        observers.append(latency_observer)

    # 8. Pipeline Assembly
    pipeline = Pipeline(
        [
            transport.input(),      # Microphone audio in
            stt,                    # Cartesia Speech-to-Text
            user_logger,            # Logs user speech right after STT
            user_aggregator,        # User turn detection & context
            llm,                    # OpenRouter (Cerebras) LLM with tools
            bot_logger,             # Logs streaming bot responses
            tts,                    # Cartesia Text-to-Speech
            transport.output(),     # Speaker audio out
            assistant_aggregator,   # Bot turn tracking
        ]
    )

    worker = PipelineWorker(
        pipeline,
        observers=observers,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    runner = WorkerRunner()
    await runner.add_workers(worker)

    # Initial Greeting in Chinese
    print("\n🤖 [助手/Bot]: 你好！我已经准备好了，支持专属知识库与实时联网搜索。请问今天有什么我可以帮你的？", flush=True)
    await tts.queue_frame(TTSSpeakFrame("你好！我已经准备好了，支持专属知识库与实时联网搜索。请问今天有什么我可以帮你的？"))

    logger.info("Bot is listening! Speak into your microphone...")
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
