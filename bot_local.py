import asyncio
import os
import sys

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


class UserTranscriptLogger(FrameProcessor):
    """Prints recognized user speech to the terminal."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            if frame.text and frame.text.strip():
                print(f"\n🗣️  [用户/User]: {frame.text.strip()}", flush=True)

        await self.push_frame(frame, direction)


class BotTranscriptLogger(FrameProcessor):
    """Prints streaming bot speech tokens to the terminal."""

    def __init__(self):
        super().__init__()
        self._in_bot_response = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TextFrame) and not isinstance(frame, (TranscriptionFrame, InterimTranscriptionFrame)):
            if not self._in_bot_response:
                print("\n🤖 [助手/Bot]: ", end="", flush=True)
                self._in_bot_response = True
            print(frame.text, end="", flush=True)
        elif isinstance(frame, LLMFullResponseEndFrame):
            if self._in_bot_response:
                print("", flush=True)
                self._in_bot_response = False

        await self.push_frame(frame, direction)


async def search_web(params: FunctionCallParams, query: str):
    """Search the live web using Parallel.ai for real-time information, latest news, weather, or current facts.

    Only call this function when the user explicitly asks for up-to-date real-time information,
    current news, weather forecasts, or external factual data that requires internet access.
    Do NOT call this for normal greetings, casual conversation, self-introductions, or calculations.

    Args:
        query: The specific search query or keyword to look up online.
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

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=6)) as resp:
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
                    print(f"✅ [网络搜索/Search]: 成功获取 {len(simplified)} 条实时结果:", flush=True)
                    for i, item in enumerate(simplified, 1):
                        title = item.get("title", "无标题")
                        url = item.get("url", "")
                        snippet = item.get("snippet", "").strip().replace("\n", " ")
                        snippet_preview = (snippet[:120] + "...") if len(snippet) > 120 else snippet
                        print(f"   [{i}] 📌 {title}\n       🔗 {url}\n       📝 {snippet_preview}", flush=True)
                    await params.result_callback({"results": simplified})
                else:
                    err_msg = await resp.text()
                    await params.result_callback({"error": f"HTTP {resp.status}: {err_msg}"})
    except Exception as e:
        await params.result_callback({"error": f"Search failed: {str(e)}"})


async def main():
    logger.info("Initializing Local Voice Agent (Cartesia STT + OpenRouter Cerebras + Parallel Search + Cartesia TTS)...")

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
                "你是一个中文极速语音助手，具备实时联网搜索能力。"
                "仅在用户明确询问实时新闻、天气、股价、最新事件等需要网络最新信息的问题时，调用 search_web 工具。"
                "严禁在日常问候、自我介绍、闲聊、基础计算时调用搜索工具。"
                "回答请保持简短、自然、口语化，严禁使用 markdown 格式或表情符号。"
            ),
            temperature=0.7,
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

    # Spoken notification when search tool is triggered
    @llm.event_handler("on_function_calls_started")
    async def on_function_calls_started(service, function_calls):
        print("\n🔍 [系统/System]: 稍等，我正在网上搜索相关信息...", flush=True)
        await tts.queue_frame(TTSSpeakFrame("稍等，我正在网上搜索相关信息..."))

    # 5. Conversation Context with Web Search Tool & VAD (Silero)
    context = LLMContext(tools=[search_web])
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # 6. Separate Loggers for User and Bot
    user_logger = UserTranscriptLogger()
    bot_logger = BotTranscriptLogger()

    # 7. Pipeline Assembly
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
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    runner = WorkerRunner()
    await runner.add_workers(worker)

    # Initial Greeting in Chinese (spoken directly via TTS without unnecessary search)
    print("\n🤖 [助手/Bot]: 你好！我已经准备好了，支持实时联网搜索。请问今天有什么我可以帮你的？", flush=True)
    await tts.queue_frame(TTSSpeakFrame("你好！我已经准备好了，支持实时联网搜索。请问今天有什么我可以帮你的？"))

    logger.info("Bot is listening! Speak into your microphone...")
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
