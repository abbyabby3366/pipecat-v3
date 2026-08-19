import asyncio
import os
import sys

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


async def main():
    logger.info("Initializing Local Voice Agent (Cartesia STT + OpenRouter Cerebras + Cartesia TTS)...")

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
                "你是一个极速中文语音助手。你的回答将被直接转换成语音朗读给用户，"
                "请保持简短、自然、口语化，像真人对话一样。"
                "严禁使用 markdown 格式、列表、粗体或表情符号。"
            ),
            temperature=0.7,
            max_tokens=150,
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

    # 5. Conversation Context & VAD (Silero)
    context = LLMContext()
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
            user_logger,            # Logs user speech right after STT!
            user_aggregator,        # User turn detection & context
            llm,                    # OpenRouter (Cerebras) LLM
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

    # Initial Greeting in Chinese
    context.add_message({
        "role": "user",
        "content": "请用一句话热情简短地打个招呼，告诉我已经准备好了。"
    })
    await worker.queue_frames([LLMRunFrame()])

    logger.info("Bot is listening! Speak into your microphone...")
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
