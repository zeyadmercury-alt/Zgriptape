from griptape.structures import Agent
from griptape.tasks import PromptTask
from griptape.drivers.prompt.google import GooglePromptDriver
from griptape.tools import WebScraperTool, PromptSummaryTool
from griptape.engines.summary import prompt_summary_engine
from griptape.configs import defaults_config
from griptape.configs.drivers.google_drivers_config import GoogleDriversConfig
from griptape.engines.summary.prompt_summary_engine import PromptSummaryEngine
from griptape.chunkers.text_chunker import TextChunker
gemini = GooglePromptDriver(
    model="gemini-2.5-flash",
    use_native_tools=False
)




agent = Agent(
    tools=[
        WebScraperTool(off_prompt=True),
        PromptSummaryTool(prompt_driver=gemini, off_prompt=False)
    ],
    prompt_driver=gemini
)
