from griptape.structures import Agent
from griptape.tasks import PromptTask
from griptape.drivers.prompt.google import GooglePromptDriver
from griptape.tools import CalculatorTool

gemini= GooglePromptDriver(
    model="gemini-2.5-flash"
)

agent = Agent(
    tools=[
        CalculatorTool(off_prompt=True)
    ],
    prompt_driver=gemini
)
