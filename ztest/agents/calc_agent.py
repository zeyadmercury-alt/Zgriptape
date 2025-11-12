from griptape.structures import Agent
from tools.calc_tools import CalcTool

def calcAgent() -> Agent:
    return Agent(
        tools=[
            CalcTool()
        ],
        systemPrompt="you are an expert mathematician. Use the CalcTool tool for any circumference questions"
    )