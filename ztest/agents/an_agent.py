from griptape.structures import Agent
from tools.math_tools import MTools

agent = Agent(tools=[MTools.add_two(1,2)])
agent.run("add the two numbers.")