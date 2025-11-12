from griptape.structures import Agent
from griptape.drivers import TavilyWebSearchDriver
from griptape.tools import PromptSummaryTool, WebSearchTool
from griptape.rules import Rule
from griptape.drivers import GooglePromptDriver
from griptape.drivers.prompt import OpenRouterPromptDriver
import os

gemini = GooglePromptDriver(model="gemini-2.5-flash")
driver = TavilyWebSearchDriver(api_key="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")
open_router = OpenRouterPromptDriver(
    api_key = "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--",
    model="google/gemini-2.5-flash",
    temperature=0.7
)
agent = Agent(
    tools=[
        WebSearchTool(web_search_driver=driver)
    ],
    prompt_driver=open_router,
    rules=[
        Rule("the url should be from pinterest"),
        Rule("the url should work so test that url work and generate the right image"),
        Rule("the number of url asked for must response with")
    ]
)
#driver.search("who is lionel Messi?")

agent.run("I want 12 direct url of type jpg for lionel Messi picture or image")
