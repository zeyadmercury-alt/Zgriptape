from griptape.structures import Pipeline
from griptape.tasks import PromptTask
from griptape.drivers.prompt.google import GooglePromptDriver
gemeini_driver=GooglePromptDriver(
    model="gemini-2.5-flash"
)

task1 = PromptTask("Create a story of a futuristic cyberpunk city at sunset in 5 sentences.",
                  id="story",
                  prompt_driver=gemeini_driver
)
task2 = PromptTask("summarize this story {{ parent_output}}",
                   id="summary",
                   parent_ids=["story"],
                   prompt_driver=gemeini_driver
)
task3 = PromptTask("extract keywords {{ parent_output}}",
                   id="keys",
                   parent_ids=["story"],
                   prompt_driver=gemeini_driver
)
task4 = PromptTask("what is the story summary about {{ parent_output}} , send the most important keywords you have from keys",
                   parent_ids=["summary","keys"],
                   prompt_driver=gemeini_driver
)

pl = Pipeline()
pl.add_tasks(
    task1,
    task2,
    task3,
    task4
)