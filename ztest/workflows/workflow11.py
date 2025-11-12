from griptape.drivers import GooglePromptDriver
from griptape.tasks.prompt_task import PromptTask
from griptape.structures import Workflow
from griptape.utils import StructureVisualizer

gemeini_driver = GooglePromptDriver(
    model="gemini-2.5-flash"
)

task1 = PromptTask("Create a story of a futuristic cyberpunk city at sunset in 5 sentences.",
                  id="story",
                  prompt_driver=gemeini_driver
)
task2 = PromptTask("summarize this story {{ parent_outputs['story']}}",
                   id="summary",
                   parent_ids=["story"],
                   prompt_driver=gemeini_driver
)
task3 = PromptTask("extract keywords {{ parent_outputs['story']}}",
                   id="keys",
                   parent_ids=["story"],
                   prompt_driver=gemeini_driver
)
task4 = PromptTask("what is the story summary about {{ parent_outputs['summary']}} , send the most important keywords you have from {{parent_outputs['keys']}}",
                   parent_ids=["summary","keys"],
                   prompt_driver=gemeini_driver
)
wf =  Workflow(
    tasks=[
        task1,
        task2,
        task3,
        task4,
    ]
)
print(StructureVisualizer(wf).to_url())
