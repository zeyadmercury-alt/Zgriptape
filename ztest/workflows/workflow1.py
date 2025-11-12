# declaritive Syntax {direct}

from griptape.structures import Workflow
from griptape.tasks import PromptTask
from griptape.utils import StructureVisualizer
from griptape.drivers import GooglePromptDriver
gemini_driver=GooglePromptDriver(model="gemini-2.5-flash")

world_task = PromptTask(
    "create a small wold contianing those keywords,\n {{ keywords|join(', ')}}",
    context={"keywords":["class", "pen", "paper"]},
    id="World",
    prompt_driver=gemini_driver
)

def CharCreation(task_id: str, char_name:str) -> PromptTask:
    return PromptTask(
        "Based on the following small world description create a character named\n {{name}}\n {{parent_outputs['World']}}",
        context={"name":char_name},
        id=task_id,
        parent_ids=["World"],
        prompt_driver=gemini_driver
    )

ali_task = CharCreation("ali", "ali")
omar_task = CharCreation("omar", "omar")

story_task= PromptTask(
    "create a story based on the world and characters for children in 3 sentences, write the story\n {{parent_outputs['World']}}\n {{parent_outputs['ali']}}\n {{parent_outputs['omar']}}",
    id="Story",
    parent_ids=['World', 'ali', 'omar'],
    prompt_driver=gemini_driver
)

wf = Workflow(
    tasks=[world_task, ali_task, omar_task, story_task]
)

print(StructureVisualizer(wf).to_url())

wf.run()