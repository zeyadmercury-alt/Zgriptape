# Imperatively specify children
 
from griptape.structures import Workflow
from griptape.tasks import PromptTask
from griptape.drivers import GooglePromptDriver
from griptape.rules import Rule
from griptape.utils import StructureVisualizer

gemini_driver = GooglePromptDriver(model="gemini-2.5-flash")
anim_task = PromptTask("name an animal", id='anim', prompt_driver=gemini_driver)
desc_task = PromptTask("describe {{parent_outputs['anim']}}, with an adjective", id='desc', prompt_driver=gemini_driver)
new_anim_task = PromptTask("name a {{parent_outputs['desc']}} animal", id='new_anim', prompt_driver=gemini_driver)

anim_task.add_child(desc_task)
desc_task.add_child(new_anim_task)

wf = Workflow(
    tasks=[
        anim_task, desc_task, new_anim_task
    ],
    rules=[
        Rule("output only one word")
    ]
)

print(StructureVisualizer(wf).to_url())
wf.run()