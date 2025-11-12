# mix imperative and declarative

from griptape.structures import Workflow
from griptape.tasks import PromptTask
from griptape.drivers import GooglePromptDriver
from griptape.utils import StructureVisualizer
from griptape.rules import Rule

gemini_driver = GooglePromptDriver(model="gemini-2.5-flash")

wf = Workflow(
    rules=[
        Rule("Output only one word")
    ]
)

anim_task = PromptTask("name an animal", id='anim', child_ids=['desc'], structure=wf, prompt_driver=gemini_driver)
desc_task = PromptTask("describe {{parent_outputs['anim']}}, with an adjective.", id='desc', structure=wf, prompt_driver=gemini_driver)
new_anim_task = PromptTask("name a {{parent_outputs['desc']}} animal", id='new_anim', structure=wf, prompt_driver=gemini_driver)

desc_task.add_child(new_anim_task)
print(StructureVisualizer(wf).to_url())
wf.run()