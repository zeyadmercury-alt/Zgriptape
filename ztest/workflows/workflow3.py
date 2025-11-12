# Declaratively specify children syntax [child_ids=['']]

from griptape.structures import Workflow
from griptape.tasks import PromptTask
from griptape.drivers import GooglePromptDriver
from griptape.rules import Rule
from griptape.utils import StructureVisualizer

gemini_driver = GooglePromptDriver(model="gemini-2.5-flash")

wf = Workflow(
    tasks=[
        PromptTask(
            "name an animal", id="anim", child_ids=['desc'], prompt_driver=gemini_driver
        ),
        PromptTask(
            "describe {{parent_outputs['anim']}}, with an adjective", id="desc", child_ids=['new_anim'], prompt_driver=gemini_driver
        ),
        PromptTask(
            "name a {{parent_outputs['desc']}} animal", id='new_anim', prompt_driver=gemini_driver
        )
    ],
    rules=[
        Rule("output a single lowercase word"),
        Rule("this is for children so be aware with the output for there age")
    ]
)
print(StructureVisualizer(wf).to_url())
wf.run()