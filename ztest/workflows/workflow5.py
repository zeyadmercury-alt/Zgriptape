# Imperatively specify parents

from griptape.structures import Workflow
from griptape.tasks import PromptTask
from griptape.rules import Rule

anim_task = PromptTask("name an animal", id='anim')
desc_task = PromptTask("describe {{parent_outputs['anim']}}, with an adjective", id='desc')
new_anim_task = PromptTask("name a {{parent_outputs['desc']}} animal", id='new_anim')

desc_task.add_parent(anim_task)
new_anim_task.add_parent(desc_task)

wf = Workflow(
    tasks=[
        anim_task, desc_task, new_anim_task
    ],
    rules=[
        Rule("output only one word")
    ]
)

wf.run()