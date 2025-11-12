# Imperatively insert parallel tasks between a parent and child

from griptape.rules import Rule
from griptape.structures import Workflow
from griptape.tasks import PromptTask
from griptape.drivers import GooglePromptDriver
from griptape.utils import StructureVisualizer

gemini_driver = GooglePromptDriver(model="gemini-2.5-flash")

wf = Workflow(
    rules=[Rule("output a single lowercase word")],
)

animal_task = PromptTask("Name an animal", id="animal", structure=wf, prompt_driver=gemini_driver)
adjective_task = PromptTask("Describe {{ parent_outputs['animal'] }} with an adjective", id="adjective", structure=wf, prompt_driver=gemini_driver)
color_task = PromptTask("Describe {{ parent_outputs['animal'] }} with a color", id="color", structure=wf, prompt_driver=gemini_driver)
new_animal_task = PromptTask("Name an animal described as: \n{{ parents_output_text }}", id="new-animal", structure=wf, prompt_driver=gemini_driver)

adjective_task.add_parent(animal_task)
color_task.add_parent(animal_task)
new_animal_task.add_parents([adjective_task, color_task])
# new_animal_task.add_parent(adjective_task)
# new_animal_task.add_parent(color_task)

print(StructureVisualizer(wf).to_url())
wf.run()