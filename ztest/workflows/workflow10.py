from griptape.structures import Workflow 
from griptape.tasks import PromptTask
from griptape.drivers import GooglePromptDriver
from griptape.utils import StructureVisualizer

gemini_driver = GooglePromptDriver(model="gemini-2.5-flash")
wf = Workflow()
categories=["tree", "flowers", "animals", "technology"]
task1 = [PromptTask(f"generate a poam about { category }", id=category, child_ids=['summary'], structure=wf, prompt_driver=gemini_driver) for category in categories]
task2 = PromptTask("summarize this {{ parents_output_text }}", id="summary", structure=wf, prompt_driver=gemini_driver)

#you can say {{ parents_output_text }}  or   {{ parent_output }}

#wf.run()
print(StructureVisualizer(wf).to_url())
print([task.id for task in wf.input_tasks])
print([task.id for task in wf.output_tasks])

