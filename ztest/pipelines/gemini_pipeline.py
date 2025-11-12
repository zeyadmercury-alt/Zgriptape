from griptape.structures import Pipeline
from griptape.tasks import PromptTask
from griptape.drivers import GooglePromptDriver

gemini_driver = GooglePromptDriver(model="gemini-2.5-flash")
pipeline = Pipeline()

task1 = PromptTask("my name is zeyad", prompt_driver=gemini_driver)
task2=PromptTask("what is my name?", prompt_driver=gemini_driver)

pipeline.add_tasks(task1, task2)
result = pipeline.run()

# print("GEMINI OUTPUT:")
# print(result.output.value)
