from griptape.structures import Pipeline
from griptape.tasks import PromptTask

pipeline = Pipeline()

pipeline.add_task(PromptTask("What is Griptape and why is it userful?"))

result = pipeline.run()
print(result.output.value)