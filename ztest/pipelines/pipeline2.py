from griptape.structures import Pipeline, Workflow
from griptape.tasks import PromptTask
from griptape.drivers import GooglePromptDriver

gemini_driver = GooglePromptDriver(model="gemini-2.5-flash")
pipeline = Pipeline()

# pipeline.add_task(task)
# pipeline.add_task(task2)

pipeline.add_tasks(
    PromptTask("{{ args[0] }}", prompt_driver=gemini_driver),
    PromptTask("summarize: {{ parent_output }}in 2 sentences", prompt_driver=gemini_driver)
)

result = pipeline.run("who is lionel messi?")
