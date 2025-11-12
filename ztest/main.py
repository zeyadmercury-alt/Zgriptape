from griptape.structures import Workflow
from griptape.tasks import PromptTask, PromptImageGenerationTask
from griptape.drivers.prompt.openrouter import OpenRouterPromptDriver
from griptape.drivers.image_generation.openrouter import OpenRouterImageGenerationDriver
orKey="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"
openrouter_image_driver = OpenRouterImageGenerationDriver(
    api_key=orKey
)
openrouter_driver = OpenRouterPromptDriver(
    model="google/gemini-2.5-flash",
    api_key= orKey
    )

task = PromptTask(
    "how are you doing today?",
    prompt_driver=openrouter_driver
)
task2 = PromptImageGenerationTask(
    "generte an image of lighting env",
    image_generation_driver=openrouter_image_driver,
    output_dir="images_generated/"
    )

wf = Workflow(
    tasks=[task]
)
result = task2.run()

# cost = openrouter_image_driver.last_generation_cost
size = openrouter_image_driver.image_size
# (f"Total: ${cost.total_cost_usd:.4f}")
# print(f"Tokens: {cost.prompt_tokens} input, {cost.completion_tokens} output")
# print(f"Images: {cost.input_images} input, {cost.output_images} output")
print("size: ",size)
# # wf.run()
# 