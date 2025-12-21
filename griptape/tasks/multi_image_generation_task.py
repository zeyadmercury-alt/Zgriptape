from __future__ import annotations
from typing import Callable, Union

from attrs import define, field
from griptape.tasks import BaseTask
from griptape.artifacts import TextArtifact, ImageArtifact, ListArtifact
from griptape.drivers.image_generation.openrouter import OpenRouterImageGenerationDriver


@define
class MultiImageGenerationTask(BaseTask):
    image_generation_driver: OpenRouterImageGenerationDriver = field(kw_only=True)

    _input: Union[
        tuple[Union[str, TextArtifact], list[ImageArtifact]],
        Callable[[BaseTask], ListArtifact],
    ] = field(default=None, alias="input")

    @property
    def input(self) -> ListArtifact:
        if isinstance(self._input, tuple):
            prompt, images = self._input
            prompt_art = TextArtifact(prompt) if isinstance(prompt, str) else prompt
            return ListArtifact([prompt_art, ListArtifact(images)])

        if callable(self._input):
            return self._input(self)

        raise ValueError("Invalid input")

    async def run(self, *args) -> ImageArtifact:
        """Async version for parallel execution."""
        return self.try_run()

    def try_run(self) -> ImageArtifact:
        prompt_artifact: TextArtifact = self.input[0]
        images_artifact: ListArtifact = self.input[1]

        images = [img for img in images_artifact.value if isinstance(img, ImageArtifact)]
        if not images:
            raise ValueError("No valid ImageArtifact inputs")

        return self.image_generation_driver.run_multi_image_generation(
            prompts=[prompt_artifact.to_text()],
            images=images,
        )
