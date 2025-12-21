from __future__ import annotations
from typing import Callable, List, Union

from attrs import define, field
from griptape.tasks import BaseTask
from griptape.artifacts import TextArtifact, ImageArtifact, ListArtifact
from griptape.drivers.image_generation.openrouter import OpenRouterImageGenerationDriver



@define
class MultiImageGenerationTask(BaseTask):
    """
    A task that generates an image from a prompt and multiple reference images.
    
    Attributes:
        image_generation_driver: The driver used to generate the image.
        _input: Accepts either:
            - tuple of (prompt string | TextArtifact, list[ImageArtifact])
            - Callable returning ListArtifact [prompt, list of images]
    """
    image_generation_driver: OpenRouterImageGenerationDriver = field(
        default=None,
        kw_only=True
    )
    
    _input: Union[
        tuple[Union[str, TextArtifact], list[ImageArtifact]],
        Callable[[BaseTask], ListArtifact]
    ] = field(default=None, alias="input")
    
    @property
    def input(self) -> ListArtifact:

        if isinstance(self._input, ListArtifact):
            return self._input
            

        if isinstance(self._input, tuple):
            prompt_str, images = self._input
            prompt_artifact = TextArtifact(prompt_str) if isinstance(prompt_str, str) else prompt_str
            

            if isinstance(images, list):
                image_list_artifact = ListArtifact(images)
            else:
                image_list_artifact = ListArtifact([images])
                
            return ListArtifact([prompt_artifact, image_list_artifact])
        
        if callable(self._input):
            return self._input(self)
        
        raise ValueError("Invalid input format")

    @input.setter
    def input(self, value: tuple[str | TextArtifact, list[ImageArtifact]] | Callable[[BaseTask], ListArtifact]):
        self._input = value

    def try_run(self) -> ImageArtifact:
        prompt_artifact: TextArtifact = self.input[0]

        images_artifact: ListArtifact = self.input[1]
        images: list[ImageArtifact] = [img for img in images_artifact.value if isinstance(img, ImageArtifact)]

        if not images:
            raise ValueError("All reference images must be ImageArtifact instances.")


        output_image_artifact = self.image_generation_driver.run_multi_image_generation(
            prompts=[prompt_artifact.to_text()],
            images=images
        )

        return output_image_artifact

