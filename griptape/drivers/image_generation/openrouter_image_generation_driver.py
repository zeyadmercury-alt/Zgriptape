from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional

import requests
from attrs import define, field

from griptape.drivers.image_generation import BaseImageGenerationDriver
from griptape.artifacts import ImageArtifact

import re


@dataclass
class ImageGenerationCost:
    prompt_tokens: int
    completion_tokens: int
    input_images: int
    output_images: int
    total_cost_usd: float
    model_used: str

    def __str__(self):
        return (
            f"Cost: ${self.total_cost_usd:.4f}, "
            f"Tokens: {self.prompt_tokens}+{self.completion_tokens}, "
            f"Images: {self.input_images}+{self.output_images}, "
            f"Model: {self.model_used}"
        )


@define
class OpenRouterImageGenerationDriver(BaseImageGenerationDriver):
    """Driver for the OpenRouter image generation API"""

    # --- pricing per 1K units ---
    MODEL_PRICES = {
        "google/gemini-2.5-flash-image": {
            "input_tokens": 0.0003,    # $0.30 / 1M input tokens
            "output_tokens": 0.0025,   # $2.50 / 1M output tokens
            "input_images": 0.001238,  # $1.238 / 1K input images
            "output_images": 0.00003,  # $0.03 / 1K output images
        },
        "google/gemini-2.5-flash-image-preview": {
            "input_tokens": 0.0003,    # $0.30 / 1M input tokens
            "output_tokens": 0.0025,   # $2.50 / 1M output tokens
            "input_images": 0.001238,  # $1.238 / 1K input images
            "output_images": 0.00003,  # $0.03 / 1K output images
        },
        "openai/gpt-5-image": {
            "input_tokens": 0.005,     # $5.00 / 1M input tokens
            "output_tokens": 0.015,    # $15.00 / 1M output tokens
            "input_images": 0.002,     # $2.00 / 1K input images
            "output_images": 0.002,    # $2.00 / 1K output images
        },
        "openai/gpt-5-image-mini": {
            "input_tokens": 0.0025,    # $2.50 / 1M input tokens
            "output_tokens": 0.0075,   # $7.50 / 1M output tokens
            "input_images": 0.001,     # $1.00 / 1K input images
            "output_images": 0.001,    # $1.00 / 1K output images
        }
    }

    base_url: str = field(default="https://openrouter.ai/api/v1", kw_only=True)
    endpoint: str = field(default="/chat/completions", kw_only=True)
    api_key: Optional[str] = field(default=None, kw_only=True)
    model: str = field(default="google/gemini-2.5-flash-image", kw_only=True)
    image_size: str = field(default="1024x1024", kw_only=True)
    timeout: int = field(default=120, kw_only=True)

    def _build_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def try_text_to_image(self, prompts: list[str], negative_prompts: Optional[list[str]] = None) -> ImageArtifact:
        """Generate an image from text prompts with cost tracking."""
        prompt = ", ".join(prompts)
        # Building the message
        payload = {
            "model": self.model,
            "prompt": prompt,
            "n": 1, # numbers of images to generate
            "size": self.image_size, # resolution
            "response_format": "b64_json",
        }

        url = f"{self.base_url.rstrip('/')}{self.endpoint}"
        # send the request
        resp = requests.post(url, json=payload, headers=self._build_headers(), timeout=self.timeout)
        resp.raise_for_status()
        body = resp.json()

        # response like this:
        # {
        #   "usage": {"prompt_tokens": 9, "completion_tokens": 1290},
        #    "data": [{"b64_json": "iVBORw0KGgo..."}]
        # }

        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        input_images = usage.get("input_images", 0)
        output_images = usage.get("output_images", 1)

        prices = self.MODEL_PRICES.get(self.model)
        if prices is None:
            prices = {
                "input_tokens": 0.001,
                "output_tokens": 0.001,
                "input_images": 0.001,
                "output_images": 0.001,
            }

        total_cost = (
            (prompt_tokens / 1000.0) * prices["input_tokens"]
            + (completion_tokens / 1000.0) * prices["output_tokens"]
            + (input_images) * prices["input_images"]
            + (output_images) * prices["output_images"]
        )

        cost_info = ImageGenerationCost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_images=input_images,
            output_images=output_images,
            total_cost_usd=total_cost,
            model_used=self.model
        )

        b64 = None
        if "data" in body and len(body["data"]) > 0 and "b64_json" in body["data"][0]:
            b64 = body["data"][0]["b64_json"]
        elif "choices" in body and len(body["choices"]) > 0:
            choice = body["choices"][0]
            if "images" in choice and len(choice["images"]) > 0:
                img_url = choice["images"][0].get("image_url", {}).get("url")
                if img_url and img_url.startswith("data:image"):
                    b64 = img_url.split(",")[1]

        if not b64:
            raise Exception(f"No base64 image found in response: {body}")

        image_bytes = base64.b64decode(b64)
        sizez = self.image_size
        width, height = [int(dim) for dim in sizez.split('x')]

        image_artifact = ImageArtifact(
            value=image_bytes,
            format="png",
            width=width,
            height=height,
            meta={
                "prompt": prompt,
                "model": self.model,
                "usage": {
                    "prompt_tokens": cost_info.prompt_tokens,
                    "completion_tokens": cost_info.completion_tokens,
                    "input_images": cost_info.input_images,
                    "output_images": cost_info.output_images,
                    "total_cost_usd": cost_info.total_cost_usd,
                    "model_used": cost_info.model_used
                }
            }
        )

        self.last_generation_cost = cost_info
        return image_artifact

    def try_image_inpainting(self, *args, **kwargs):
        raise NotImplementedError("Inpainting not yet supported for OpenRouter driver.")

    def try_image_outpainting(self, *args, **kwargs):
        raise NotImplementedError("Outpainting not yet supported for OpenRouter driver.")

    def try_image_variation(self, *args, **kwargs):
        raise NotImplementedError("Variation not yet supported for OpenRouter driver.")
