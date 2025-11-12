from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional

import requests
from attrs import define, field

from griptape.drivers.image_generation import BaseImageGenerationDriver
from griptape.artifacts import ImageArtifact


@dataclass
class ImageGenerationCost:
    prompt_tokens: int
    completion_tokens: int
    input_images: int
    output_images: int
    total_cost_usd: float
    model_used: str


@define
class OpenRouterImageGenerationDriver(BaseImageGenerationDriver):
    """Compact OpenRouter image generation driver with model-aware cost calculation."""

    # Model prices expressed per 1K units (tokens or images)
    # These values need to be manually updated if OpenRouter changes its pricing, as there is no direct API to fetch them dynamically.
    MODEL_PRICES = {
        "google/gemini-2.5-flash-image": {
            "input_tokens": 0.0003,    # $0.30 / 1M input tokens -> $0.0003 / 1K
            "output_tokens": 0.0025,   # $2.50 / 1M output tokens -> $0.0025 / 1K
            "input_images": 1.238,     # $1.238 / 1K input images
            "output_images": 0.03,     # $0.03 / 1K output images
        },
        "google/gemini-2.5-flash-image-preview": {
            "input_tokens": 0.0003,
            "output_tokens": 0.0025,
            "input_images": 1.238,
            "output_images": 0.03,
        },
        "openai/gpt-5-image": {
            "input_tokens": 0.01,      # $10.00 / 1M -> $0.01 / 1K
            "output_tokens": 0.01,     # $10.00 / 1M -> $0.01 / 1K
            "input_images": 0.01,      # $0.01 / 1K input images
            "output_images": 0.04,     # $0.04 / 1K output images
        },
        "openai/gpt-5-image-mini": {
            "input_tokens": 0.0025,    # $2.50 / 1M -> $0.0025 / 1K
            "output_tokens": 0.002,    # $2.00 / 1M -> $0.002 / 1K
            "input_images": 0.0025,    # $0.0025 / 1K input images
            "output_images": 0.008,    # $0.008 / 1K output images
        },
    }

    base_url: str = field(default="https://openrouter.ai/api/v1", kw_only=True, metadata={"serializable":True})
    endpoint: str = field(default="/chat/completions", kw_only=True)
    api_key: Optional[str] = field(default=None, kw_only=True)
    model: str = field(default="google/gemini-2.5-flash-image", kw_only=True)
    image_size: str = field(default="832x1248", kw_only=True)
    aspect_ratio: Optional[str] = field(default=None, kw_only=True)
    quality: str = field(default="standard", kw_only=True)
    style: str = field(default="natural", kw_only=True)
    timeout: int = field(default=120, kw_only=True)
    last_generation_cost: Optional[ImageGenerationCost] = field(default=None, init=False)

    def _build_headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
            }

    def try_text_to_image(self, prompts: list[str], negative_prompts: Optional[list[str]] = None) -> ImageArtifact:
        prompt = ", ".join(prompts)
        
        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self.model,
            "messages": messages,
            "modalities": ["image", "text"],
        }

        if self.model.startswith("google/"):
            aspect_ratio = self.aspect_ratio
            if not aspect_ratio:
                # A simple mapping from size to aspect ratio, can be expanded.
                size_to_aspect_ratio = {
                    "1024x1024": "1:1", "832x1248": "2:3", "1248x832": "3:2",
                    "864x1184": "3:4", "1184x864": "4:3", "896x1152": "4:5",
                    "1152x896": "5:4", "768x1344": "9:16", "1344x768": "16:9",
                    "1536x672": "21:9"
                }
                aspect_ratio = size_to_aspect_ratio.get(self.image_size, "1:1")
            payload["image_config"] = {"aspect_ratio": aspect_ratio}
        elif self.model.startswith("openai/"):
            payload["n"] = 1
            payload["size"] = self.image_size
            payload["quality"] = self.quality
            payload["style"] = self.style
            payload["response_format"] = "b64_json"


        resp = requests.post(f"{self.base_url.rstrip('/')}{self.endpoint}", json=payload, headers=self._build_headers(), timeout=self.timeout)
        resp.raise_for_status()
        body = resp.json()

        usage = body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        input_images = usage.get("input_images", 0)
        output_images = usage.get("output_images", 1)

        prices = self.MODEL_PRICES.get(
            self.model,
            {"input_tokens": 0.0003, "output_tokens": 0.0025, "input_images": 1.0, "output_images": 1.0},
        )

        total_cost = (
            (prompt_tokens / 1000.0) * prices["input_tokens"]
            + (completion_tokens / 1000.0) * prices["output_tokens"]
            + (input_images / 1000.0) * prices["input_images"]
            + (output_images / 1000.0) * prices["output_images"]
        )

        cost_info = ImageGenerationCost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            input_images=input_images,
            output_images=output_images,
            total_cost_usd=total_cost,
            model_used=self.model,
        )

        b64 = None
        if "choices" in body and len(body["choices"]) > 0:
            message = body["choices"][0].get("message", {})
            if message and "images" in message and len(message["images"]) > 0:
                img_url = message["images"][0].get("image_url", {}).get("url")
                if img_url and img_url.startswith("data:image"):
                    b64 = img_url.split(",")[1]
        
        # Fallback for older/different response formats
        if not b64 and "data" in body and len(body["data"]) > 0 and "b64_json" in body["data"][0]:
            b64 = body["data"][0]["b64_json"]

        if not b64:
            raise Exception(f"No base64 image found in response: {body}")

        image_bytes = base64.b64decode(b64)
        try:
            width, height = [int(dim) for dim in self.image_size.split("x")]
        except Exception:
            width = height = 1024

        image_artifact = ImageArtifact(value=image_bytes, format="png", width=width, height=height, meta={
            "prompt": prompt,
            "model": self.model,
            "usage": {
                "prompt_tokens": cost_info.prompt_tokens,
                "completion_tokens": cost_info.completion_tokens,
                "input_images": cost_info.input_images,
                "output_images": cost_info.output_images,
                "total_cost_usd": cost_info.total_cost_usd,
                "model_used": cost_info.model_used,
            },
        })

        self.last_generation_cost = cost_info
        return image_artifact

    def try_image_inpainting(self, *args, **kwargs):
        raise NotImplementedError("Inpainting not yet supported for OpenRouter driver.")

    def try_image_outpainting(self, *args, **kwargs):
        raise NotImplementedError("Outpainting not yet supported for OpenRouter driver.")

    def try_image_variation(self, *args, **kwargs):
        raise NotImplementedError("Variation not yet supported for OpenRouter driver.")
