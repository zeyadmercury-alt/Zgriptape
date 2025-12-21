from __future__ import annotations

import asyncio
import base64
import logging
from typing import Optional, List

import httpx
from attrs import define, field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from griptape.drivers.image_generation import BaseImageGenerationDriver
from griptape.artifacts import ImageArtifact

logger = logging.getLogger(__name__)


@define
class OpenRouterImageGenerationDriver(BaseImageGenerationDriver):
    base_url: str = field(default="https://openrouter.ai/api/v1", kw_only=True)
    endpoint: str = field(default="/chat/completions", kw_only=True)
    api_key: Optional[str] = field(default=None, kw_only=True)
    model: str = field(default="google/gemini-2.5-flash-image", kw_only=True)
    image_size: str = field(default="1024x1024", kw_only=True)
    timeout: int = field(default=120, kw_only=True)

    # ------------------------------------------------------------------
    # Required sync
    # ------------------------------------------------------------------

    def _run_async_in_sync_context(self, coro):
        """Helper to run async coroutines in sync context."""
        try:
            asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        except RuntimeError:
            return asyncio.run(coro)

    def try_text_to_image(self, prompts: List[str], negative_prompts=None) -> ImageArtifact:
        return self._run_async_in_sync_context(self.try_text_to_image_async(prompts, negative_prompts))

    def try_image_variation(
        self,
        prompts: List[str],
        image: ImageArtifact,
        negative_prompts=None,
    ) -> ImageArtifact:
        return self._run_async_in_sync_context(self.try_image_variation_async(prompts, image, negative_prompts))

    def run_multi_image_generation(
        self,
        prompts: List[str],
        images: List[ImageArtifact],
    ) -> ImageArtifact:
        return self._run_async_in_sync_context(self.run_multi_image_generation_async(prompts, images))

    # ------------------------------------------------------------------
    # Async implementations (real logic)
    # ------------------------------------------------------------------


    def _get_dimensions(self) -> tuple[int, int]:
        try:
            w, h = self.image_size.lower().split("x")
            return int(w), int(h)
        except Exception:
            return 1024, 1024


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def try_text_to_image_async(self, prompts, negative_prompts=None) -> ImageArtifact:
        prompt = ", ".join(prompts)

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "modalities": ["image", "text"],
        }

        body = await self._post(payload)
        image_bytes = self._extract_image_bytes(body)

        width, height = self._get_dimensions()

        return ImageArtifact(
            value=image_bytes,
            format="png",
            width=width,
            height=height,
            meta={"prompt": prompt, "model": self.model},
        )


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def try_image_variation_async(self, prompts, image, negative_prompts=None) -> ImageArtifact:
        prompt = ", ".join(prompts)
        image_b64 = base64.b64encode(image.value).decode()

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            "modalities": ["image", "text"],
        }

        body = await self._post(payload)
        image_bytes = self._extract_image_bytes(body)

        width, height = self._get_dimensions()

        return ImageArtifact(
            value=image_bytes,
            format="png",
            width=width,
            height=height,
            meta={"prompt": prompt, "model": self.model},
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def run_multi_image_generation_async(
        self,
        prompts: List[str],
        images: List[ImageArtifact],
    ) -> ImageArtifact:
        prompt = ", ".join(prompts)

        image_parts = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64.b64encode(img.value).decode()}"
                },
            }
            for img in images
        ]

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}, *image_parts],
                }
            ],
            "modalities": ["image", "text"],
        }

        body = await self._post(payload)
        image_bytes = self._extract_image_bytes(body)
        width, height = self._get_dimensions()

        return ImageArtifact(
            value=image_bytes,
            format="png",
            width=width,
            height=height,
            meta={"prompt": prompt, "model": self.model},
        )

    def try_image_inpainting(
        self,
        prompts: List[str],
        image: ImageArtifact,
        mask: ImageArtifact,
        negative_prompts=None,
    ) -> ImageArtifact:
        raise NotImplementedError("Image inpainting is not supported by OpenRouter driver")

    def try_image_outpainting(
        self,
        prompts: List[str],
        image: ImageArtifact,
        mask: ImageArtifact,
        negative_prompts=None,
    ) -> ImageArtifact:
        raise NotImplementedError("Image outpainting is not supported by OpenRouter driver")

    async def _post(self, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url.rstrip('/')}{self.endpoint}",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _extract_image_bytes(self, body: dict) -> bytes:
        b64 = None

        if "choices" in body:
            msg = body["choices"][0].get("message", {})
            if "images" in msg:
                url = msg["images"][0]["image_url"]["url"]
                b64 = url.split(",")[1]

        if not b64 and "data" in body:
            b64 = body["data"][0].get("b64_json")

        if not b64:
            raise RuntimeError(f"No image found in response: {body}")

        return base64.b64decode(b64)
