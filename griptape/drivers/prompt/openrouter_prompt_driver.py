from __future__ import annotations
from typing import TYPE_CHECKING, Iterator
import json
import requests
from attrs import define, field, Factory
from griptape.drivers.prompt import BasePromptDriver
from griptape.tokenizers import OpenAiTokenizer
from griptape.common import Message, DeltaMessage, TextDeltaMessageContent, PromptStack
from griptape.artifacts import TextArtifact
from griptape.common import TextMessageContent

if TYPE_CHECKING:
    from griptape.tokenizers import BaseTokenizer

@define(kw_only=True)
class OpenRouterPromptDriver(BasePromptDriver):
    """OpenRouter Prompt Driver for Griptape.
    
    Attributes:
        api_key: OpenRouter API key
        model: Model identifier from OpenRouter (e.g., "anthropic/claude-2")
        base_url: OpenRouter API base URL
        tokenizer: Tokenizer instance for token counting
    """
    
    api_key: str = field(metadata={"serializable": True})
    base_url: str = field(default="https://openrouter.ai/api/v1", metadata={"serializable": True})
    tokenizer: BaseTokenizer = field(
        default=Factory(lambda self: OpenAiTokenizer(model=self.model), takes_self=True),
        metadata={"serializable": False},
    )

    def try_run(self, prompt_stack: PromptStack) -> Message:
        """Execute a completion request to OpenRouter."""
        prompt = self.prompt_stack_to_string(prompt_stack)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://griptape.ai",
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **self.extra_params
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        completion = result["choices"][0]["message"]["content"]
        
        return Message(
            content=[TextMessageContent(TextArtifact(completion))],
            role=Message.ASSISTANT_ROLE,
            usage=Message.Usage(
                input_tokens=result["usage"]["prompt_tokens"],
                output_tokens=result["usage"]["completion_tokens"]
            )
        )

    def try_stream(self, prompt_stack: PromptStack) -> Iterator[DeltaMessage]:
        """Stream completion from OpenRouter."""
        prompt = self.prompt_stack_to_string(prompt_stack)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://griptape.ai",
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **self.extra_params
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            stream=True
        )
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                if line.startswith(b"data: "):
                    json_data = json.loads(line[6:])
                    if json_data["choices"]:
                        delta = json_data["choices"][0]["delta"]
                        if "content" in delta:
                            yield DeltaMessage(
                                content=TextDeltaMessageContent(
                                    text=delta["content"],
                                    index=0
                                ),
                                usage=DeltaMessage.Usage(
                                    input_tokens=0,
                                    output_tokens=1
                                )
                            )
