"""Text generation from audio and a prompt with Qwen2.5-Omni."""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence

import numpy as np
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

__all__ = ["Qwen2_5OmniInference"]

logger = logging.getLogger(__name__)


class Qwen2_5OmniInference:
    """Answer a text prompt about an audio clip with Qwen2.5-Omni, greedily.

    The model is loaded the way the Qwen2.5-Omni README does for text output:
    ``Qwen2_5OmniForConditionalGeneration`` without its audio output (no
    talker), in ``dtype``. Answers come from its text model, the thinker,
    with greedy decoding (``do_sample=False``) of at most ``max_new_tokens``
    tokens. The checkpoint's generation config sets nothing else.

    The thinker is called directly, as in the transformers docstring of
    ``Qwen2_5OmniThinkerForConditionalGeneration``: the outer ``generate``
    in transformers 5.x reads ``self.talker`` even for text output and fails
    once the talker is disabled. For text output it would only forward to
    the thinker anyway.

    Each conversation is one user turn holding the audio clip and then the
    prompt, preceded by ``system_prompt`` when one is given; otherwise the
    chat template inserts its default system turn ("You are a helpful
    assistant.").

    The processor needs ``torchvision``: it builds the video processor even
    for audio-only input.

    Args:
        model_name: Model name on the Hugging Face Hub, or a local directory.
        dtype: Torch dtype name for the weights, e.g. ``bfloat16``.
        attn_implementation: Attention implementation passed to
            ``from_pretrained``, e.g. ``sdpa`` or ``flash_attention_2``.
        max_new_tokens: Maximum number of generated tokens per answer.
        system_prompt: Text of the system turn; ``None`` keeps the chat
            template's default.
        device: Torch device string, injected by the inference provider.

    Examples:
        Selected from a recipe's inference config, usually wrapped by a
        recipe class that builds the prompts:

        .. code-block:: yaml

            model:
              _target_: espnet3.systems.qwen2_5_omni.inference.Qwen2_5OmniInference
              model_name: Qwen/Qwen2.5-Omni-7B
              max_new_tokens: 512

        >>> model = Qwen2_5OmniInference(device="cuda:0")  # doctest: +SKIP
        >>> model(speech=waveform, prompt="What is the emotion?")  # doctest: +SKIP
        'Happy.'
    """

    sample_rate = 16000

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Omni-7B",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
        max_new_tokens: int = 256,
        system_prompt: Optional[str] = None,
        device: str = "cpu",
    ) -> None:
        """Load the model without its talker, and the processor."""
        self.device = torch.device(device)
        self.max_new_tokens = int(max_new_tokens)
        self.system_prompt = system_prompt

        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            dtype=getattr(torch, dtype),
            attn_implementation=attn_implementation,
            enable_audio_output=False,
        )
        model.disable_talker()
        self.model = model.to(self.device).eval()
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        self.dtype = getattr(torch, dtype)

    def build_conversation(self, prompt: str) -> List[dict]:
        """Return the chat messages for one clip and one prompt.

        Args:
            prompt: The user's text, placed after the audio clip.

        Returns:
            list[dict]: Messages for ``processor.apply_chat_template``.
        """
        messages = []
        if self.system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": [{"type": "text", "text": self.system_prompt}],
                }
            )
        messages.append(
            {
                "role": "user",
                "content": [{"type": "audio"}, {"type": "text", "text": prompt}],
            }
        )
        return messages

    @torch.no_grad()
    def generate(
        self, speech: Sequence[Any], prompts: Sequence[str]
    ) -> List[str]:
        """Answer one prompt per clip, as a batch.

        Args:
            speech: 16 kHz mono waveforms, one per prompt.
            prompts: The prompts, aligned with ``speech``.

        Returns:
            list[str]: The generated answers, without the prompt and without
            special tokens.

        Raises:
            ValueError: If ``speech`` and ``prompts`` differ in length.
        """
        if len(speech) != len(prompts):
            raise ValueError(
                f"Got {len(speech)} waveforms for {len(prompts)} prompts."
            )
        texts = [
            self.processor.apply_chat_template(
                self.build_conversation(prompt),
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]
        audio = [_to_numpy(waveform) for waveform in speech]
        inputs = self.processor(
            text=texts,
            audio=audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = inputs.to(self.device)
        inputs["input_features"] = inputs["input_features"].to(self.dtype)

        sequences = self.model.thinker.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        answers = sequences[:, inputs["input_ids"].shape[1] :]
        return self.processor.batch_decode(
            answers, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

    def __call__(self, speech: Any, prompt: Any) -> Any:
        """Answer one prompt, or a batch of them.

        Args:
            speech: One waveform, or a list of them.
            prompt: One prompt, or a list aligned with ``speech``.

        Returns:
            str | list[str]: One answer, or one per clip.
        """
        if isinstance(prompt, str):
            return self.generate([speech], [prompt])[0]
        return self.generate(list(speech), list(prompt))


def _to_numpy(waveform: Any) -> np.ndarray:
    """Return a waveform as a 1-D float32 array."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    return np.asarray(waveform, dtype=np.float32).reshape(-1)
