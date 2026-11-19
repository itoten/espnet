"""Zero-shot speech emotion recognition with the VoxEmo protocol.

VoxEmo (Zhang, Chou, Narayanan and Hain, "VoxEmo: Benchmarking Speech
Emotion Recognition with Speech LLMs", arXiv:2603.08936) evaluates a speech
LLM with five prompts built from the blocks of its Table 3, greedy decoding,
and a ``FINAL_LABEL`` field matched case-insensitively against the corpus's
closed label set; an answer that cannot be matched counts as wrong. Its
prompt ensemble turns the five answers into a distribution in which each
failed answer is a uniform vote of ``1/C`` (its Eq. 2).

The texts below are those of Table 3. The paper does not release its toolkit
and leaves these details open; this module assumes:

- The blocks are joined by newlines in Table 3's order: task description,
  emotion categories, instruction blocks, then the response format,
  introduced by a ``Response format:`` line.
- The Direct decision block is used by the Direct prompt only; the other
  prompts add their blocks (T, A, R) instead.
- ``{labels}`` is the label set joined by ``", "``.
- ``FINAL_LABEL`` is the last line starting with ``FINAL_LABEL:`` in the
  answer; its value is matched after stripping whitespace and the
  characters ``<>[]()"'*.`` around it.
- A tie in the ensemble goes to the label that comes first in the label set.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "PROMPT_BLOCKS",
    "VoxEmoClassifier",
    "build_prompt",
    "ensemble_votes",
    "parse_final_label",
]

TASK_DESCRIPTION = (
    "You are a speech emotion recognition system. You are given an audio clip."
)
EMOTION_CATEGORIES = "Choose exactly one emotion label from this closed set: {labels}."
INSTRUCTIONS = {
    "direct": "Decide the emotion directly from the audio.",
    "transcript": (
        "Transcribe the spoken content from the audio as accurately as possible."
    ),
    "caption": (
        "Describe acoustic and paralinguistic cues you hear (e.g., pitch, "
        "loudness, speaking rate, pauses, voice quality, non-speech events)."
    ),
    "reasoning": (
        "Explain how the transcript content and the acoustic cues together "
        "support your final choice."
    ),
}
RESPONSE_FIELDS = {
    "transcript": "ASR_TRANSCRIPT: <transcript>",
    "caption": "ACOUSTIC_CAPTION: <description>",
    "reasoning": "REASONING: <reasoning>",
}
FINAL_LABEL_FORMAT = "FINAL_LABEL: <one label from {labels}>"

# The five prompts of VoxEmo's Tables 4-5 (Direct, +T, +A, +T+A, +T+A+R),
# by the name used in output files and metric keys.
PROMPT_BLOCKS: Dict[str, tuple] = {
    "direct": ("direct",),
    "t": ("transcript",),
    "a": ("caption",),
    "t_a": ("transcript", "caption"),
    "t_a_r": ("transcript", "caption", "reasoning"),
}

_FINAL_LABEL = re.compile(r"^\s*FINAL_LABEL\s*:(.*)$", re.MULTILINE)
_STRIP = " \t<>[]()\"'*."


def build_prompt(variant: str, labels: Sequence[str]) -> str:
    """Return the text of one VoxEmo hard-label prompt.

    Args:
        variant: A key of ``PROMPT_BLOCKS``.
        labels: The corpus's closed label set.

    Returns:
        str: The prompt.

    Raises:
        KeyError: If ``variant`` is unknown.

    Examples:
        >>> print(build_prompt("direct", ["neutral", "angry"]))
        You are a speech emotion recognition system. You are given an audio clip.
        Choose exactly one emotion label from this closed set: neutral, angry.
        Decide the emotion directly from the audio.
        Response format:
        FINAL_LABEL: <one label from neutral, angry>
    """
    blocks = PROMPT_BLOCKS[variant]
    joined = ", ".join(labels)
    lines = [TASK_DESCRIPTION, EMOTION_CATEGORIES.format(labels=joined)]
    lines += [INSTRUCTIONS[block] for block in blocks]
    lines.append("Response format:")
    lines += [RESPONSE_FIELDS[block] for block in blocks if block in RESPONSE_FIELDS]
    lines.append(FINAL_LABEL_FORMAT.format(labels=joined))
    return "\n".join(lines)


def parse_final_label(answer: str, labels: Sequence[str]) -> Optional[str]:
    """Return the label named by an answer's ``FINAL_LABEL`` field, if any.

    Args:
        answer: The model's text answer.
        labels: The corpus's closed label set.

    Returns:
        str | None: The matching label as spelled in ``labels``, or ``None``
        when the answer has no ``FINAL_LABEL`` line or its value is not in
        the set (a parse failure).

    Examples:
        >>> parse_final_label("REASONING: ...\\nFINAL_LABEL: Angry", ["angry"])
        'angry'
        >>> parse_final_label("FINAL_LABEL: <sad>.", ["angry", "sad"])
        'sad'
        >>> parse_final_label("The speaker sounds angry.", ["angry"]) is None
        True
    """
    matches = _FINAL_LABEL.findall(answer)
    if not matches:
        return None
    value = matches[-1].strip(_STRIP).lower()
    for label in labels:
        if label.lower() == value:
            return label
    return None


def ensemble_votes(
    predictions: Sequence[Optional[str]], labels: Sequence[str]
) -> np.ndarray:
    """Return VoxEmo's prompt-ensemble distribution (Eq. 2) over ``labels``.

    Each prompt votes for its predicted class; a failed prompt (``None``)
    gives ``1/C`` to every class. The votes are divided by the number of
    prompts.

    Args:
        predictions: One prediction per prompt, ``None`` for a failure.
        labels: The corpus's closed label set (``C`` classes).

    Returns:
        np.ndarray: Probabilities aligned with ``labels``, summing to one.

    Examples:
        >>> ensemble_votes(["sad", None, "sad", "angry", None], ["angry", "sad"])
        array([0.4, 0.6])
    """
    votes = np.zeros(len(labels))
    for prediction in predictions:
        if prediction is None:
            votes += 1.0 / len(labels)
        else:
            votes[list(labels).index(prediction)] += 1.0
    return votes / len(predictions)


class VoxEmoClassifier:
    """Label an utterance with a speech LLM, by the VoxEmo protocol.

    For every prompt in ``variants`` the generator answers once (greedy
    decoding is the generator's job); the answers are parsed and combined by
    the prompt ensemble.

    Args:
        generator: Partially instantiated generator; called as
            ``generator(device=device)`` so that it is built on the
            provider's device. The result must offer
            ``generate(speech, prompts) -> list[str]``, e.g.
            ``Qwen2_5OmniInference``.
        variants: Keys of ``PROMPT_BLOCKS`` to run; all five by default. The
            ensemble needs all five to be VoxEmo's.
        device: Torch device string, injected by the inference provider.

    Examples:
        .. code-block:: yaml

            model:
              _target_: src.voxemo.VoxEmoClassifier
              generator:
                _target_: espnet3.systems.qwen2_5_omni.inference.Qwen2_5OmniInference
                _partial_: true
                max_new_tokens: 512
            input_key: [speech, label_set]
    """

    def __init__(
        self,
        generator: Callable[..., Any],
        variants: Sequence[str] = tuple(PROMPT_BLOCKS),
        device: str = "cpu",
    ) -> None:
        """Build the generator on ``device`` and check the prompt names."""
        unknown = sorted(set(variants) - set(PROMPT_BLOCKS))
        if unknown:
            raise ValueError(
                f"Unknown VoxEmo prompts {unknown}; choose from {list(PROMPT_BLOCKS)}"
            )
        self.variants = list(variants)
        self.generator = generator(device=device)

    def __call__(self, speech: Any, label_set: Any) -> Any:
        """Label one utterance, or a batch of them.

        Args:
            speech: One waveform, or a list of them.
            label_set: The corpus's labels joined by ``", "`` (the dataset's
                ``label_set`` field), or a list of them aligned with
                ``speech``.

        Returns:
            dict | list[dict]: Per utterance, ``answers`` (the raw answer per
            prompt), ``predictions`` (the parsed label per prompt, ``None``
            for a failure), ``labels`` (the label set), ``distribution`` (the
            ensemble's probabilities aligned with ``labels``) and
            ``prediction`` (its most likely label).
        """
        if isinstance(label_set, str):
            return self.classify([speech], [label_set])[0]
        return self.classify(list(speech), list(label_set))

    def classify(
        self, speech: List[Any], label_sets: List[str]
    ) -> List[Dict[str, Any]]:
        """Run every prompt over a batch and combine the answers per utterance."""
        label_lists = [[s.strip() for s in ls.split(",")] for ls in label_sets]
        results = [
            {"answers": {}, "predictions": {}, "labels": labels}
            for labels in label_lists
        ]
        for variant in self.variants:
            prompts = [build_prompt(variant, labels) for labels in label_lists]
            answers = self.generator.generate(speech, prompts)
            for result, answer in zip(results, answers):
                result["answers"][variant] = answer
                result["predictions"][variant] = parse_final_label(
                    answer, result["labels"]
                )
        for result in results:
            distribution = ensemble_votes(
                list(result["predictions"].values()), result["labels"]
            )
            result["distribution"] = distribution
            result["prediction"] = result["labels"][int(np.argmax(distribution))]
        return results
