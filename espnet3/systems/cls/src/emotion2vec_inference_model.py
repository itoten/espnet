"""Zero-shot inference with a pre-trained emotion2vec+ classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Tuple

import numpy as np
import torch

from espnet3.systems.cls.src.label_map import (
    gather_class_scores,
    resolve_label_map,
)
from espnet3.systems.esp2_cls.src.scoring_utils import load_class_labels

__all__ = ["Emotion2vecInferenceModel"]


class Emotion2vecInferenceModel:
    """Label one waveform with emotion2vec+ as it was released, without training.

    emotion2vec+ (https://github.com/ddlBoJack/emotion2vec) is a speech
    emotion classifier released with its own head over nine labels (angry,
    disgusted, fearful, happy, neutral, other, sad, surprised, ``<unk>``).
    This class runs that head unchanged and maps its labels onto the recipe's
    classes, so that the ``infer`` and ``measure`` stages score it exactly as
    they score a trained ``ClsModel``. Use it with a standalone inference
    config, which carries its own ``exp_tag`` because no training config is
    involved.

    The model is loaded through FunASR, which is an optional dependency
    (``pip install funasr``).

    Scores are restricted to the recipe's classes: the probabilities of the
    emotion2vec+ labels named in ``label_map`` are gathered in ``token_list``
    order and renormalised to sum to one, and the probability of every other
    label (``other``, ``<unk>``) is discarded. The prediction is therefore
    always one of the recipe's classes, as for a trained model. A recipe class
    that no emotion2vec+ label maps to gets a score of zero and is never
    predicted.

    Args:
        token_list: Token list whose final entry is ``<unk>``; it fixes the
            classes and their order in the score vector.
        label_map: Mapping from an emotion2vec+ label to a recipe class. The
            emotion2vec+ label is the English part of its token
            (``angry`` for ``生气/angry``).
        model_tag: Model name on the hub given by ``hub``.
        hub: ``hf`` (Hugging Face) or ``ms`` (ModelScope).
        device: Torch device string, injected by the inference provider.

    Raises:
        ImportError: If FunASR is not installed.
        ValueError: If ``label_map`` names a label the model does not have or
            a class not in ``token_list``, or maps two labels to one class.

    Examples:
        Selected from a recipe's standalone inference config:

        .. code-block:: yaml

            exp_tag: zeroshot_emotion2vec_plus_large
            model:
              _target_: espnet3.systems.cls.src.emotion2vec_inference_model.Emotion2vecInferenceModel
              model_tag: emotion2vec/emotion2vec_plus_large
              token_list: ${data_dir}/token_list
              label_map:
                angry: angry
                happy: happy
    """

    def __init__(
        self,
        token_list: str | Path,
        label_map: Mapping[str, str],
        model_tag: str = "emotion2vec/emotion2vec_plus_large",
        hub: str = "hf",
        device: str = "cpu",
    ) -> None:
        """Load the model and resolve which of its labels feed each class."""
        try:
            from funasr import AutoModel
        except ImportError as e:
            raise ImportError(
                "Emotion2vecInferenceModel needs FunASR: `pip install funasr`."
            ) from e

        self.classes = load_class_labels(token_list)
        self.model = AutoModel(
            model=model_tag, hub=hub, device=device, disable_update=True
        )
        model_labels = [
            token.split("/")[-1] for token in self.model.kwargs["token_list"]
        ]
        self.source_index = resolve_label_map(
            model_labels, dict(label_map), self.classes, "emotion2vec+"
        )

    @torch.no_grad()
    def __call__(self, speech: Any) -> Tuple[List[int], torch.Tensor, str]:
        """Classify one waveform.

        Args:
            speech: 16 kHz mono waveform as an array or tensor of shape
                ``(samples,)``. FunASR standardises it itself.

        Returns:
            tuple: ``(prediction, scores, label)`` where ``prediction`` holds
            the class index, ``scores`` are per-class probabilities in
            ``token_list`` order and ``label`` is the class name, the same
            as ``ClsInferenceModel``.
        """
        waveform = np.asarray(
            speech.cpu() if isinstance(speech, torch.Tensor) else speech,
            dtype=np.float32,
        )
        result = self.model.generate(
            waveform,
            granularity="utterance",
            extract_embedding=False,
            disable_pbar=True,
        )[0]
        scores = gather_class_scores(result["scores"], self.source_index)

        prediction = int(scores.argmax())
        return [prediction], scores, self.classes[prediction]
