"""Qwen2.5-Omni system: prompted inference with the released model.

Qwen2.5-Omni (https://github.com/QwenLM/Qwen2.5-Omni) is used as released,
without training: a recipe gives it audio and a text prompt, reads its text
answer and scores it. So the system declares only ``infer`` and ``measure``.
"""

import copy
import logging
from typing import ClassVar

from omegaconf import OmegaConf

from espnet3.contract.stages import StageSpec
from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


class Qwen2_5OmniSystem(BaseSystem):
    """System for recipes that prompt Qwen2.5-Omni without training it.

    Only ``infer`` and ``measure`` are declared; there is nothing to prepare
    or train. A recipe runs standalone, with an inference config that carries
    its own ``exp_tag``, and reads data another recipe has prepared.

    ``infer`` accepts ``only`` in the inference config: a list of test set
    names (or one space-separated string) to run instead of every entry of
    ``dataset.test``. Several processes, one per GPU, can then share the test
    sets of one config, each writing its own ``inference_dir/<test_name>``.

    Examples:
        >>> [s.name for s in Qwen2_5OmniSystem.stages]
        ['infer', 'measure']
    """

    stages: ClassVar[tuple[StageSpec, ...]] = (
        StageSpec(name="infer", config="inference", log_dir="inference_dir"),
        StageSpec(name="measure", config="metrics", log_dir="inference_dir"),
    )

    def infer(self, *args, **kwargs):
        """Run inference on the configured test sets, or on ``only`` of them.

        Raises:
            ValueError: If ``only`` names a test set that ``dataset.test``
                does not define.
        """
        self._reject_stage_args("infer", args, kwargs)
        only = _parse_only(OmegaConf.select(self.inference_config, "only"))
        if only is None:
            return super().infer()

        test_sets = self.inference_config.dataset.test
        names = [entry.name for entry in test_sets]
        unknown = sorted(set(only) - set(names))
        if unknown:
            raise ValueError(
                f"inference_config.only names test sets that dataset.test does "
                f"not define: {unknown}"
            )
        logger.info("Running only these test sets: %s", only)

        # A deep copy: `dataset` is a nested node, which a shallow copy would
        # share with the original config.
        config = self.inference_config
        self.inference_config = copy.deepcopy(config)
        self.inference_config.dataset.test = [
            entry for entry in test_sets if entry.name in only
        ]
        try:
            return super().infer()
        finally:
            self.inference_config = config


def _parse_only(value) -> list[str] | None:
    """Return the test set names in ``only``, or ``None`` to run them all.

    Args:
        value: ``None``, an empty string, a space-separated string, or a list.

    Returns:
        list[str] | None: The names, or ``None`` when ``value`` names none.
    """
    if value is None:
        return None
    if isinstance(value, str):
        names = value.split()
    else:
        names = [str(name) for name in value]
    return names or None
