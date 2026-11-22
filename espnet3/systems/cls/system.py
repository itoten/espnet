"""Classification system over pre-trained speech encoders.

The stages are those of :class:`~espnet3.systems.esp2_cls.system.ESP2CLSSystem`
(``remove_long_short`` and ``prepare_labels`` between ``create_dataset`` and
``collect_stats``). This system changes two of them for the models in
``espnet3.systems.cls.src``: ``collect_stats`` runs without the waveform
normalizer, and ``prepare_labels`` can also write class weights for the loss.
"""

import logging
from collections import Counter
from pathlib import Path
from typing import List

from omegaconf import OmegaConf

from espnet3.systems.esp2_cls.system import ESP2CLSSystem

logger = logging.getLogger(__name__)


class ClsSystem(ESP2CLSSystem):
    """Classification system for ``ClsModel`` and its frontends.

    ``esp2_cls`` trains ESPnet2's classification task; this system trains the
    models in ``espnet3.systems.cls.src`` (a pre-trained or from-scratch
    encoder, attentive statistics pooling and a classification head). The
    data stages are shared with ``esp2_cls``.

    Examples:
        >>> [s.name for s in ClsSystem.stages[:4]]
        ['create_dataset', 'remove_long_short', 'prepare_labels', 'collect_stats']
    """

    def collect_stats(self, *args, **kwargs):
        """Collect statistics without the waveform normalizer.

        ``model.frontend.normalize`` loads ``feats_stats.npz``, which this
        stage is about to create. It is dropped from a copy of the config, so
        the later stages still see it.
        """
        config = self.training_config
        frontend = OmegaConf.select(config, "model.frontend")
        if frontend is None or "normalize" not in frontend:
            return super().collect_stats(*args, **kwargs)

        self.training_config = config.copy()
        self.training_config.model.frontend.pop("normalize")
        try:
            return super().collect_stats(*args, **kwargs)
        finally:
            self.training_config = config

    def prepare_labels(self, *args, **kwargs):
        """Build the label list, and optionally the class weights of the loss.

        Everything of :meth:`ESP2CLSSystem.prepare_labels` applies. In
        addition, ``training_config.prepare_labels.weights_filename`` writes
        one inverse-frequency weight per class next to the label list, for
        ``ClsModel(class_weights=...)``:

        .. code-block:: yaml

            prepare_labels:
              save_path: data
              filename: token_list
              weights_filename: class_weights
              add_symbol:
                - "<unk>:-1"

        Raises:
            RuntimeError: Everything :meth:`ESP2CLSSystem.prepare_labels`
                raises.
            ValueError: If ``weights_filename`` is set and an ``add_symbol``
                entry was inserted anywhere but at the end, since the inserted
                symbol would then be a class without any example.
        """
        super().prepare_labels(*args, **kwargs)

        prepare_labels_config = self.training_config.prepare_labels
        weights_filename = prepare_labels_config.get("weights_filename")
        if not weights_filename:
            return

        save_dir = Path(prepare_labels_config.save_path)
        token_list = save_dir / prepare_labels_config.filename
        labels = [
            line.strip()
            for line in token_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest_path = Path(
            prepare_labels_config.get("manifest_path", "data/manifest/train.tsv")
        ).resolve()
        self._write_class_weights(
            self._count_labels(manifest_path),
            labels,
            save_dir / str(weights_filename),
        )

    @staticmethod
    def _count_labels(manifest_path: Path) -> Counter:
        """Count the labels of the third manifest column, as prepare_labels."""
        counter: Counter = Counter()
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    counter.update(parts[2].split())
        return counter

    @staticmethod
    def _write_class_weights(
        counter: Counter, labels: List[str], output_file: Path
    ) -> None:
        """Write one inverse-frequency weight per class, in token_list order.

        The model's classes are every entry of the token list but the last
        (``<unk>``, see ``load_class_labels``). The weight of class ``c`` is
        ``N / (C * n_c)``, matching ``sklearn``'s ``class_weight="balanced"``.

        Args:
            counter: Occurrence count per label in the training manifest.
            labels: Final token list, ending with ``<unk>``.
            output_file: Destination path.

        Raises:
            ValueError: If a class of the model has no training example, which
                happens when ``add_symbol`` inserted a symbol before the end.

        Examples:
            >>> ClsSystem._write_class_weights(
            ...     Counter({"a": 3, "b": 1}), ["a", "b", "<unk>"], Path("w")
            ... )  # writes 0.6666666667 and 2
        """
        classes = labels[:-1]
        missing = [label for label in classes if counter[label] == 0]
        if missing:
            raise ValueError(
                f"Classes {missing} have no training example, so they have no "
                "class weight. Insert add_symbol entries at the end "
                '(e.g. "<unk>:-1") when weights_filename is set.'
            )
        total = sum(counter[label] for label in classes)
        weights = [total / (len(classes) * counter[label]) for label in classes]

        with open(output_file, "w", encoding="utf-8") as f:
            for weight in weights:
                f.write(f"{weight:.10g}\n")

        logger.info(
            "prepare_labels: wrote %d class weights (%.3g..%.3g) -> %s",
            len(weights),
            min(weights),
            max(weights),
            output_file,
        )
