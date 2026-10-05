"""Classification system for EEG epochs.

This module adds the ``prepare_labels`` stage on top of the base system.
"""

import logging
from collections import Counter
from pathlib import Path

from omegaconf import DictConfig

from espnet3.systems.base.system import BaseSystem

logger = logging.getLogger(__name__)


class EegClsSystem(BaseSystem):
    """System that classifies EEG epochs.

    This system adds one stage to the base stages:
      - ``prepare_labels``: build the label list consumed as ``token_list``

    Additional stage-log mappings:

    .. list-table::
       :header-rows: 1

       * - Stage
         - Path reference
       * - ``prepare_labels``
         - ``training_config.prepare_labels.save_path``
    """

    def __init__(
        self,
        training_config: DictConfig | None = None,
        inference_config: DictConfig | None = None,
        metrics_config: DictConfig | None = None,
        publication_config: DictConfig | None = None,
        stage_log_mapping: dict | None = None,
        demo_config: DictConfig | None = None,
    ) -> None:
        """Initialize the system with optional stage configs.

        Args:
            training_config: Training configuration.
            inference_config: Inference configuration.
            metrics_config: Measurement configuration.
            publication_config: Publication configuration for model packing
                and upload stages.
            stage_log_mapping: Optional per-stage log directory overrides.
            demo_config: Demo configuration for demo packing and upload
                stages.
        """
        super().__init__(
            training_config=training_config,
            inference_config=inference_config,
            metrics_config=metrics_config,
            publication_config=publication_config,
            stage_log_mapping={
                "prepare_labels": "training_config.prepare_labels.save_path",
                **(stage_log_mapping or {}),
            },
            demo_config=demo_config,
        )

    def prepare_labels(self, *args, **kwargs):
        """Build the label list used as ``token_list`` by the classifier.

        Labels are collected from the third column of the training manifest,
        split on whitespace so that multi-label utterances contribute every
        label they carry. The list is ordered by descending frequency, which
        matches stage 4 of the ESPnet2 ``cls.sh`` recipe.

        Under ``training_config.prepare_labels``, configure:

        .. list-table::
           :header-rows: 1

           * - Field
             - Description
           * - ``save_path``
             - Directory to write the label list into.
           * - ``filename``
             - Label list file name, such as ``token_list``.
           * - ``manifest_path``
             - Training manifest to read. Defaults to
               ``data/manifest/train.tsv``.
           * - ``add_symbol``
             - Optional ``"<symbol>:<index>"`` entries; a negative index
               counts from the end.

        Example:
            .. code-block:: yaml

                prepare_labels:
                  save_path: data
                  filename: token_list
                  manifest_path: data/manifest/train.tsv
                  add_symbol:
                    - "<unk>:-1"

        Raises:
            RuntimeError: If required configuration is missing, the manifest
                is not found, an ``add_symbol`` entry is malformed, or the
                manifest holds no label.
        """
        self._reject_stage_args("prepare_labels", args, kwargs)
        logger.info("EegClsSystem.prepare_labels(): starting label list creation")

        prepare_labels_config = self._get_required_config(
            self.training_config,
            "prepare_labels",
            "training_config.prepare_labels must be set for prepare_labels stage.",
        )
        save_path_str = self._get_required_config(
            prepare_labels_config,
            "save_path",
            "training_config.prepare_labels.save_path must be set for "
            "prepare_labels stage.",
        )
        filename = self._get_required_config(
            prepare_labels_config,
            "filename",
            "training_config.prepare_labels.filename must be set "
            "(e.g. 'token_list'); save_path is the output directory.",
        )
        save_dir = Path(save_path_str)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / filename

        manifest_path = Path(
            prepare_labels_config.get("manifest_path", "data/manifest/train.tsv")
        ).resolve()
        if not manifest_path.exists():
            raise RuntimeError(
                f"Manifest file not found for label list creation: "
                f"{manifest_path}. Please ensure the manifest file is "
                "generated and the path is correct."
            )

        counter = Counter()
        n_unlabelled = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t")
                labels = parts[2].split() if len(parts) >= 3 else []
                if not labels:
                    # The row still reaches training, so say so rather than
                    # letting it disappear from the label inventory in silence.
                    n_unlabelled += 1
                    continue
                for label in labels:
                    counter[label] += 1

        if n_unlabelled:
            logger.warning(
                "prepare_labels: %d row(s) in %s carry no label",
                n_unlabelled,
                manifest_path,
            )

        if not counter:
            raise RuntimeError(f"No label found in manifest: {manifest_path}")

        # Sort by the number of occurrences in descending order.
        labels = [label for label, _ in sorted(counter.items(), key=lambda x: -x[1])]

        add_symbol = [str(item) for item in prepare_labels_config.get("add_symbol", [])]
        for symbol_and_id in add_symbol:
            # e.g. symbol="<unk>:-1"
            try:
                symbol, idx = symbol_and_id.split(":")
                idx = int(idx)
            except ValueError:
                raise RuntimeError(f"Format error: e.g. '<unk>:-1': {symbol_and_id}")
            symbol = symbol.strip()

            # e.g. idx=0  -> insert as the first symbol
            # e.g. idx=-1 -> append as the last symbol
            if idx < 0:
                idx = len(labels) + 1 + idx
            labels.insert(idx, symbol)

        with open(output_file, "w", encoding="utf-8") as f:
            for label in labels:
                f.write(f"{label}\n")

        logger.info(
            "prepare_labels: wrote %d labels from %d occurrences -> %s",
            len(labels),
            sum(counter.values()),
            output_file,
        )

        # Counting the labels here keeps the class weights next to the label
        # list they must line up with.
        weights_filename = prepare_labels_config.get("weights_filename")
        if weights_filename:
            self._write_class_weights(counter, labels, save_dir / str(weights_filename))

    @staticmethod
    def _write_class_weights(counter, labels, output_file: Path) -> None:
        """Write one inverse-frequency weight per class, in token_list order.

        The weight of class ``c`` is ``N / (C * n_c)``, matching
        ``sklearn``'s ``class_weight="balanced"``. Symbols that `add_symbol`
        inserted carry no count and are left out, so the file lines up with
        the classes the model predicts.

        Args:
            counter: Occurrence count per label.
            labels: Final token list, including any inserted symbols.
            output_file: Destination path.
        """
        classes = [label for label in labels if label in counter]
        total = sum(counter.values())
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
