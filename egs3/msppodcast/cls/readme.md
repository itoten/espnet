# MSP-Podcast speech emotion recognition (8 classes)

Trains `cls` on MSP-Podcast's Train partition alone (angry, contempt,
disgust, fear, happy, neutral, sad, surprise), validates and early-stops on
its Development partition, and scores the best checkpoint against MSP-Podcast's
own Test1/Test2 plus the same 5 out-of-domain corpora
`egs3/multicorpus_ood20/cls` uses (EmoV-DB, IEMOCAP, TESS, SAVEE,
RAVDESS, each read there in full).

MSP-Podcast's own standard is eight classes, so contempt is a class like any
other. `O` (free-text) and `X` (no annotator majority) are dropped: neither is
a hard emotion label to train on.

Compared with `multicorpus_ood20`, this recipe asks whether a single large,
naturalistic corpus (MSP-Podcast, ~137k training utterances) generalizes
better to unseen corpora than ten smaller, mostly acted corpora used together
(~55k utterances). It also differs in using real validation and early
stopping: MSP-Podcast is far larger than the other corpora, so a held-out
slice of it is a reasonable proxy for its own held-out data.

All outputs (manifests, token list, class weights, 16 kHz WAV files) go to
`$MSPPODCAST_OUTPUT`; the Train and Development partitions are used for
training and validation, Test1/Test2 only as extra test sets.

## Running

```bash
python run.py --stages create_dataset remove_long_short prepare_labels collect_stats \
    --training_config conf/training.yaml
python run.py --stages train --training_config conf/training.yaml
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`dataset.test` (in both `conf/training.yaml` and `conf/inference.yaml`) reads
`multicorpus_ood20`'s already-built manifests via `data_src`, so both models
are scored on byte-identical evaluation data. It is read-only: running this
recipe never writes into `multicorpus_ood20`'s directory.

## Scoring

`conf/metrics.yaml` computes WA, UA and MacroF1, without `token_list`: the
evaluation corpora carry labels this model cannot predict (EmoV-DB's
`amused`/`sleepy`, IEMOCAP's `excited`, RAVDESS's `calm`), and passing
`token_list` would make UA/MacroF1 raise on those instead of scoring them as
always-wrong. mAP and AUC are not computed here since they require
`token_list` with no such fallback.
