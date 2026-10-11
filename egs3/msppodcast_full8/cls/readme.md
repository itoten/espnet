# MSP-Podcast (full, 8 classes) speech emotion recognition

Trains `cls` on MSP-Podcast's Train partition alone (angry, disgust,
fear, happy, neutral, sad, surprise, contempt), validates and early-stops on
its Development partition, and scores the best checkpoint against MSP-Podcast's
own Test1/Test2 plus the same 5 out-of-domain corpora
`egs3/multicorpus_ood20/cls` uses (EmoV-DB, IEMOCAP, TESS, SAVEE,
RAVDESS, each read there in full).

This recipe exists to compare against `multicorpus_ood20`: does a single
large, naturalistic corpus (MSP-Podcast, ~137k training utterances) or ten
smaller, mostly acted corpora used together (~55k utterances) generalize
better to corpora neither has trained on?

## How this differs from `egs3/msppodcast/cls`

- **Contempt is kept as an eighth class.** The other recipe drops it because
  none of the corpora it is compared against there has a contempt label;
  that reason does not apply here, so all the corpus's usable labels are
  used. `O` (free-text) and `X` (no annotator majority) are still dropped --
  neither is a hard emotion label to train on.
- **Only Train and Development are used for training/validation.**
  Test1/Test2 are prepared too, but only as extra test sets, alongside the
  5 out-of-domain corpora.
- **Converted audio is shared**, not duplicated: both recipes resample into
  the same `$MSPPODCAST_OUTPUT/wav/`, keyed by the same `utt_id`. This
  recipe only converts the contempt-labelled recordings the other one skips.
  Neither recipe's manifests, token list or class weights are shared.
- **Real validation and early stopping**, unlike `multicorpus_ood20`'s
  every-epoch, no-validation design: MSP-Podcast is far larger than the
  other corpora in this comparison, so a held-out slice of it is a
  reasonable proxy for its own held-out data.

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

Comparing against `multicorpus_ood20`'s 20-class model needs more than this
raw score, since the two models can predict different label sets: the
labels the two share (angry, disgust, fear, happy, neutral, sad, surprise)
need to be scored separately from the labels only one of them knows. That
comparison is computed from `score.scp`/`ref.scp` after inference, not by
this recipe.
