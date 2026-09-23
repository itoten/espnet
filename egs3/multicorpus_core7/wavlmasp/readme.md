# Multi-corpus speech emotion recognition (core7)

Trains `wavlmasp` (fine-tuned [WavLM
Large](https://github.com/microsoft/unilm/tree/master/wavlm) + attentive
statistics pooling + a fully connected head, the [Interspeech 2025 SER in
Naturalistic Conditions challenge](https://www.isca-archive.org/interspeech_2025/naini25_interspeech.html)
baseline) on the union of ten corpora — CREMA-D, EMNS, EmoDB, Emozionalmente,
eNTERFACE'05, JL-Corpus, MELD, Quechua Collao, SUBESCO and Thorsten — and
scores it on each corpus's own test split plus MSP-Podcast Test1/Test2, which
this recipe never trains on.

Labels are restricted to the **core7** set shared with the IS2025 baseline:
`angry`, `sad`, `happy`, `surprise`, `fear`, `disgust`, `neutral`. Every other
label a corpus carries (`excited`, `boredom`, `calm`, `sleepy`, `amused`,
`drunk`, `whisper`, `sarcastic`, JL-Corpus's secondary emotions, and
MSP-Podcast's `contempt`) is dropped, row and all — see
`espnet_work/label_inventory.md` for the full per-corpus breakdown. The
sibling recipe `egs3/multicorpus_full20/wavlmasp` keeps every one of those
labels instead of dropping them; the two are meant to be compared against
each other, not run in isolation. See `espnet_work/multicorpus_ser_plan.md`.

## How the merge works

Each corpus is prepared by its own `esp2_cls` recipe (download, conversion,
its own train/valid/test split) exactly as if it were used on its own —
`dataset/builder.py` here only reads those recipes' manifests, applies
`dataset/config.yaml`'s `label_map`, and writes:

- one merged `manifest/train.tsv` / `manifest/valid.tsv` across all ten
  training corpora (the file each row came from is not recorded; per-corpus
  kept/dropped counts are logged instead)
- one unmerged, label-mapped `manifest/<corpus>.tsv` per corpus's own test
  split, plus `manifest/msppodcast_test.tsv` / `manifest/msppodcast_test2.tsv`

so every downstream stage (`remove_long_short`, `prepare_labels`,
`collect_stats`) still sees a single manifest per split, unmodified from the
single-corpus recipes.

Each corpus's own `<CORPUS>_OUTPUT` environment variable must already point
at that corpus's prepared data (or be unset, in which case that corpus's own
recipe default applies and it prepares itself on first use).
`MULTICORPUS_CORE7_OUTPUT` controls where this recipe's own merged manifests
are written; it defaults to `data/` under this recipe.

## Quick start

```bash
# 1) Merge manifests, filter by duration, build the label list
python run.py --stages create_dataset remove_long_short prepare_labels \
    --training_config conf/training.yaml

# 2) Collect feature statistics
python run.py --stages collect_stats \
    --training_config conf/training.yaml

# 3) Train
python run.py --stages train \
    --training_config conf/training.yaml

# 4) Infer (all 12 test sets)
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 5) Score
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

## Notes

- `dataloader`/`trainer` settings (`batch_size: 8`, `accumulate_grad_batches: 4`,
  `max_epochs: 30`) are carried over from the single-corpus `emodb/wavlmasp`
  smoke-test recipe as a starting point, not tuned for this data volume; with
  roughly 32k training utterances the optimizer sees far more updates per
  epoch than EmoDB's ~350 total, so this may warrant revisiting once a first
  run completes.
- `class_weights` (inverse label frequency) corrects for class imbalance
  within the merged set, but not for corpus imbalance: MELD alone contributes
  roughly 40% of the training rows. Per-corpus test scores (this recipe
  reports all 12 separately) are the way to check whether results are being
  driven by one corpus.
