# ANAD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on ANAD alone and scores it on ANAD's own test split, classifying each
utterance into one of three emotions.

ANAD (Arabic Natural Audio Dataset) is a natural Arabic corpus built from eight
recordings of live phone calls on Arabic talk shows. The release cuts them into
1-second chunks (`<call>_<turn> (<chunk>).wav`, e.g. `V7_1 (145).wav`) and
`ANAD.csv` gives each chunk one label: `angry`, `happy` or `surprised`
(written `surprise` here, as in the other recipes).

Measured on the copy used here: 1,334 chunks that have both a wav file and a
label, 0.34 hours in total, 0.05 to 1.0 seconds long (104 are shorter than
0.5 s, 61 shorter than 0.1 s). They are 44.1 kHz stereo as distributed and are
converted to 16 kHz mono. Classes: angry 723, happy 469, surprise 142.

Things to know when reading the scores:

- Every label is constant within a call: calls V1, V3, V8 are `happy`, V2 and
  V6 are `surprise`, V4, V5 and V7 are `angry`. A model can therefore learn the
  call (speaker, line, topic) instead of the emotion.
- No speaker id is published. The splits below keep a call's turn together but
  do not hold speakers out.
- The release is not fully consistent: `ANAD.csv` lists 1,383 chunks, but 49 of
  them have no wav file, and 86 wav chunks (plus 8 full-call recordings, not
  used) have no label in the table. Only chunks with both are used.

## Provenance

Kaggle dataset `suso172/arabic-natural-audio-dataset` (version 11, 2017-12-01),
licence shown on Kaggle: CC BY-NC-SA 4.0 (non-commercial, share-alike). The
required citation was not checked; see the Kaggle page.

Kaggle requires a login, so the recipe does not download. Download the zip and
extract only the audio and the table:

```bash
unzip arabic-natural-audio-dataset.zip '*.wav' '*.csv' -x 'V?.wav'
```

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `ANAD` — the directory holding `1sec_segmented_part{1,2,3}/` and `ANAD.csv`
- `ANAD_OUTPUT` — where the converted audio and manifests are written

## Download

The zip (587 MB) took 24 seconds to download. `create_dataset`,
`remove_long_short` and `prepare_labels` together took about 90 seconds.

## Splits

No partition is published and the labels cannot be tied to speakers. Chunks of
the same call and turn (`V7_1`) are kept in one split, and within each label the
groups are assigned largest first to the split furthest below its target
(70/10/20, `split_seed` in `dataset/config.yaml`).

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 937 | 894 |
| valid | 131 | 107 |
| test | 266 | 266 (not filtered) |

## Quick start

Run the three commands separately. With several GPUs, `train` starts one
process per GPU by re-running its own command, so any other stage listed in the
same command would run once per GPU as well.

```bash
python run.py --stages create_dataset remove_long_short prepare_labels \
    collect_stats --training_config conf/training.yaml
python run.py --stages train \
    --training_config conf/training.yaml
python run.py --stages infer measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```
