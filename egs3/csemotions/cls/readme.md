# CSEMOTIONS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on CSEMOTIONS alone and scores it on CSEMOTIONS's own test split, classifying each
utterance into one of seven emotions.

[CSEMOTIONS](https://huggingface.co/datasets/ATH-MaaS/CSEMOTIONS) (released with the
Marco-Voice technical report, [arXiv 2508.02038](https://arxiv.org/abs/2508.02038)) is
a Mandarin Chinese acted corpus. Ten voice actors (5 male, 5 female, per the dataset
card) read one emotional sentence per clip in seven emotions: `angry`, `fear`
(`fearful` in the source), `happy`, `neutral`, `playfulness`, `sad` and `surprise`.
Measured on the downloaded data: 4,160 clips, 10.20 hours, 1.75 to 34.2 seconds long,
all 48 kHz mono 16-bit PCM wav (the dataset card says 24-bit). Counts per emotion:
angry 769, fear 623, playfulness 621, happy 603, sad 546, surprise 508, neutral 490.
Speakers are named `female001`-`female005` and `male001`-`male005`; each has 323 to 473
clips. No duplicate clips were found.

The label is the `emotion` column of the parquet files; `dataset/config.yaml` maps it
to the manifest label. `playfulness` is kept as its own class, since it has no
counterpart in the other recipes. The clips are converted to 16 kHz mono with ffmpeg
by `create_dataset`.

## Provenance

Hugging Face dataset `ATH-MaaS/CSEMOTIONS` (the repository formerly named
`AIDC-AI/CSEMOTIONS` redirects there), pinned to commit
`19498e62e6745cf9ec9b69a44b826caa467eac31`. It is not gated. The licence is
Apache-2.0. The repository's `NOTICE` file states that the dataset incorporates
Emotional-Speech-Data (ESD, MIT licence), noting that "this database can only be used
for research purpose"; which clips come from ESD is not stated, so check this before
any non-research use.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `CSEMOTIONS` — the directory holding the eight parquet shards, or where to download them to
- `CSEMOTIONS_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the eight parquet shards (3.08 GB in total) and checks each
SHA-256. Measured on one run: about 2 minutes to download; `create_dataset`
(parquet to 16 kHz wav) took about 249 seconds, `remove_long_short` 18 seconds and
`prepare_labels` under 1 second.

## Splits

The corpus publishes no partition. Whole speakers are held out
(`n_test_speakers`, `n_valid_speakers`, `split_seed` in `dataset/config.yaml`): with
seed 0, `male003` and `male004` form the test set, `female002` the valid set, and the
other seven speakers the training set. Scores are therefore speaker-independent, but
the sentences may be shared with the training set (not checked), and the test speakers
are both male.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 2,932 | 2,917 |
| valid | 378 | 378 |
| test | 850 | 850 (not filtered) |

`remove_long_short` drops the 15 train clips longer than 30 s.

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
