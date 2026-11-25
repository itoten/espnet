# sinhala-emotional-tts-dataset speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on sinhala-emotional-tts-dataset alone and scores it on a held-out part of the
same release, classifying each recording into one of its five labels.

[sinhala-emotional-tts-dataset](https://huggingface.co/datasets/jithara/sinhala-emotional-tts-dataset)
is published on Hugging Face with a cc-by-4.0 license tag. Its dataset card has
no description: who recorded it, how, and how the labels were given are not
stated. The release holds one folder per label (`anger`, `fear`, `happy`,
`neutral`, `sad`) with 100 files each, and a `metadata.csv`.

The label of a recording is the folder it is in; all five labels are kept with
their names as in the release. `metadata.csv` is not read (see "The data").
`create_dataset` converts every file to 16 kHz mono 16-bit PCM with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `SINHALA_EMOTIONAL_TTS_DATASET` — the directory holding the label folders, or where to download them to
- `SINHALA_EMOTIONAL_TTS_DATASET_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the five label folders, `metadata.csv` and `README.md`
from Hugging Face at a pinned commit (`e245128188c2f09ff58cb93344cc4519c9535d25`)
with `huggingface_hub.snapshot_download`; no login is needed. Files already
present are not fetched again.

## The data

TODO: filled after the first build (counts of the release).

## Splits

The release publishes no partition and does not identify speakers. Files are
split 70/10/20 at random within each label (`train_ratio`, `valid_ratio`,
`test_ratio`, `split_seed` in `dataset/config.yaml`).

## License and citation

The Hugging Face card's license field is cc-by-4.0; the card gives no other terms
and no citation.

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
