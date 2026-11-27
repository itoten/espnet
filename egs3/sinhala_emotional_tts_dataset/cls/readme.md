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

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-25 against the Hugging Face dataset `jithara/sinhala-emotional-tts-dataset` (repository revision `e245128188c2f09ff58cb93344cc4519c9535d25`, last modified 2026-03-01). All counts cover every recording in the release, before any filtering by this recipe.

### Access

Hugging Face, direct download, no sign-in.

### Size of the release

| recordings | speakers | total hours |
|---|---|---|
| 500 | not stated | 0.66 |

How these were counted: every file in the five label folders; hours from the audio headers (4.01 to 5.85 s). 499 files are 16 kHz mono WAV named `<NN>_<MMM>_<code>.wav.wav`; one, `happy/01_100_hap.wav.m4a`, is AAC at 48 kHz stereo. Every file name starts with `01`, and the middle field runs from 01 to 100 in each folder.

### Files per label

| label folder (code in the file name) | recordings | hours |
|---|---|---|
| anger (ang) | 100 | 0.135 |
| fear (fea) | 100 | 0.133 |
| happy (hap) | 100 | 0.129 |
| neutral (neu) | 100 | 0.135 |
| sad (sad) | 100 | 0.127 |

How these were counted: the folder of each file; the code in the file name agrees with the folder for every file. Hours per label are measured in the same way as the total hours above and summed over the recordings of that label.

### Issues in the release

- `metadata.csv` has 500 rows with the columns `file_name, speaker_id, sentence_id, emotion, gender, sample_rate, duration_sec`, but its `emotion` column is empty in every row, `duration_sec` is 0.04 in every row (the recordings last 4 to 6 s) and its file names end in a single `.wav`. The recipe does not read it.
- One recording (`happy/01_100_hap.wav.m4a`) is AAC in an MP4 container, unlike the others.
- No two recordings have byte-identical content.

### Not stated in the documentation

- The file names follow the same pattern and emotion codes as EmoTa (Tamil), `<speaker>_<sentence>_<ang|fea|hap|neu|sad>`. Compared with the 936 EmoTa recordings (both converted to 16 kHz), no pair of recordings whose lengths differ by 0.15 s or less has a waveform correlation of 0.5 or more (lags up to 0.5 s), so this check found no recording taken from EmoTa.


## Splits

The release publishes no partition and does not identify speakers. Files are
split 70/10/20 at random within each label (`train_ratio`, `valid_ratio`,
`test_ratio`, `split_seed` in `dataset/config.yaml`).

| split | recordings | per label |
|---|---|---|
| train | 350 | 70 |
| valid | 50 | 10 |
| test | 100 | 20 |

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
