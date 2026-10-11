# Maleo Emotion speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Maleo Emotion alone and scores it on its own test split, classifying each
recording into one of seven emotions.

Maleo Emotion (`maleo-ai/maleo-emotion` on Hugging Face, CC BY 4.0) is Indonesian
speech: 700 clips of about 3 seconds, 100 for each of `angry`, `disgust`, `fear`,
`happy`, `neutral`, `sad` and `surprise`, the same seven labels the other recipes
share. According to the dataset card of the copy used here, the clips are cut
from YouTube videos of spontaneous speech, the label is the dominant emotion
(some clips mix two), and some clips are code-mixed or entirely in a foreign
language. The speakers are not identified.

## What is used

The copy [`dikal/maleo-emotion-annotated`](https://huggingface.co/datasets/dikal/maleo-emotion-annotated),
pinned to one commit (`revision` in `dataset/config.yaml`). It adds transcripts
and word timestamps to the original, which this recipe does not use. All 700
recordings (0.59 hours) are used. The label is the name of the folder the file is
in (`Jijik`, `Marah`, `Netral`, `Sedih`, `Senang`, `Takut`, `Terkejut`, which are
Indonesian for disgust, angry, neutral, sad, happy, fear and surprise). The files
are 44.1 kHz stereo; `create_dataset` converts them to 16 kHz mono 16-bit WAV.
Durations range from 1.5 to 4.5 seconds (median 3.0), so `remove_long_short`
drops nothing.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `MALEO` — the directory holding `audio/<emotion>/`, or where to download it to
- `MALEO_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` lists the 700 files of the pinned revision through the Hugging
Face API and downloads them with 8 threads. Hugging Face lists each file's size
and SHA-256, and both are checked. No login is needed for this copy; the original
repository asks for one. Because the copy is not the original authors' upload,
its contents rest on the copy's author having kept the original files unchanged,
which was not checked against the original.

## Splits

The release publishes no partition. Recordings are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`), which gives every label 70, 10 and 20 recordings.
Speakers are unknown and cannot be held out, so the splits probably share
speakers.

| split | recordings | after `remove_long_short` |
|---|---|---|
| train | 490 | 490 |
| valid | 70 | 70 |
| test | 140 | (not filtered) |

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`). The download
depends on the network, and a 28 GB download was running at the same time.

| step | time |
|---|---|
| `prepare_source` (700 files, 337 MB, 8 threads, SHA-256 checked) | 2 min 20 s |
| `create_dataset` (conversion to 16 kHz, one file at a time) | 43.4 s |
| `remove_long_short` | 16.4 s |
| the whole `create_dataset remove_long_short prepare_labels` command | 1 min 8 s (includes starting `pixi` and Python) |

The downloaded audio takes 337 MB and the converted data 67 MB.

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
