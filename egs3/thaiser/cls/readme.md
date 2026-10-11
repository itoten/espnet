# Thai SER speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Thai SER alone and scores it on Thai SER's own test split, classifying each
utterance into one of five emotions.

[Thai SER](https://github.com/vistec-AI/dataset-releases/releases/tag/v1) (VISTEC-depa
AI Research Institute of Thailand with Chulalongkorn University) is acted Thai speech
from 200 actors: sentence reading (3 sentences, each in 5 emotions at 2 intensities,
spoken twice) and two-person improvised scenes. 80 sessions were recorded in a
studio and 20 over Zoom. Each recording was labeled by 3 to 8 crowd annotators and
the label is their majority vote. The labels are `neutral`, `angry`, `happy`, `sad`
and `frustrated`; `frustrated` is not one of the seven labels shared with the other
recipes, and there is no `disgust`, `fear` or `surprise`.

## What is used

The release lists 27,854 recordings (41.6 hours) in `emotion_label.json`, one
microphone per recording (`con` in the studio, `mic` on Zoom). The `clip` and
`middle` microphone files in the studio archives are not used. A recording is
kept when its majority label is one of the five emotions (not `None`, nor
`other`) and its annotator agreement is at least 0.71. This gives 14,182
recordings (21.3 hours), the count the release page reports. They come from 200 actors:
11,146 studio and 3,036 Zoom, 8,255 improvised and 5,927 read. The source audio is
44.1 kHz FLAC, mono or stereo; `create_dataset` converts it to 16 kHz mono 16-bit
WAV, `num_workers` files at a time. Durations range from 0.5 to 108.8 seconds
(median 4.6). `remove_long_short` drops 42 recordings from train/valid (40 longer
than 30 seconds, 2 of about 0.5 seconds); test is not filtered and holds 5 of the
45 recordings longer than 30 seconds.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `THAISER` — the directory holding `emotion_label.json` and the extracted audio, or where to download it to
- `THAISER_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `emotion_label.json` and the ten zip archives from the
GitHub release (12.6 GB) and extracts only the kept recordings. No checksum is
published, so each file's size is checked, and the 14,182 recordings are
counted. The release is CC BY-SA 4.0.

## Splits

The release publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so the splits share the actors,
and the three read sentences appear in all splits.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 9928 | 9890 |
| valid | 1417 | 1413 |
| test | 2837 | (not filtered) |

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`, 16 workers). The
download depends on the network.

| step | time |
|---|---|
| download of the 10 archives, 12.6 GB | about 17 min with `curl`; `prepare_source` fetched one 549 MB archive in 14 s |
| extraction of the 14,182 recordings | 8 s |
| `create_dataset remove_long_short prepare_labels` (conversion to 16 kHz) | 1 min 22 s |

The converted data takes 2.4 GB and the extracted FLAC files 2.9 GB.

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
