# UA-SER speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on UA-SER alone and scores it on UA-SER's own test split, classifying each
utterance into one of four emotions.

[UA-SER](https://huggingface.co/datasets/OlhaHavryliuk/UA-SER) is a Ukrainian
emotional speech corpus (CC BY 4.0): 952 short clips (0.52 to 5.5 s, 0.55 hours)
labelled `angry`, `happy`, `neutral` or `sad`, with a speaker id for each clip
(400 speakers). The clips are already 16 kHz mono; `create_dataset` copies them
through ffmpeg unchanged in format.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `UA_SER` — the directory holding `dataset.csv` and `clips/`, or where to download them to
- `UA_SER_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `dataset.csv` and the 952 clips (63 MB) from Hugging Face, pinned
to one commit; every file is checked against the size and SHA-256 (or git blob id) the
listing gives. Measured on one run, download plus `create_dataset`, `remove_long_short`
and `prepare_labels` took about 3 minutes 18 seconds.

## Splits

`dataset.csv` marks each clip `train` or `test`; the two share no speaker. The published
split is kept (test 181 clips). A valid split is cut from train by whole speakers
(`valid_speaker_ratio` 0.1, `split_seed` in `dataset/config.yaml`).

| split | utterances |
|---|---|
| train | 708 |
| valid | 63 |
| test | 181 |

No clip is shorter than the 0.5 s floor of `remove_long_short`.

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
