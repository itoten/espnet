# Kannada speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Kannada alone and scores it on Kannada's own test split, classifying each
utterance into one of six emotions.

[A Kannada Emotional Speech Dataset](https://zenodo.org/records/6345107)
(Vishakha Agrawal, 2022) is Kannada read speech from 13 volunteers (9 female,
4 male, aged 12 to 48), each reading the same six sentences in six emotions:
468 utterances, 78 per emotion (`angry`, `fear`, `happy`, `neutral`, `sad`,
`surprise`; no `disgust`), 0.50 hours in total, each 1.5 to 7.2 seconds long.
The recordings were made by the volunteers themselves, not in a controlled
environment.

The label comes from the file name, `<speaker>-<emotion>-<sentence>.wav` (e.g.
`09-03-02.wav` is speaker 09, emotion 03, sentence 02); `dataset/config.yaml`
maps the six emotion numbers to labels. The sampling rate and channel count
differ between files (16, 44.1 and 48 kHz; mono and stereo), so `create_dataset`
converts every recording to 16 kHz mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `KANNADA` — the directory of WAV files, or where to download them to
- `KANNADA_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the 468 WAV files one by one from
[Zenodo](https://zenodo.org/records/6345107) (the record has no archive),
checking each file's MD5 against the record and resuming after an interrupted
run. The corpus is CC BY 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 324 |
| valid | 48 |
| test | 96 |

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
