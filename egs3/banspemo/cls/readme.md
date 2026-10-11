# BanSpEmo speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on BanSpEmo alone and scores it on BanSpEmo's own test split, classifying each
utterance into one of six emotions.

[BanSpEmo](https://data.mendeley.com/datasets/rdwn4bs5ky/1) (a Bangla emotional
speech corpus; arXiv:2312.14020) is acted Bangla speech: 22 non-professional
speakers (11 male, 11 female) read 12 sentences, in two sets of six, in the six
emotions `angry`, `disgust`, `fear`, `happy`, `sad` and `surprise`. There are 792
recordings (132 per emotion), 1.39 hours in total, each 2.4 to 11.9 seconds
long. There is no `neutral`.

The label comes from the file name,
`ss<set>_s<sentence>_sp<speaker>_<gender>_<emotion>.wav` (e.g.
`ss1_s2_sp13_m_01.wav` is sentence set 1, sentence 2, speaker 13, male,
anger); the emotion codes are 01 anger, 02 disgust, 03 fear, 04 happy, 05 sad
and 06 surprise, as in the paper's Table 5. The recordings are 44.1 or 48 kHz
stereo; `create_dataset` converts them to 16 kHz mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `BANSPEMO` — the directory holding the extracted audio, or where to download it to
- `BANSPEMO_OUTPUT` — where the resampled audio and manifests are written

## Download

`prepare_source` downloads the archive that Mendeley Data builds on request
(582 MB) and extracts the WAV files. The corpus is CC BY 4.0. Mendeley publishes no
checksum for it, so completeness is checked by counting the 792 files.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so the splits share all
22 speakers.

| split | utterances |
|---|---|
| train | 558 |
| valid | 78 |
| test | 156 |

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
