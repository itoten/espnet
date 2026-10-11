# Oréau speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Oréau alone and scores it on Oréau's own test split, classifying each
utterance into one of seven emotions.

[Oréau](https://zenodo.org/records/4405783) is French read speech from 32
non-professional speakers, each emotion with its own sentences about everyday
classroom life (79 sentences in all). This recipe uses version 2 of the
release: the 434 of version 1's 502 utterances that listeners in a perception
test recognised and judged natural. Seven emotions: `neutral` (70), `angry`
(67), `disgust` (67), `happy` (62), `surprise` (62), `sad` (55) and `fear`
(51).

The label comes from the file name, `<speaker><sentence><emotion><suffix>.wav`
(e.g. `25a09Da.wav`); no annotation file is read. Because each emotion has its
own sentences, the words carry information about the label.

The recordings are 44.1 kHz stereo from two microphones. The two channels are
not identical in any file and are poorly correlated in some, so averaging them
can attenuate the speech; `create_dataset` keeps one channel (`channel` in
`dataset/config.yaml`) and resamples it to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `OREAU` — a directory holding the extracted `OréauFR_02/`, or where to
  download it to
- `OREAU_OUTPUT` — where the resampled audio and manifests are written

## Download

`prepare_source` downloads `OréauFR_02.rar` (109 MB) from
[Zenodo](https://zenodo.org/records/4405783). The corpus is CC BY 4.0.

The archive is RAR, which Python cannot read, so extracting it needs `bsdtar`
on `PATH` (from libarchive, e.g. the conda-forge `libarchive` package).

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 305 |
| valid | 44 |
| test | 85 |

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
