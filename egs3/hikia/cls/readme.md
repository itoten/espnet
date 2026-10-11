# HiKIA speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on HiKIA alone and scores it on HiKIA's own test split, classifying each
utterance into one of four emotions.

[Hi, KIA](https://zenodo.org/records/7091465) (Kim et al., APSIPA 2022) is a
set of Korean-accented wake-up word utterances ("Hi, KIA") from 8 speakers (4
female, 4 male) in four emotions, `angry`, `happy`, `neutral` and `sad`: 488
utterances (107, 129, 119 and 133), 0.09 hours in total, each only 0.24 to
1.75 seconds long.

The label comes from the last letter of the file name,
`<speaker>_S<scene>_<trial>_<emotion>.wav` (e.g. `F5_S07_1_h.wav` is happy, from
speaker F5); `dataset/config.yaml` maps `a`, `h`, `n` and `s` to labels. The
letters agree with the `emo` column of the release's `annotation.csv` for all
488 files. The recordings are 48 kHz stereo, so `create_dataset` converts them
to 16 kHz mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `HIKIA` — the extracted `hi_kia/wav/` directory, or where to download it to
- `HIKIA_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `hi_kia_1.0.tar.gz` (750 MB) from
[Zenodo](https://zenodo.org/records/7091465) and extracts only its `wav/`
directory; the rest of the archive is precomputed features and pretrained
weights. The corpus is CC BY 4.0.

## Splits

The release publishes eight speaker-held-out folds (`split/`), not a single
partition. This recipe does not use them: utterances are split 70/10/20 at
random within each label (`train_ratio`, `valid_ratio`, `test_ratio`,
`split_seed` in `dataset/config.yaml`), so all three splits contain all 8
speakers.

| split | utterances |
|---|---|
| train | 341 |
| valid | 49 |
| test | 98 |

Because the utterances are so short, `remove_long_short` applies no minimum
duration (99 of the 488 are under 0.5 s).

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
