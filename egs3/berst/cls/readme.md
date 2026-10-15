# BERSt speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on BERSt alone and scores it on BERSt's own test split, classifying each
utterance into one of seven emotions.

[BERSt](https://huggingface.co/datasets/Rosie-Lab/BERSt) (Basic Emotion Random
phrase Shouts) is an English corpus of 98 actors who recorded 13 nonsense phrases
at home on their own smartphones, asking each to speak, raise their voice and
shout the phrase while moving the phone to different places in the house. The
recipe uses the speech segments the authors cut from those recordings
(4,523 clips, 3.73 hours, 1.03 to 22.5 seconds long). The clips are mono, 48 kHz
(3,704) or 44.1 kHz (819), so `create_dataset` converts them to 16 kHz with ffmpeg.

The label is the `affect` column of the metadata: the emotion the actor was asked
to portray. The dataset card states that these labels were not validated by
listeners. `dataset/config.yaml` maps `anger`, `joy` and `sadness` to `angry`,
`happy` and `sad`; `disgust`, `fear`, `neutral` and `surprise` are kept. All seven
classes are roughly balanced in train (454 to 564 clips).

## Provenance

Hugging Face dataset `Rosie-Lab/BERSt`, pinned to commit
`fed35c477427cff44206464b7f85e68d1fc99062`, licence CC BY 4.0 as stated on the
card; no login is needed. All 4,523 clips match the SHA-256 listed by the Hub.
The authors are Paige Tuttösí et al.; please cite their paper:

```bibtex
@article{tuttosi2025bersting,
  title   = {BERSting at the Screams: A Benchmark for Distanced, Emotional and Shouted Speech Recognition},
  author  = {Paige Tuttösí and Mantaj Dhillon and Luna Sang and Shane Eastwood and Poorvi Bhatia and Quang Minh Dinh and Avni Kapoor and Yewon Jin and Angelica Lim},
  journal = {Computer Speech & Language},
  volume  = {95},
  pages   = {101815},
  year    = {2026},
  doi     = {10.1016/j.csl.2025.101815},
}
```

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `BERST` — the directory holding `berst-data/{train,validation,test}/`, or where to download them to
- `BERST_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` fetches the three `metadata.csv` files and each listed clip (1.27 GB
in total) with 8 parallel connections. Measured on one run: about 14.5 minutes to download, then
about 4 minutes to convert, so `create_dataset` took 1,107 seconds;
`remove_long_short` and `prepare_labels` together about 18 seconds.

## Splits

The published train / validation / test split is used. It is speaker-disjoint
(78, 10 and 10 actors, verified from `user_id`).

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 3,503 | 3,503 |
| valid | 488 | 488 |
| test | 532 | 532 (not filtered) |

The test split is not balanced: `sad` has only 26 clips (angry 99, fear 99, neutral
94, happy 79, surprise 69, disgust 66), so per-class scores for `sad` are noisy.

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
