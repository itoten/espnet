# UFTBESD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on UFTBESD alone and scores it on UFTBESD's own test split, classifying each
utterance into one of seven emotions.

[UFTBESD](https://data.mendeley.com/datasets/h3f8pjbw73) is an acted Bangla
corpus: 100 speakers (50 female, 50 male, aged 15 to 38) each say two sentences
in seven emotions, `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad` and
`surprise`, once each. The 1,400 wav files (200 per emotion) total 1.04 hours,
0.74 to 6.04 seconds long. They were recorded on the speakers' own smartphones,
112 of them outdoors and the rest indoors, as 44.1 kHz mono 16-bit wav.

The label is the intended emotion, taken from the `emotion_label` column of
`UFTBESD_Metadata.csv`; `dataset/config.yaml` maps it to lower-case labels.
`create_dataset` converts the recordings to 16 kHz mono with ffmpeg.

Some recordings are byte-identical: 27 groups (68 files) share the same audio,
21 of them across two different speakers with the same label and 6 within one
speaker under two different emotions (for example `disgust` and `surprise` of
the same sentence). They are kept as they are.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `UFTBESD` — the directory holding `UFTBESD_Metadata.csv` and the extracted wav files, or where to download them to
- `UFTBESD_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `UFTBESD.zip` (244 MB) and the metadata CSV from
[Mendeley Data](https://data.mendeley.com/datasets/h3f8pjbw73) (version 4,
SHA-256 checked) and extracts the wav files. Mendeley rejects the default Python
user agent, so the recipe sends its own. Measured on one run: about 7 minutes to
download; `create_dataset`, `remove_long_short` and `prepare_labels` together
took about 96 seconds.

## Splits

The release publishes no partition. Speakers are split 70/10/20 at random
(`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`), so no speaker appears in two splits. Speakers who share
a byte-identical recording are kept together. Every speaker has all seven
emotions, so the splits are exactly balanced.

| split | speakers | utterances | after `remove_long_short` |
|---|---|---|---|
| train | 70 | 980 | 980 |
| valid | 10 | 140 | 140 |
| test | 20 | 280 | 280 (not filtered) |

## License and citation

The dataset is licensed CC BY 4.0 (Mendeley Data). The record names Md Rayhan Ali
and colleagues of Bangabandhu Sheikh Mujibur Rahman Digital University as
authors; please cite the Mendeley Data record
(https://doi.org/10.17632/h3f8pjbw73.4).

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
