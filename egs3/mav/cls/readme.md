# MAV speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on MAV alone and scores it on MAV's own test split, classifying each
utterance into one of nine emotions.

The [Montreal Affective Voices](https://zenodo.org/records/17036776) (MAV) are 90
nonverbal affect bursts, such as an "ah" produced with an emotion, not words. 10
actors (5 male, 5 female; actor ids 6, 42, 45, 46, 53, 55, 58, 59, 60, 61) each
produced one burst for each of nine labels: `angry`, `disgust`, `fear`, `happy`,
`neutral`, `pain`, `pleasure`, `sad` and `surprise`, so every label has exactly 10
clips. Clips are 0.24 to 4.31 seconds long, 97.8 seconds (0.027 hours) in total, and
are 44.1 kHz mono; `create_dataset` converts them to 16 kHz with ffmpeg.

The label is the intended emotion, taken from the `Emotion` column of the corpus
table `Sounds.csv` (tab-separated, also holding `Actor_ID`, `Sex` and acoustic
measures); `dataset/config.yaml` maps `anger`, `happiness` and `sadness` to `angry`,
`happy` and `sad` and keeps `pain` and `pleasure`. The release also holds the
original article as a PDF, which this recipe does not use.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `MAV` — the directory holding `Sounds.csv` and the extracted `Sounds/` directory, or where to download them to
- `MAV_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `Sounds.zip` (7.6 MB) and `Sounds.csv` from
[Zenodo](https://zenodo.org/records/17036776), checks both md5 sums and extracts the
wav files. Measured on one run: about 28 seconds to download; `create_dataset`,
`remove_long_short` and `prepare_labels` together took about 28 seconds.

## Splits

The release publishes no partition. Whole actors are held out (`split_seed` in
`dataset/config.yaml`): 7 train actors (6, 45, 46, 53, 55, 58, 61), 1 valid actor
(42, male) and 2 test actors (59 male, 60 female). Each actor recorded every
label once, so every split holds all nine labels, evenly.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 63 | 63 |
| valid | 9 | 9 |
| test | 18 | 18 (not filtered) |

All clips are 0.24 s or longer, so `remove_long_short` drops nothing. With
18 test clips and 9 valid clips, scores are very noisy.

## License and citation

CC BY 4.0 (Zenodo record metadata). Cite: Pascal Belin, Sarah Fillion-Bilodeau and
Frédéric Gosselin, "The Montreal Affective Voices: A validated set of nonverbal
affect bursts for research on auditory affective processing", Behavior Research
Methods 40(2), 531-539, 2008, https://doi.org/10.3758/BRM.40.2.531.

## Listening test

Thirty participants rated each vocalization for the 8 emotions (valence, arousal and intensity); the Zenodo description gives a mean recognition of 68%. The per-clip ratings are not in the release.

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
