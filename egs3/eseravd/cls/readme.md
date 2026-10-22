# E-SERAVD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on E-SERAVD alone and scores it on E-SERAVD's own test split, classifying each
utterance into one of six emotions.

E-SERAVD (EEPIS Speech Emotion Recognition Audio Video Dataset) is an
Indonesian-language corpus of short clips cut from films
([Kaggle](https://www.kaggle.com/datasets/ahmadnafibudianto/e-seravd-speech-emotion-recognition-av-dataset)).
The Kaggle archive holds three releases; this recipe uses `E-SERAVD 1.2`, whose
`02_Audio-Only/<NN_Emotion>/<film>/NN-NN-<film>-T-<clip>.wav` files number 3,000,
500 per emotion, from 11 films. The label is the emotion code in the file name
(the second number, which also prefixes the folder): `angry`, `sad`, `neutral`,
`happy`, `disgust` and `fear`. According to the dataset's README, three human
raters scored each clip and a clip kept its main emotion only when the mean
probability was at least 0.67. The recordings are 44.1 kHz stereo, so
`create_dataset` converts them to 16 kHz mono with ffmpeg. After conversion the
clips are 0.83 to 7.5 seconds long and 2.0 hours in total.

The two older releases in the archive (`E-SERAVD` and `E-SERAVD 1.1`) hold the
same 1,200 four-emotion clips. Of 200 randomly chosen `1.1` clips, 115 (57.5%) are
byte-identical to a `1.2` clip, and 81 of those 115 carry a different emotion
label in `1.2`. The README does not say how the labels were reassigned. This
recipe takes the `1.2` labels as published.

## Excluded clips

Twelve pairs of `DRNT` clips (24 clips, all in the training split) hold the same
audio, with a correlation of at least 0.97, under two different emotion labels
(for example one `sad` and one `fear`). Which label is right cannot be decided,
so both clips of every pair are listed in `exclude_files` in
`dataset/config.yaml`, and `create_dataset` stops if one of the names is not
found. Pairs whose two clips are cut with different lengths were not searched
for. Training goes from 2,003 to 1,979 utterances.

Environment variables (both optional; defaults are `download/` and `data/` under the
recipe):

- `ESERAVD` — the directory holding the extracted `E-SERAVD 1.2/` folder
- `ESERAVD_OUTPUT` — where the converted audio and manifests are written

## Download

The Kaggle dataset needs a login and the zip is about 6.4 GB because it also holds
the video, so the recipe does not download it. Download it by hand, extract only the
`.wav` files of `E-SERAVD 1.2` and point `$ESERAVD` at the folder holding it;
otherwise `create_dataset` stops with the URL. On one run `create_dataset` took 130
seconds. The download time was not measured.

## Splits

The archive publishes no partition or speaker id. Whole films are held out as the
closest stand-in for speaker independence (`n_test_groups`, `n_valid_groups`,
`split_seed` in `dataset/config.yaml`): three films for test, one for valid and
seven for training (the exclusion above does not change which films go where).
Scores are therefore film-independent, but the label balance of
the held-out films is uneven; the single valid film has only 7 `sad` clips.

| split | films | utterances | hours | after `remove_long_short` | angry | disgust | fear | happy | neutral | sad |
|---|---|---|---|---|---|---|---|---|---|---|
| train | AADC, DHB, DRNT, HA3, HSL, KKNP, TAOL | 1,979 | 1.35 | 1,979 | 383 | 319 | 300 | 319 | 378 | 280 |
| valid | D1990 | 181 | 0.12 | 181 | 49 | 33 | 23 | 21 | 48 | 7 |
| test | ADNJ, PKGL, SWDN | 816 | 0.51 | 816 (not filtered) | 64 | 146 | 172 | 158 | 71 | 205 |

## License

The Kaggle page lists the dataset as CC BY 4.0. The clips are cut from films; the
rights to the film audio itself have not been checked.

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
