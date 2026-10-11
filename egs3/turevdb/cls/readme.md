# TurEV-DB speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on TurEV-DB alone and scores it on TurEV-DB's own test split, classifying each
utterance into one of four emotions.

[TurEV-DB](https://github.com/Xeonen/TurEV-DB) (Turkish Emotion Voice Database;
Canpolat, Ormanoglu and Zeyrek, SLTU/CCURL 2020) is acted Turkish speech: single
words read in four emotions, `angry`, `calm`, `happy` and `sad`, 357 to 487
recordings each, 1,735 in total, 0.46 hours. Each is 0.35 to 2.5 seconds long.
There is no `disgust`, `fear`, `neutral` or `surprise`.

The label comes from the file name, `<speaker>_<emotion>_<word>.wav` (e.g.
`1157_kz_acik.wav`); the emotion codes are kz, sk, mt and hl. Each file also
sits in a folder named after its emotion (`Angry`, `Calm`, `Happy`, `Sad`), and
`create_dataset` fails if the two disagree. The first field takes six values;
it is used as the speaker. The recordings are 44.1 kHz mono; `create_dataset`
resamples them to 16 kHz.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `TUREVDB` — the extracted `Sound Source/` directory, or where to download it to
- `TUREVDB_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the repository archive from GitHub, pinned to the
latest commit, and extracts only the WAV files under `Sound Source/`. The
repository's README states CC BY-NC-ND 4.0; its LICENSE file is the GNU GPL v3.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 1214 |
| valid | 174 |
| test | 347 |

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
