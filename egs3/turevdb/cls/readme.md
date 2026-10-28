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

According to the paper, the six speakers (three women and three men, early twenties to
mid-thirties) are volunteer graduate and undergraduate students of Cognitive Science and
Psychology with no acting experience. Following the authors' guidelines, each recorded
the 82 words on their own with the Sound Recorder app of Sony Mobile (mono WAV, 44.1 kHz),
pronouncing each word as if in the emotional state. The numbers per speaker are
uneven: `1157` has 150 recordings with no `calm` or `happy`, and `7895` has 273 with only 29
`happy`, while the other four have 328 each.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `TUREVDB` — the extracted `Sound Source/` directory, or where to download it to
- `TUREVDB_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the repository archive from GitHub, pinned to the
latest commit, and extracts only the WAV files under `Sound Source/`. The
repository's README states CC BY-NC-ND 4.0, while its LICENSE file is the GNU GPL v3 (and
GitHub lists GPL-3.0); the two disagree and the authors' intent is unverified.

## Splits

The authors' train (1,386) and test (349) sets are published only as spectrogram
images, which correspond one-to-one to the 1,735 recordings. The recipe does not use them:
utterances are split 70/10/20 at random within each label (`train_ratio`, `valid_ratio`,
`test_ratio`, `split_seed` in `dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 1214 |
| valid | 174 |
| test | 347 |

## Listening test

**The recordings were evaluated by listeners** (paper, sections 2.1.3 and 4.1).

- While recording, each speaker listened to their own recordings and played them to an
  assessor, a third person of their choice. Recordings that lacked an emotional tone
  according to the speaker or the assessor were re-recorded. No scores come from this check.
- For the evaluation, three volunteer judges (graduate students with a Bachelor's degree
  in psychology) listened to the validation tokens (20% of the recordings). They were
  given the tokens under pseudo-names and no information on their emotion, and chose one
  of the four emotions. Each judge heard the tokens of one of three experiments: the
  full band (0-8000 Hz), 0-5000 Hz or 500-8000 Hz. The accuracy against the intended
  emotion was 0.65, 0.64 and 0.64 respectively. In the full band the judge's recall was
  0.80 for `angry`, 0.68 for `calm`, 0.56 for `happy` and 0.54 for `sad`. Each token was
  judged by one person per experiment, so no inter-rater agreement exists.
- The individual responses are not in the download; the recipe uses the intended emotion
  of the file name.

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
