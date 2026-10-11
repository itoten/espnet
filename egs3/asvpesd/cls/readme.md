# ASVP-ESD speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on ASVP-ESD alone and scores it on ASVP-ESD's own test split, classifying each
utterance into one of seven emotions.

[ASVP-ESD](https://zenodo.org/records/4782712) (Audio, Speech and Vision
Processing Lab Emotional Sound Database; Tientcheu Touko Landry Dejoli, Qianhua
He and Wei Xie) is a collection of
emotional sounds taken from websites, YouTube and films, labelled with 12
emotions. This recipe uses the speech recordings only (3,555 of the 9,092
files; the rest are non-speech vocalizations such as laughs and cries) and the
seven emotions shared with the other corpora, `neutral`, `happy`, `sad`,
`angry`, `fear`, `disgust` and `surprise`. The 314 speech recordings labelled
boredom, excited, pleasure, pain or disappointment are left out, which leaves
3,241 recordings, 5.48 hours, from 76 speakers. Each is 0.4 to 30.6 seconds
long. The speech is in several languages, mostly Chinese and English. The
classes are very uneven, from 997 `neutral` down to 76 `disgust`.

The label comes from the file name,
`<modality>-<channel>-<emotion>-<intensity>-<statement>-<actor>-<age>-<source>-<language>[-...].wav`
(e.g. `03-01-06-01-02-12-02-01-01-16-04.wav`); the emotion codes are 02 neutral,
03 happy, 04 sad, 05 angry, 06 fearful, 07 disgust and 08 surprised, and the
vocal channel is 01 for speech, as in the Zenodo description. The number of
fields varies between files, so only the first four are read. The speaker is
taken from the `actor_N` folder. The recordings are already 16 kHz mono;
`create_dataset` still passes them through ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `ASVPESD` — the directory holding the extracted audio, or where to download it to
- `ASVPESD_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `ASVP_ESD.zip` (1.26 GB) from Zenodo, checks it
against the MD5 Zenodo lists, and extracts only the speech WAV files. The
corpus is CC BY 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so the splits share all
76 speakers.

| split | utterances |
|---|---|
| train | 2268 |
| valid | 325 |
| test | 648 |

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
