# SantaliESC speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on SantaliESC alone and scores it on SantaliESC's own test split,
classifying each utterance into one of five emotions.

[SantaliESC](https://zenodo.org/records/20702265) (Santali Emotional Speech
Corpus) is acted Santali speech recorded in Koraput, Odisha, India. The Zenodo
record states 2,000 recordings from 80 speakers aged 18 to 60: five sentences
(Ol Chiki script) spoken in five emotions, `angry`, `happy`, `neutral`, `sad`,
`surprise`, 400 recordings per emotion, about 1 h 53 min in all, cut into
3-5 s WAV files at 48 kHz.

The record does not describe how the files inside the archive are named, so
the labels are read from the paths with configurable rules
(`emotion_aliases`, `emotion_re`, `speaker_re` in `dataset/config.yaml`):
every audio file must contain the name of exactly one emotion in its folder or
file name. If a file does not, `create_dataset` stops and prints a sample of
the paths it found. Every file found under the source root is listed in
`source_files.txt`, and the members of a downloaded archive in
`archive_members.txt`, both under the output directory. `create_dataset`
converts all recordings to 16 kHz mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `SANTALIESC` — an existing unpacked copy of the archive, or where to download it to
- `SANTALIESC_OUTPUT` — where the resampled audio, manifests and file lists are written

## The data

TODO: filled after the first build (counts of the release).

## Download

`prepare_source` downloads
`SantaliESC (Santali Emotional Speech Corpus)-20260615T095634Z-3-001.zip`
(892 MB) from [Zenodo](https://zenodo.org/records/20702265), checks it against
the MD5 the record lists, unpacks it and deletes the archive. The corpus is
CC BY 4.0.

## Splits

The corpus publishes no partition. When the speakers can be identified (by
`speaker_re`, or by a folder per speaker that holds all five emotions), the
speakers are shuffled with `split_seed` and split 80/10/10, so no speaker is
in two splits (80 speakers give 64/8/8). Otherwise the utterances are split
70/10/20 at random within each label, and the splits share speakers.
`split_mode` in `dataset/config.yaml` forces either one.

## Listening test

The Zenodo record states that the emotion labels were validated by ten human
evaluators, with an average recognition accuracy of about 74.6% for the
intended emotions. Per-recording ratings are not part of the download; the
recipe uses the intended emotion.

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
