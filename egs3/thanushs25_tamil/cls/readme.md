# Thanushs25 Tamil speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on this corpus alone and scores it on its own test split, classifying each
utterance into one of five emotions.

The corpus is a small set of Tamil recordings published on Hugging Face as
`Thanushs25/tamil-audio-emotion-classification`, one folder per emotion:
`angry`, `fear`, `happy`, `neutral` and `sad`. The repository holds 195 wav files
(49 angry, 49 sad, 41 happy, 28 fear, 28 neutral); two pairs are byte-identical
copies (`happy/audio_11.wav` = `happy/audio_22.wav`,
`neutral/audio_8.wav` = `neutral/audio_9.wav`), so 193 distinct recordings
remain (1.09 hours in total, 2.7 to 140.5 seconds long). The files are 44.1 or
48 kHz stereo, so `create_dataset` converts them to 16 kHz mono with ffmpeg.
Many recordings are long: 29 of the 195 exceed 30 seconds. Two more pairs are
the same sound without being byte-identical (`happy/audio_33` and `happy/audio_0`
decode to identical 16 kHz audio; `angry/audio_31` and `audio_32` are almost
identical, correlation 0.9994). All four are kept for now; the question is to be
reopened when it is decided that the corpus will be used.

The label is the name of the folder a recording sits in; `dataset/config.yaml` maps it
to the manifest label. The repository also holds `final_data/fear/documentation.pdf`,
a printout of an unrelated notebook (natural language to SQL); it is not used.

## Provenance

The dataset card states no licence, no source of the recordings, no recording
conditions, no speaker information and no labelling procedure. Whether the labels
describe acted emotion or were assigned some other way is therefore unknown, and
so is the right to redistribute the audio. The licence and the origin are to be
asked of the uploader (not yet done), and the data stays on hold until then;
use it only for local experiments. A listening check found that the clips sound
like film or television audio (Whisper transcripts also read as colloquial Tamil
dialogue), which makes the rights question more pressing.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `THANUSHS25_TAMIL` — the directory holding `final_data/<emotion>/*.wav`, or where to download them to
- `THANUSHS25_TAMIL_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the 195 wav files (0.69 GB) from the repository at the
pinned commit `2a01784b7a9d8d7fbd36a7d2480214d9a26e5312` with four parallel
requests and checks each against the SHA-256 listed by the Hub. The download took
about 91 seconds; `create_dataset`, `remove_long_short` and `prepare_labels`
together took about 40 seconds.

## Splits

No partition and no speaker id are published, so utterances are split at random
within each label (70/10/20, `split_seed` in `dataset/config.yaml`). Speakers are
not held out, and since several recordings may come from one source the splits
may not be independent. Test is small (39 utterances), so scores vary strongly
with the seed.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 134 | 115 |
| valid | 20 | 15 |
| test | 39 | 39 (not filtered) |

`remove_long_short` drops clips outside 0.2 to 30 seconds from train and valid;
the long recordings account for all of the 24 dropped. In the filtered train
split: angry 26, fear 15, happy 26, neutral 16, sad 32.

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

**Duplicates.** `exclude_duplicates` in `dataset/config.yaml` (default `true`) keeps one recording per group of byte-identical files with one label (2 extra copies are dropped); `false` keeps every copy. Copies with conflicting labels are always dropped.
