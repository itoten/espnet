# UrduSER speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on UrduSER alone and scores it on its own test split, classifying each
recording into one of seven emotions.

[UrduSER](https://data.mendeley.com/datasets/jcpfjnk5c2/4) (Mendeley Data, V4,
doi:10.17632/jcpfjnk5c2.4) is Urdu speech cut from Pakistani drama serials and
telefilms on YouTube. According to the dataset description, 10 professional actors
(equal numbers of men and women) contribute 50 clips for each of the seven emotions, and the
clips were checked by experts and by a model. Akhtar, Jahangir, Ain, Nauman, Uddin
and Ullah describe it in "UrduSER: A comprehensive dataset for speech emotion
recognition in Urdu language", Data in Brief 60 (2025) 111627,
doi:10.1016/j.dib.2025.111627 (same authors as the Mendeley record). The labels are `angry`, `fear`, `boredom`, `disgust`, `happy`,
`neutral` and `sad`. There is no `surprise`. This is a different corpus from the
`urdu` recipe (400 clips, four emotions).

## How it was made

By the paper, the ten actors (five men and five women, aged 20 to 65, from different
Urdu-speaking regions) were chosen for their ability to convey emotions, and scenes with
emotional dialogue were selected for each actor from Pakistani Urdu drama serials, telefilms
and TV shows on YouTube. More than 5,000 dialogues were trimmed, of which 3,500 (500 per
emotion) were selected. The dialogue is nearly always different: the paper stresses that
each utterance has almost unique content. "Fear" and "anxiety" are merged into `fear`. The
audio of each dialogue was cut by hand in Audacity 3.6.1 from the YouTube video, at 44.1 kHz.

**The audio was processed after extraction** (paper, section 4.6), all by hand in Audacity
with the OpenVINO extensions: music separation (only the "Vocals" track is kept; "Drum",
"Bass" and "Other" are removed), noise suppression with the default settings, and trimming of
the silence at the start and end (threshold -60 dB). The sound is therefore not the original
broadcast sound and may differ in character from the other corpora.

The release also has `UrduSER_Description.xlsx` with, for each file, the duration, format,
sample rate, actor name, gender, emotion, context and the Urdu text of the dialogue. The actors
are well-known Pakistani TV actors named in the paper's naming table; their names are not
repeated here.

## What is used

The release has 3,500 clips, 500 per emotion (3.04 hours, 0.53 to 18.7 seconds,
median 2.81). File names are `<actor>_<gender>_<emotion>_<number>.wav` (for example
`1_0_1_01.wav`): actor 1 to 10, gender 0 or 1 (1,750 clips each, as in the release's
`UrduSER_Description.xlsx`), emotion code 1 to 7, then 01 to 50. The emotion codes
are 1 angry, 2 fear, 3 boredom, 4 disgust, 5 happy, 6 neutral, 7 sad. They were
checked two ways: against the Urdu emotion words in the xlsx, and against the
emotion folders of the release's zip, and both agree with the code for all 3,500
files. No two clips have the same content (SHA-256).

The source audio is 44.1 kHz stereo 16-bit PCM, whereas the xlsx describes it as
32-bit float at 41,100 Hz. `create_dataset` converts it to 16 kHz mono 16-bit WAV
with ffmpeg; `remove_long_short` drops nothing.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `URDUSER` — the directory holding `audio/*.wav`, or where to download it to
- `URDUSER_OUTPUT` — where the converted audio and manifests are written

## Rights of the source videos

The clips are extracted from third-party videos. The paper's ethics statement says:

> This article utilized emotional speech dialogs from Pakistani drama serials, telefilms, and
> TV shows that were publicly available on YouTube. To mitigate copyright concerns, we carefully
> selected clips that were either explicitly licensed under Creative Commons or used short
> excerpts consistent with fair use principles for research and academic purposes. [...] the
> dataset only includes the extracted speech audio, not the original video content, minimizing
> any potential copyright issues.

The release is labelled CC BY 4.0 (Mendeley listing and the article). The statement does not
show which clips were under a Creative Commons license and which are fair-use excerpts, nor
which Creative Commons license (it could be one that forbids commercial use), and fair use for
research does not by itself allow commercial use. The recipe follows the stated CC BY 4.0, but the
actual terms for commercial use are **unconfirmed**.

## Download

`prepare_source` downloads the dataset's zip from Mendeley Data (1.5 GB, the "Download
all" file of the dataset page), reads each recording's size and SHA-256 from the
Mendeley Data API listing, and writes only recordings that match it. Mendeley rejects
requests that do not send a browser-like `User-Agent`, so the recipe sends one. The
dataset is licensed CC BY 4.0 (the license field of the Mendeley listing). Please
cite the paper above. The clips come from third-party videos; whether the rights of
the broadcasters were cleared is not stated in the Mendeley record.

## Splits

The release publishes no partition. Recordings are split 70/10/20 at random within
each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). The actors are not held out, so the splits share all 10
actors.

| split | recordings | after `remove_long_short` | per label |
|---|---|---|---|
| train | 2450 | 2450 | 350 |
| valid | 350 | 350 | 50 |
| test | 700 | (not filtered) | 100 |

The labels are balanced, so every class weight is 1.0.

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`). The download
depends on the network.

| step | time |
|---|---|
| `prepare_source` (download of the 1.5 GB zip, SHA-256 checks, extraction of 3,500 files) | 6 min 28 s (387.5 s) |
| `create_dataset` (conversion to 16 kHz mono, one file at a time) | 215.9 s |
| `remove_long_short` | 17.2 s |
| the whole `create_dataset remove_long_short prepare_labels` command | 4 min 1 s (includes starting `pixi` and Python) |

The downloaded audio takes 1.9 GB and the converted data 343 MB.

## Listening test

**The labels were made and checked by people** (paper, sections 4.5 and 4.9).

- Three native Urdu speakers, who are the authors, labelled the dialogues: one author
  categorized each dialogue by emotion, a second author independently verified the label, and a
  third reviewed the process and randomly checked several samples. The labels come from watching
  the scene (audio and video), since the dialogues were extracted from the videos.
- For validation, a survey was given to native Urdu-speaking undergraduate students: about 100
  participants each listened to 10 random audio samples from the dataset and selected the
  emotion that best represented each. Compared with the original annotations, the accuracy was
  94% (paper, figure 4). The number of ratings per clip, per-emotion accuracy and an agreement
  coefficient are not given.

The individual ratings are not distributed. The recipe uses the label of the file name (the
emotion folder).

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

**Duplicates.** `3_0_5_03` and `3_0_5_04` (the same actor, `happy`) have almost the same waveform (correlation 0.9997 after an 80 Hz high-pass), although the files differ in bytes. The later one, `urduser-3_0_5_04`, is listed in `duplicate_ids` and is dropped only when `exclude_duplicates` in `dataset/config.yaml` is `true` (default `false`).
