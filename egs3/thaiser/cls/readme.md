# Thai SER speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Thai SER alone and scores it on Thai SER's own test split, classifying each
utterance into one of five emotions.

[Thai SER](https://huggingface.co/datasets/airesearch/thai-ser) (THAI-SER; AI Research
Institute of Thailand with VISTEC, depa and Chulalongkorn University; Wongpithayadisai et
al., "Thai Speech Emotion Recognition (THAI-SER) Corpus", Language Resources and Evaluation,
2026; arXiv 2507.09618) is acted Thai speech from 200 professional actors (112 women, 88
men, aged 18 to 55 according to the paper), directed by students, alumni and professors of
the Department of Dramatic Arts of Chulalongkorn University. It has 27,854 recordings
(41.6 hours). The labels are `neutral`, `angry`, `happy`, `sad` and `frustrated`;
`frustrated` is not one of the seven labels shared with the other recipes, and there is no
`disgust`, `fear` or `surprise`.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Hugging Face dataset `airesearch/thai-ser`, version 2.0 (2025-04-15, which fixed the calculation of `agreement`), repository revision `382232b5d45cbca5da82efe841e66e2b213713ae` (29 parquet files; `emotion_label.json`; downloaded 2026-10-28). Version 1.0 on GitHub has the same audio and majority labels. All counts cover every sample in the release, before any filtering by this recipe.

### Access

Hugging Face, direct download, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 27,854 | 200 | 41.6 |

How these were counted: every entry of `emotion_label.json` (44.1 kHz FLAC, mono or stereo); 25,185 of them have a majority label and were converted (0.29 to 108.8 s, median 4.7 s; 37.7 hours); the hours of the whole release are the figure from the release description (41.6). Speakers are the 200 distinct `actor_id` values (112 women and 88 men according to the paper).

### Files per label

| label (`majority_emo` in `emotion_label.json`) | samples | assigned to the actor (`assigned_emo`) |
|---|---|---|
| Frustrated | 7,864 | 5,270 |
| Neutral | 7,359 | 5,090 |
| Happy | 4,070 | 6,348 |
| Angry | 3,111 | 6,148 |
| Sad | 2,781 | 4,998 |
| None (no majority) | 2,664 | - |
| other | 5 | - |

How these were counted: the `majority_emo` field (the majority of the crowd annotators, 3 to 11 per recording) and the `assigned_emo` field (the emotion the actor was asked for) over all 27,854 entries. The majority label equals the assigned emotion for 16,491 of the 25,185 recordings with a majority (65%). The release has no disgust, fear or surprise class.

### Issues in the release

- Of the 41.6 hours, 2,669 recordings have no usable majority label (`None` 2,664, `other` 5).
- The paper strongly recommends discarding recordings with an agreement below 0.71 (the alpha rises from 0.413 to 0.692); the threshold keeps 14,182 recordings in version 1 and 14,181 in version 2. Versions 1 and 2 differ in `agreement` for 39 recordings.
- Three pairs of recordings of the same actor and scene are byte-identical (`s001_..._actor002_impro1_25` and `_28`, `s048_..._actor136_impro6_9` and `_10`, `s078_..._actor195_impro12_7` and `_8`); the labels agree. A search for near-identical recordings among the 25,185 found no more.

### Listening test ratings in the release

`emotion_label.json` holds, for every recording, the individual annotators' answers (`annotated`: 3 to 11 crowd annotators, audio only, several emotions may be chosen), the `agreement` and the majority emotion. 984 annotators passed a pre-test out of 1,759 applicants (paper). The paper reports alpha 0.413 for all recordings and 0.692 for agreement 0.71 or above, and the share of majority labels equal to the assigned emotion 0.592 and 0.772 respectively.


## How it was recorded

There are 100 recording sessions: 80 in a studio (sessions 1 to 18 in a noise-controlled
soundproof room, 19 to 80 in an ordinary room) and 20 online over Zoom and Zencastr. Each
session has two parts:

- **Script session**: the actor reads 3 fixed sentences, each twice for every emotion at
  two intensity levels (normal and strong), plus an additional neutral expression.
- **Improvisation session**: two actors improvise a scene for each of 15 scenarios, each
  given an emotion for each role (for example an angry customer and a hotel receptionist).

The emotion the actor was asked for is stored as `assigned_emo` in the label file. In the
studio each recording exists for three microphones (`con`, a cardioid condenser 0.5 m from
the actor; `clip`, a lavalier on the collar; `middle`, a figure-8 condenser between the two
actors); on Zoom there is one (`mic`, the actor's own). The label file lists one microphone
per recording and the recipe uses it (`con` or `mic`); `clip` and `middle` are not used.
The paper treats the Zoom recordings as an out-of-domain challenge set.

## How the labels were made

The label is **not** the emotion the actor was asked for: it is the majority vote of
crowd-sourced listeners (3 to 8 annotators per recording according to the paper, 3 to 11 in
the data, on the platform wang.in.th). They heard the audio only, without the actors' faces.

- Annotators had to pass a pretest (a tutorial video, ten sample utterances and a trick
  question); 984 of 1,759 applicants passed (56%). Each could do at most 30 tasks per
  project.
- Each task also held gold utterances (with an expert label) and consistency utterances;
  an annotator who scored 50% or less on either was removed and the recording's annotations
  were recomputed. Recordings with fewer than three annotators were annotated again.
- Annotators chose among `angry`, `happy`, `neutral`, `sad`, `frustrated` and `other`, and
  could tick more than one. Free-text `other` answers (for example calm, joyful, annoyed)
  were mapped by the authors to the nearest of the five emotions (paper, table 7).
- `majority_emo` is the emotion most annotators chose; it is `None` when no majority was
  reached (2,664 recordings) and `other` for 5 recordings. These 2,669 are never used.

**Agreement.** `agreement` is the share of annotator votes that went to the majority emotion,
each annotator's vote being split over the emotions they ticked (an annotator who ticks all
five gives each 1/5). With three annotators a 2-to-1 vote gives 0.667. It is 0 when there is
no majority. In the paper, the Krippendorff alpha of the whole corpus is 0.413, below the
0.667 that the authors take as the minimum for a reliable corpus; keeping only recordings
with agreement of at least 0.71 (14,182 recordings in version 1, 14,181 in version 2) gives an
alpha of 0.692, and the authors "strongly recommend" removing recordings below 0.71.

**Human recognition accuracy** (the share of recordings whose majority label equals the
emotion the actor was asked for, paper table 9): 0.592 over all 27,854 recordings and 0.772
over those with agreement of at least 0.71. By the authors' figures, `neutral` is recognized
best (78% overall, 93% after filtering), `happy` and `frustrated` worst (about 61% and 62%
overall); `frustrated` is often taken for `angry`, `sad` or `neutral`. For the scripted
sessions after filtering, the accuracy is 0.690 for low-intensity and 0.883 for high-intensity
recordings. The individual annotations (`annotated`: each annotator's chosen emotions) are
distributed with the data.

## What is used

The recipe uses the Hugging Face release `airesearch/thai-ser`, **version 2.0** (2025-04-15,
which fixed the calculation of `agreement`), pinned to one revision. Version 1.0 on GitHub
(`vistec-AI/dataset-releases`, 2021) has the same audio (compared byte for byte on the
recordings of the first of the 29 parquet files) and the same majority labels and votes; only
`agreement` differs, for 39 recordings. The threshold of 0.71 keeps
14,182 recordings in version 1 and 14,181 in version 2.

All 25,185 recordings with a majority label are converted: 7,359 `neutral`, 7,864
`frustrated`, 4,070 `happy`, 3,111 `angry` and 2,781 `sad` (37.7 of the 41.6 hours). They are
11,164 studio and 3,263 Zoom improvisations plus 8,612 studio and 2,146 Zoom scripted
recordings, from 200 actors. The majority label is the emotion the actor was asked for in
16,491 of them (65%). The source audio is 44.1 kHz FLAC, mono or stereo; `create_dataset`
converts it to 16 kHz mono 16-bit WAV, `num_workers` files at a time. Durations range from
0.29 to 108.8 seconds (median 4.7). `remove_long_short` drops the 53 recordings of
train/valid longer than 30 seconds; test is not filtered and holds 17 recordings longer than
30 seconds.

**Agreement is not used by default.** `min_agreement` in `dataset/config.yaml` is 0.0, so
every recording with a majority label goes into the manifests. Setting it to 0.71 follows
the authors' recommendation and keeps 14,181 recordings (delete `manifest/` and
`manifest_filtered/` and rerun `create_dataset` to rebuild; the converted audio is reused).
The splits are drawn before this filter, so a higher threshold only removes utterances from
the same train/valid/test split. `agreement.tsv` in the data directory lists every converted
recording with its split, label, the emotion the actor was asked for, the agreement and the
number of annotators.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `THAISER` — the directory holding `emotion_label.json` and the extracted audio, or where to download it to
- `THAISER_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the 29 parquet files of the release from Hugging Face (12.7 GB;
each file's size is checked), writes the usable recordings as FLAC files and the label columns
(votes, agreement, the emotion asked for, actor, session and so on) to `emotion_label.json`,
and deletes the parquet files. Reading them needs `pyarrow`. The release is CC BY-SA 4.0.

## Splits

The release publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so the splits share the actors,
and the three read sentences appear in all splits.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 17630 | 17582 |
| valid | 2518 | 2513 |
| test | 5037 | (not filtered) |

## Time needed

Measured on this machine. The download depends on the network.

| step | time |
|---|---|
| `create_dataset` (download of the 29 files, 12.7 GB, unpacking, conversion to 16 kHz with 16 workers) | 7 min 53 s; 14,182 of the 25,185 converted files already existed, so about 11,000 were converted |

The extracted FLAC files take 5.0 GB and the converted data 4.1 GB.

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

**Duplicates.** 3 recording(s) are byte-identical to another one (one per group is kept). They are listed in `duplicate_ids` and are dropped only when `exclude_duplicates` in `dataset/config.yaml` is `true` (default `false`).
