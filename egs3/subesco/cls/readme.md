# subesco speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on subesco alone, and scores it on subesco's own test split (7
classes: `angry`/`disgust`/`fear`/`happy`/`neutral`/`sad`/`surprise`).

[SUBESCO](https://doi.org/10.5281/zenodo.4526477) is 7,000 Bangla utterances:
20 actors (10 male, 10 female) each performing ten sentences in seven
emotions with five takes apiece. Every field comes from the file name
(`F_01_OISHI_S_1_ANGRY_1.wav`). The audio is 44.1 kHz (4,555 files) or 48 kHz
(2,445 files), so `create_dataset` resamples it to 16 kHz. `SUBESCO` points at an
existing corpus tree, or where to download it to; the Zenodo archive is checked
against its published MD5.

724 of the files are stereo with an almost silent right channel (the median level is
-13 dB on the left and -81 dB on the right), so only the left channel is kept; the
mono files are converted as they are. 31 pairs of takes are the same performance
(decoded waveform correlation 1.0000; 2 pairs identical sample for sample, the others
differ by a few tens of milliseconds of padding or under 7% of the peak). The
first file of each pair is kept and the second is left out (`exclude_files` in
`dataset/config.yaml`), so 6,969 recordings are used (994 angry, 995 disgust, 998 fear,
997 happy, 997 neutral, 992 sad, 996 surprise). The transcripts of the ten sentences
are in the paper, not in the corpus. A listening experiment with 50 people (71%
accuracy) is reported only in aggregate.

The licence is CC BY 4.0 (Zenodo); the corpus paper is cited by its DOI,
[10.1371/journal.pone.0250173](https://doi.org/10.1371/journal.pone.0250173) (PLOS ONE).

No partition is published; speakers are dealt out in numeric order, the same
number of each gender per split, so the split is balanced by construction.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 4526477 (SUBESCO; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in (the archive is checked against its published MD5).

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 7,000 | 20 | 7.83 |

How these were counted: every `.wav` in the archive (44.1 kHz in 4,555 files and 48 kHz in 2,445; 724 of them stereo; 2.75 to 6.03 s); hours from the file lengths. Speakers are the 20 names in the file names `F_01_OISHI_S_1_ANGRY_1.wav` (10 male, 10 female; 10 sentences, 7 emotions, 5 takes each).

### Files per label

| emotion in the file name | samples |
|---|---|
| ANGRY | 1,000 |
| DISGUST | 1,000 |
| FEAR | 1,000 |
| HAPPY | 1,000 |
| NEUTRAL | 1,000 |
| SAD | 1,000 |
| SURPRISE | 1,000 |

How these were counted: the emotion field of each file name.

### Issues in the release

- In the 724 stereo files the right channel is almost silent (median level -13.0 dB on the left, -81.5 dB on the right).
- 31 pairs of takes are the same performance: decoded waveform correlation 1.0000, 2 pairs identical sample for sample, the others differing by about 0.02 s of padding or under 7% of the peak amplitude. 30 pairs have the same speaker, sentence and emotion with different take numbers; one is `M_09` sentence 4 take 5 and sentence 5 take 1 (neutral).
- Observation by the recipe author (2026-10-26 at the latest): the 31 pairs were listened to and are the same sound.
- The sampling rate and channel count are mixed (the README says 48 kHz).
- The transcripts of the ten sentences are in the paper, not in the release.


## Quick start

```bash
# 1) Write the manifests
python run.py --stages create_dataset remove_long_short prepare_labels \
    --training_config conf/training.yaml

# 2) Collect feature statistics
python run.py --stages collect_stats \
    --training_config conf/training.yaml

# 3) Train
python run.py --stages train \
    --training_config conf/training.yaml

# 4) Infer
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 5) Score
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`SUBESCO_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.
