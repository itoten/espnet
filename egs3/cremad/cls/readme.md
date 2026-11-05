# cremad speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on cremad alone, and scores it on cremad's own test split (6
classes: `angry`/`disgust`/`fear`/`happy`/`neutral`/`sad`).

[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D) is already
16 kHz mono WAV, so nothing is converted. The label comes from the file name
(`<actor>_<sentence>_<emotion>_<intensity>.wav`); no annotation file is read.
`CREMAD` points at an existing corpus tree, or where to clone it to.

The corpus has no official partition. Speaker ids are sorted and dealt out to
train/valid/test, which keeps the split reproducible from the corpus alone;
`dataset/config.yaml` holds the per-split speaker counts.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the GitHub repository `CheyneyComputerScience/CREMA-D` (commit `1658cd342dff90010aa843eaeebd53610a08b1dc`; WAV, MP3 and FLV versions via Git LFS; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Cloned from GitHub (Git LFS), no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 7,442 | 91 | 5.3 (approx.) |

How these were counted: every `.wav` in `AudioWAV/` (16 kHz mono, 1.27 to 5.01 s; the MP3 and FLV folders hold 7,442 files each too); the hours are the figure from the corpus description (about 5.3), not re-measured. Speakers are the 91 actor numbers in the file names `<actor>_<sentence>_<emotion>_<intensity>.wav` (48 male, 43 female, aged 20 to 74).

### Files per label

| emotion code in the file name | samples | hours |
|---|---|---|
| ANG | 1,271 | 0.912 |
| DIS | 1,271 | 0.981 |
| FEA | 1,271 | 0.887 |
| HAP | 1,271 | 0.827 |
| NEU | 1,087 | 0.734 |
| SAD | 1,271 | 0.916 |

How these were counted: the emotion field of each file name (12 sentences; four intensity levels: low, medium, high and unspecified). Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- Five pairs of clips have the same sound, always from one actor. In four pairs the names carry different emotions (`1006_TIE_HAP_XX` / `1006_TIE_NEU_XX`, `1013_WSI_DIS_XX` / `1013_WSI_SAD_XX`, `1017_IWW_ANG_XX` / `1017_IWW_FEA_XX`, `1067_IEO_DIS_LO` / `1067_IEO_FEA_LO`); the fifth pair has two sentences with the same emotion (`1045_ITH_DIS_XX` / `1045_IWL_DIS_XX`).

### Not stated in the documentation

- Sentence codes `DFA`, `IEO`, `IWW` and `TAI` carry the same text in SES-ED (English) and, as translations, in SES-SD; SES-ED's `TSI` has the same code but another sentence.

### Listening test ratings in the release

The repository holds the raters' votes (`finishedResponses.csv`, `finishedEmoResponses.csv`, `processedResults/tabulatedVotes.csv` and `summaryTable.csv`; 2,443 crowd workers rated 90 clips each: 30 audio only, 30 video only and 30 both). Counted from `processedResults/`: the majority vote of the audio-only ratings equals the acted emotion for 3,388 of the 7,442 clips (45.5%), against 5,136 for the face alone and 5,569 for audio and video together.


## What the recipe uses

CREMA-D (Cao et al., IEEE Transactions on Affective Computing 2014) has 7,442 clips
from 91 actors (48 male, 43 female, aged 20 to 74, of several ethnicities), who each
read 12 sentences in six emotions and four intensity levels (low, medium, high,
unspecified). The audio is 1.27 to 5.0 seconds long, about 5.3 hours in total.
The corpus also holds the video and the MP3 and FLV versions, which this recipe does not use.

The label is the emotion the actor was asked to act. Crowd workers (2,443 people)
also rated the clips; their votes are in `processedResults/`. Judged by the audio alone,
the majority vote equals the acted emotion for only 3,388 of the 7,442 clips (45.5%),
against 5,136 for the face alone and 5,569 for audio and video together. The ratings are
not used here; the label is the acted emotion only.

Five pairs of clips have the same sound, always from one actor. In four pairs the names
carry different emotions (`1006_TIE_HAP_XX`/`1006_TIE_NEU_XX`,
`1013_WSI_DIS_XX`/`1013_WSI_SAD_XX`, `1017_IWW_ANG_XX`/`1017_IWW_FEA_XX`,
`1067_IEO_DIS_LO`/`1067_IEO_FEA_LO`); the fifth pair has two sentences with the same
emotion (`1045_ITH_DIS_XX`/`1045_IWL_DIS_XX`). All are in the training split and are
kept as they are, for the record.

The sentence codes `DFA`, `IEO`, `IWW` and `TAI` are also used, with the same text, by
SES-ED (English) and, as translations, by SES-SD. SES-ED's `TSI` has the same code but a
different sentence ("The sun is very bright", against "The surface is slick" here). So
the same texts occur when these corpora are used together.

## License and citation

Open Database License 1.0 for the database, Database Contents License 1.0 for the
contents (`LICENSE.txt`). Cite Cao, Cooper, Keutmann, Gur, Nenkova and Verma, "CREMA-D:
Crowd-sourced Emotional Multimodal Actors Dataset", IEEE Transactions on Affective
Computing 5(4), 2014.

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

`CREMAD_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.

**Duplicates.** Five pairs of recordings have the same sound (identical after decoding, or a
waveform correlation of 0.998 or more). One pair has the same label (`1045_ITH_DIS_XX` and
`1045_IWL_DIS_XX`, both `disgust`); in four pairs the two recordings carry different emotions
(`1006_TIE_HAP_XX`/`NEU`, `1013_WSI_DIS_XX`/`SAD`, `1017_IWW_ANG_XX`/`FEA`, `1067_IEO_DIS_LO`/`FEA`).
All are used by default. `exclude_duplicates: true` in `dataset/config.yaml` drops the second
recording of the same-label pair (`duplicate_ids`), and `exclude_conflicting: true` drops the
eight recordings of the different-label pairs (`conflicting_ids`); both default to `false`.
