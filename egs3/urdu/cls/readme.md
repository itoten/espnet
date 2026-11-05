# URDU speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on URDU alone and scores it on URDU's own test split, classifying each
utterance into one of four emotions.

[URDU](https://github.com/siddiquelatif/URDU-Dataset) is Urdu speech cut from
talk shows on YouTube: 400 utterances, 100 each of `angry`, `happy`, `neutral`
and `sad`, 0.28 hours in total, each 1 to 3 seconds long. The README gives the
speaker count as 38 (27 male, 11 female), but the file names carry 29 distinct
speaker ids (18 male, 11 female). The corpus is described in Latif et al.,
"Cross Lingual Speech Emotion Recognition: Urdu vs. Western Languages".

According to the paper, the clips come from Urdu TV talk shows on YouTube and are
spontaneous: excerpts of unscripted discussions between guests, so the speakers' emotional
state was not controlled. The clips were collected from videos chosen by the discussion and
situation in the shows, and only for four emotions (happy, sad, angry and neutral); the paper
notes that music and other noise sometimes accompany the speech. The README adds that the
speakers were selected randomly. Every clip is cut to a fixed length: 200 are 2.0 s, 199 are
3.0 s and one is 0.99 s. There is no transcript.

The speaker ids are very uneven: the 29 ids have between 2 and 80 clips each (the five most
frequent have 80, 48, 40, 38 and 28), and 26 of the 29 ids appear with a single emotion
(two with two, one with three), so the speaker's voice is tied to the emotion. The paper's
own experiments are speaker-independent (30 of the 38 speakers for training and 8 for
testing); this recipe does not hold speakers out.

The label comes from the file name, `S<gender><speaker>_F<file>_<emotion><index>.wav`
(e.g. `SM1_F10_A010.wav`: male speaker 1, that speaker's 10th file, 10th
`angry` file); the emotion codes are A, H, N and S. Each file also sits in a
folder named after its emotion, and `create_dataset` fails if the two disagree.
The recordings are mono at 44.1 kHz, so it converts them to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `URDU` — the extracted repository archive, or where to download it to
- `URDU_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the GitHub repository `siddiquelatif/URDU-Dataset` (commit `4b546d4e0fa36d1c86c53a281631b2ab7ce2fdec`; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from GitHub, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 400 | 29 (README: 38) | 0.28 |

How these were counted: every `.wav` in the repository (44.1 kHz mono 16-bit; 200 clips are 2.0 s, 199 are 3.0 s and one is 0.99 s, so the clips were cut to a fixed length); hours from the file lengths. Speakers are the 29 distinct speaker IDs in the file names `S<gender><speaker>_F<file>_<emotion><index>.wav` (18 male, 11 female); the README and the paper say 38 speakers (27 male, 11 female), and the reason for the difference is not known.

### Files per label

| label (emotion folder; code in the file name) | samples | hours |
|---|---|---|
| Angry (A) | 100 | 0.069 |
| Happy (H) | 100 | 0.069 |
| Neutral (N) | 100 | 0.069 |
| Sad (S) | 100 | 0.069 |

How these were counted: the emotion folder each file is in (the letter A, H, N or S in the file name agrees). According to the paper, four students of two universities labelled the clips from audio and video and the clips on which at least two agreed were kept; the individual ratings are not in the release. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- The speaker IDs are very uneven: the 29 IDs have 2 to 80 clips each (the five most frequent have 80, 48, 40, 38 and 28), and 26 of the 29 IDs appear with a single emotion (two with two, one with three).
- The README says 38 speakers; the file names carry 29.
- The clips come from Urdu TV talk shows on YouTube (unscripted discussions); the paper notes that music and other noise sometimes accompany the speech. There is no transcript, and no license for the shows is given.


## Download

`prepare_source` downloads the repository archive from GitHub (about 86 MB once extracted,
pinned to the latest commit). The repository states that anyone can use the
data "only for research purposes" and carries no separate license file. The clips are cut
from YouTube videos of talk shows, and neither the README nor the paper says anything about
the rights to those videos.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out.

| split | utterances |
|---|---|
| train | 280 |
| valid | 40 |
| test | 80 |

## Listening test

**The labels are perceptual judgments** (paper, section III-A). After collecting the clips, the
authors mixed the files and gave them to four students of two universities (NUST and CIIT).
The students were asked to annotate the speaker's emotional state using both the audio and
the video, and a clip got a final label when at least two annotators assigned it the same
emotion. The paper does not give the number of clips shown or dropped, an agreement
coefficient or an accuracy, and the individual ratings are not distributed; the recipe uses
the final label of the file name. Because the annotators saw the video, the label may rely on
facial expression and context that the audio alone does not carry.

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
