# BERSt speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on BERSt alone and scores it on BERSt's own test split, classifying each
utterance into one of seven emotions.

[BERSt](https://huggingface.co/datasets/Rosie-Lab/BERSt) (Basic Emotion Random
phrase Shouts) is an English corpus of 98 actors who recorded 13 nonsense phrases
at home on their own smartphones, asking each to speak, raise their voice and
shout the phrase while moving the phone to different places in the house. The
recipe uses the speech segments the authors cut from those recordings
(4,523 clips, 3.73 hours, 1.03 to 22.5 seconds long); 63 of them are left out
(see "Clips left out"), so 4,460 are used. The clips are mono, 48 kHz
(3,704) or 44.1 kHz (819), so `create_dataset` converts them to 16 kHz with ffmpeg.

The label is the `affect` column of the metadata: the emotion the actor was asked
to portray. The dataset card states that these labels were not validated by
listeners. `dataset/config.yaml` maps `anger`, `joy` and `sadness` to `angry`,
`happy` and `sad`; `disgust`, `fear`, `neutral` and `surprise` are kept. All seven
classes are roughly balanced in train (436 to 561 clips).

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Hugging Face dataset `Rosie-Lab/BERSt` (repository revision `fed35c477427cff44206464b7f85e68d1fc99062`, last modified 2025-09-06; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Hugging Face, direct download, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 4,523 | 98 | 3.73 |

How these were counted: every row of the `metadata.csv` of the `train`, `validation` and `test` folders (mono, 48 kHz in 3,704 clips and 44.1 kHz in 819, 1.03 to 22.5 s); hours from the clip lengths. Speakers are the distinct `user_id` values.

### Files per label

| label (the `affect` column of the release) | samples |
|---|---|
| fear | 684 |
| disgust | 677 |
| joy | 657 |
| neutral | 657 |
| anger | 651 |
| surprise | 613 |
| sadness | 584 |

How these were counted: the `affect` column of the three `metadata.csv` files. The dataset card states that the labels were not validated by listeners.

### Issues in the release

- 31 groups (82 clips) of byte-identical content: 24 groups are different recordings with the same sound, 2 are different chunks of one recording, and 5 carry metadata of a different speaker, emotion or phrase (one of these spans validation and test); in the 5 the metadata of one of the copies is wrong.
- Some clips hold the three levels of voice (speak, raise, shout) of one recording uncut: the phrase is repeated two or three times (11 clips found).
- Some clips start with a voice reading the emotion name aloud ("Anger."; 3 clips of one speaker).
- Only the 86 clips longer than 6 s were transcribed with Whisper to find the last two problems; about 4,437 clips of 6 s or less were not checked.
- Observation by the recipe author (2026-10-23): the 31 duplicate groups were listened to; in the 5 groups with conflicting metadata the correct emotion could not be decided.
- The card states 4,526 clips; the release has 4,523. The paper's Table 7 (emotion counts per split) does not match the counts of the files.


## Clips left out

63 clips are listed in `exclude_files` in `dataset/config.yaml` and skipped. The
list was made from the 4,523 clips; whether other clips share the problems below
was not checked everywhere (see the last paragraph).

- **Identical audio (52 clips, not counting the 6 below).** 31 groups of clips have byte-identical audio
  (MD5 of the raw files). In 24 groups the same speaker, emotion and phrase were
  uploaded in several recordings; one clip per group (the first by utterance id)
  is kept and 42 are left out. In 5 groups the metadata disagrees (a different
  speaker, emotion or phrase on the same audio, in train, and one pair across
  valid and test), so both copies, 10 clips, are left out; the label cannot be
  trusted either way. In 2 groups, three chunks of one recording are identical
  (see the next item).
- **Several takes in one clip (11 clips).** The three chunks of one recording
  (`chunka`, `chunkb`, `chunkc`) sometimes hold the phrase repeated, as in
  "A. phrase B. phrase C. phrase": the actor appears to have recorded three
  levels in one take and the cut was not made. These are left out: the 6 identical
  clips above (7.85 s and 22.5 s), three clips of speaker `bfd2fa` (the chunks
  hold all three, the last two, or the last one of the repetitions, plus a spoken
  "Anger." at the start), one clip of `5b6bd2` (phrase twice) and one of `0140dc`
  ("A. B. C." read aloud). The `chunkc` clip of `bfd2fa` was dropped only for the
  spoken label.
- They could be used after trimming each by hand; that was not done.

Only the 86 clips longer than 6 seconds were transcribed (Whisper large-v3) and
checked for a repeated phrase (31 of them were not recognised as the phrase,
mostly because the speech was hard to hear; the ones listened to were fine).
The other 4,437 clips were not transcribed, so repeated phrases or spoken labels
in shorter clips may remain.

## Provenance

Hugging Face dataset `Rosie-Lab/BERSt`, pinned to commit
`fed35c477427cff44206464b7f85e68d1fc99062`, licence CC BY 4.0 as stated on the
card; no login is needed. All 4,523 clips match the SHA-256 listed by the Hub.
The authors are Paige Tuttösí et al.; please cite their paper:

```bibtex
@article{tuttosi2025bersting,
  title   = {BERSting at the Screams: A Benchmark for Distanced, Emotional and Shouted Speech Recognition},
  author  = {Paige Tuttösí and Mantaj Dhillon and Luna Sang and Shane Eastwood and Poorvi Bhatia and Quang Minh Dinh and Avni Kapoor and Yewon Jin and Angelica Lim},
  journal = {Computer Speech & Language},
  volume  = {95},
  pages   = {101815},
  year    = {2026},
  doi     = {10.1016/j.csl.2025.101815},
}
```

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `BERST` — the directory holding `berst-data/{train,validation,test}/`, or where to download them to
- `BERST_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` fetches the three `metadata.csv` files and each listed clip (1.27 GB
in total) with 8 parallel connections. Measured on one run: about 14.5 minutes to download, then
about 4 minutes to convert, so `create_dataset` took 1,107 seconds;
`remove_long_short` and `prepare_labels` together about 18 seconds.

## Splits

The published train / validation / test split is used. It is speaker-disjoint
(78, 10 and 10 actors, verified from `user_id`).

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 3,445 | 3,445 |
| valid | 487 | 487 |
| test | 528 | 528 (not filtered) |

The test split is not balanced: `sad` has only 26 clips (angry 99, fear 99, neutral
91, happy 79, surprise 68, disgust 66), so per-class scores for `sad` are noisy.

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
