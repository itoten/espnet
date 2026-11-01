# YSED speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on YSED alone and scores it on YSED's own test split, classifying each
recording into one of five emotions.

[YSED](https://zenodo.org/records/15227219) (Yemeni Speech Emotion Dataset;
Derhem, AL-Mekhlafi, AL-Majmar and AL-Makhlafi, *Data in Brief* 63, 2025,
[doi:10.1016/j.dib.2025.112233](https://doi.org/10.1016/j.dib.2025.112233)) is
Yemeni Arabic speech (the first emotional speech dataset of the Yemeni dialect, according
to the paper). The Zenodo record and the paper's abstract describe 1,432 recordings from
71 volunteers (37 men and 34 women) aged 15 to 45; the paper's section on participants
says 68 participants (37 men and 31 women) aged 15 to 40 were selected after the judges
excluded three, and the paper gives both age ranges. All come from the Ibb Governorate and
speak the standard Yemeni dialect: 19 high school students, 43 university students and
6 government school teachers (68 in all), which a panel of judges rated beforehand as
professional (20), semi-professional (30) or amateur (18) at expressing emotions. The
emotions are `angry`, `fear` (`fearful` in the release), `happy`, `neutral` and `sad`;
there is no `disgust` or `surprise`.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 15227219 (YSED, Yemeni Speech Emotion Dataset; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,432 | 67 (participant numbers in the file names) | 1.01 |

How these were counted: every `.wav` under `YSED/<emotion>/` (48 kHz stereo 16-bit, nearly identical channels, 1.0 to 7.5 s); hours from the file lengths (the paper gives 59 min 10 s). The file names carry 67 distinct participant numbers (68 in combination with gender); the record and the paper's abstract say 71 volunteers (37 men and 34 women, aged 15 to 45), and the paper's section on participants says 68 (37 men and 31 women, aged 15 to 40). The number of files per participant ranges from 1 to 39 (53 participants have all five emotions).

### Files per label

| label (emotion folder in the release) | samples |
|---|---|
| neutral | 305 |
| sad | 303 |
| happy | 294 |
| angry | 286 |
| fearful | 244 |

How these were counted: the emotion folder each file is in (the initial letter of the file name agrees). According to the paper the initial 6,035 recordings were reduced to the 1,432 that at least four of six judges accepted as the intended emotion. The release has no disgust or surprise class.

### Issues in the release

- The counts of the paper's Table 1 (angry 268, happy 249, sad 303, fear 244, neutral 305, total 1,369) do not match the files (angry 286, happy 294, sad 303, fear 244, neutral 305, total 1,432).
- The numbers of participants (71, 68, 67 distinct numbers) and the age range (15 to 45 or 15 to 40) differ between and within the documents.

### Listening test ratings in the release

Not in the release: according to the paper, six native Yemeni judges (a university faculty member, a dentist, an accountant, an interior designer, a school teacher and a company employee) rated every recording blind to the target emotion and kept a recording only if at least four accepted it (Fleiss' kappa 0.9); the single judgements are not distributed.


## How it was recorded

The participants read five sentences per emotion, four repetitions for each of the four
emotions and one for `neutral`, which makes 85 recordings per participant and an initial
6,035 (71 x 85). The sentences were selected per emotion (paper, figure 4). By the paper, 80%
of the recordings used the "induced" approach (the participant is first prompted to feel the
emotion, then reads the sentence) and 20% the "simulated" one (the participant reads the
sentence while consciously expressing the emotion). Recording took place at Ibb University
(College of Science, Department of Information Technology), "in as quiet an environment as
possible", with the microphone 25 to 30 cm from the mouth, and a researcher reviewed every
recording; the participant re-recorded those with errors. The paper's ethics statement
says that written informed consent was obtained from all adult participants and from the
legal guardians of minors, verified by the Ibb University Research Ethics Committee.

## What is used

All 1,432 recordings (1.01 hours). The label is the name of the folder the file
is in; the file name's first letter (`a`, `f`, `h`, `n`, `s`) is checked against it.
The release is 48 kHz stereo; `create_dataset` converts it to 16 kHz mono 16-bit
WAV. Durations range from 1.0 to 7.5 seconds (median 2.3), so
`remove_long_short` drops nothing.

The file names are `<emotion code>-<sentence 1-5>-<repetition 1-4>-<gender>-<participant>.wav`
(for example `n1-02-01-01-15.wav`). The emotion codes are `a5`, `f2`, `h3`, `n1` and `s4`; the
gender is `01` for women and `02` for men. The participant number takes 67 values in the files
(68 together with the gender; the paper numbers the participants 1 to 68), and the number of
recordings per participant ranges from 1 to 39: 53 participants have all five emotions, the
others 1 to 4. The recipe reads only the folder and the first letter and does not use the
participant.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `YSED` — the directory holding the `YSED/<emotion>/` folders, or where to download it to
- `YSED_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `YSED-v1.0.0.zip` (506 MB) from Zenodo, checks it against
the MD5 Zenodo lists, drops the archive's top-level folder while extracting, and
counts the 1,432 files. The zip is deleted afterwards. The release is CC BY 4.0.

## Splits

The release publishes no partition. Recordings are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Participants are not held out, so the splits share them.

| split | recordings | after `remove_long_short` |
|---|---|---|
| train | 1003 | 1003 |
| valid | 142 | 142 |
| test | 287 | (not filtered) |

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`). The download
depends on the network, and another large download ran on the same connection
for part of this one, so treat it as an upper bound.

| step | time |
|---|---|
| `prepare_source` (download of the 506 MB zip, MD5 check, extraction) | 10 min 10 s |
| `create_dataset` (conversion to 16 kHz, one file at a time) | 88.6 s |
| `remove_long_short` | 18.2 s |
| the whole `create_dataset remove_long_short prepare_labels` command | 1 min 55 s (includes starting `pixi` and Python) |

The extracted audio takes 667 MB and the converted data 114 MB.

## Listening test

**The recordings were judged by listeners** (paper, section 4, "Judges and Evaluation Process"). A panel of six
judges, all native Yemeni speakers fluent in the dialect and from different professions (a
university faculty member, a dentist, an accountant, an interior designer, a school teacher
and a company employee), reviewed every recording independently. According to the paper they
were blinded to the target emotion and rejected a recording whose emotional expression was
ambiguous. A recording was kept only if at least four of the six judges accepted it as
expressing the intended emotion; for example, a recording of anger that some judges took for
fear was kept only if at least four judges agreed it was anger. Of the initial 6,035
recordings, 1,432 remained (59 minutes 10 seconds by the paper). The paper reports a Fleiss'
kappa of 0.9 among the six judges for the dataset. The number of recordings each judge
accepted or rejected, and the individual judgements, are not distributed; the recipe uses the
emotion of the folder.

The paper's table 1 gives 268 `angry`, 249 `happy`, 303 `sad`, 244 `fear` and 305 `neutral`
recordings (1,369 in all), which does not match the release: it has 286 `angry`, 294 `happy`,
303 `sad`, 244 `fear` and 305 `neutral` files (1,432).

## Reference

Derhem, AL-Mekhlafi, AL-Majmar and AL-Makhlafi, "YSED: Yemeni Speech Emotion Dataset",
*Data in Brief* 63 (2025), 112233, doi:10.1016/j.dib.2025.112233.

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
