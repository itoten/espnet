# TESS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on TESS alone and scores it on TESS's own test split, classifying
each utterance into one of seven emotions.

[TESS](https://doi.org/10.5683/SP2/E8H2MF) (Toronto emotional speech set) has
two female speakers, aged 26 (`YAF`) and 64 (`OAF`), each saying 200 target
words in the carrier phrase "Say the word ___" in seven emotions: 2800
utterances labelled `angry`, `disgust`, `fear`, `happy`, `sad`, `neutral` and
`surprise` (`ps`, "pleasant surprise", in the file names).

According to the corpus's paper (Dupuis & Pichora-Fuller, 2011, Canadian Acoustics 39(3)),
both actresses are native English speakers recruited from the Toronto area, with clinically
normal hearing, and recorded alone in a sound-attenuating booth. The words are the 200 items
of the NU-6 auditory test, each ending a carrier phrase. Each emotion was recorded in one
block, and the experimenter asked for repetitions until the emotion was judged to be
conveyed. The paper reports 2607 recorded stimuli for the younger actress and 3004 for the
older one; three students and the experimenter then listened to them and picked, for each
word and emotion, the token most representative of the emotion. The experimenter and at
least one student chose the same token for 80% (younger) and 92% (older) of the items. The
2800 files are those selected tokens. The actresses consented to the recordings being used as
stimuli for research, in presentations at scientific or professional conferences and in
public education.

The label comes from the file name, `<speaker>_<word>_<emotion>.wav`; no
annotation file is read. The recordings are 24.4 kHz (one is 96 kHz), so
`create_dataset` resamples all of them to 16 kHz with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `TESS` — an existing directory of the WAV files, or where to download it to
- `TESS_OUTPUT` — where the resampled audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the TESS (Toronto emotional speech set) record in Borealis Dataverse (doi 10.5683/SP2/E8H2MF; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Download from Borealis Dataverse, no sign-in. The record offers no archive; the 2,800 files were fetched through a zip that Dataverse builds on request (about 270 MB). The raw dataset also contains a `MANIFEST.TXT` as a 2,801st file.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 2,800 | 2 | 1.60 |

How these were counted: every `.wav` in the download (24.4 kHz, one file at 96 kHz; 1.25 to 2.98 s); hours from the file lengths. Speakers are two women: `YAF` (aged 26) and `OAF` (aged 64), each saying 200 target words in the carrier phrase "Say the word ___".

### Files per label

| emotion in the file name | samples | hours |
|---|---|---|
| angry | 400 | 0.205 |
| disgust | 400 | 0.272 |
| fear | 400 | 0.184 |
| happy | 400 | 0.219 |
| neutral | 400 | 0.228 |
| ps (pleasant surprise) | 400 | 0.223 |
| sad | 400 | 0.267 |

How these were counted: the last field of the file name `<speaker>_<word>_<emotion>.wav`. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- No byte-identical, decoded-identical or near-identical recordings were found.


## Download

The dataset on Borealis Dataverse has no archive file. `prepare_source` asks
Dataverse to zip all 2800 files (about 270 MB) and extracts the result. The
corpus is CC BY-NC 4.0.

## Splits

The corpus publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Neither speakers nor words are held out, so all three
splits contain both speakers, and a word in test also appears in training.

| split | utterances |
|---|---|
| train | 1960 |
| valid | 280 |
| test | 560 |

## Listening test

**The recordings were evaluated by listeners** (Experiment 2 of the paper above). Fifty-six
University of Toronto undergraduates (mean age 19.7 years; native English speakers with
clinically normal hearing) each listened to the stimuli of either the younger or the older
actress, with the same number of stimuli per emotion (the number per listener is not
stated). They listened over a loudspeaker at about 70 dBA and answered on a touch screen
which emotion the speaker was portraying. The overall accuracy was 82% (SD 11.08), about six
times the chance level. Anger and sadness were recognized best, disgust and pleasant
surprise worst; the per-emotion values are given only in a figure. The effect of the talker
was not significant. The individual responses are not in the download; the recipe uses the
intended emotion of the file name.

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
