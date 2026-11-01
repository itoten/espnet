# CSEMOTIONS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on CSEMOTIONS alone and scores it on CSEMOTIONS's own test split, classifying each
utterance into one of seven emotions.

[CSEMOTIONS](https://huggingface.co/datasets/ATH-MaaS/CSEMOTIONS) (released with the
Marco-Voice technical report, [arXiv 2508.02038](https://arxiv.org/abs/2508.02038)) is
a Mandarin Chinese acted corpus. Ten voice actors (5 male, 5 female, per the dataset
card) read one emotional sentence per clip in seven emotions: `angry`, `fear`
(`fearful` in the source), `happy`, `neutral`, `playfulness`, `sad` and `surprise`.
Measured on the downloaded data: 4,160 clips, 10.20 hours, 1.75 to 34.2 seconds long,
all 48 kHz mono 16-bit PCM wav (the dataset card says 24-bit). Counts per emotion:
angry 769, fear 623, playfulness 621, happy 603, sad 546, surprise 508, neutral 490.
Speakers are named `female001`-`female005` and `male001`-`male005`; each has 323 to 473
clips. No duplicate clips were found.

Each clip comes with the sentence read aloud (`text` column; the recipe does not use
it). The 1,296 sentences were written for one emotion each: none is read in more
than one emotion, and in the samples we read the wording often shows the emotion.
By ear
the emotions also differ in how they are spoken (listened to on 2026-10-23: four
clips per emotion, different speakers).

The label is the `emotion` column of the parquet files; `dataset/config.yaml` maps it
to the manifest label. `playfulness` is kept as its own class, since it has no
counterpart in the other recipes. The clips are converted to 16 kHz mono with ffmpeg
by `create_dataset`.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Hugging Face dataset `ATH-MaaS/CSEMOTIONS` (8 parquet files; repository revision `19498e62e6745cf9ec9b69a44b826caa467eac31`, last modified 2026-08-17; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Hugging Face, direct download, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 4,160 | 10 | 10.20 |

How these were counted: every row of the 8 parquet files (48 kHz mono 16-bit, 1.75 to 34.2 s); hours from the decoded audio lengths. Speakers are the 10 distinct speaker names (`female001` to `female005`, `male001` to `male005`; 323 to 473 clips each).

### Files per label

| label (the `emotion` column of the release) | samples |
|---|---|
| angry | 769 |
| fearful | 623 |
| playfulness | 621 |
| happy | 603 |
| sad | 546 |
| surprise | 508 |
| neutral | 490 |

How these were counted: the `emotion` column of the parquet files (the source spells `fearful`).

### Issues in the release

- The dataset card says 24-bit audio; the files are 16-bit.
- The repository's NOTICE file states that data from ESD (a corpus for research use only) was incorporated, without saying which clips. No overlap was found: clips of the same length as ESD's 35,000 were compared by waveform, and ESD's 350 Chinese sentences were compared with Whisper transcripts of these clips (closest sentence 19% similar).
- No duplicate clips were found.
- Observation by the recipe author (2026-10-23): four clips per emotion from different speakers were listened to; the emotions sound different in how they are spoken.

### Not stated in the documentation

- The 1,296 sentences (`text` column) are each read in one emotion only: none occurs in more than one emotion (checked by exact match).


## Provenance

Hugging Face dataset `ATH-MaaS/CSEMOTIONS` (the repository formerly named
`AIDC-AI/CSEMOTIONS` redirects there), pinned to commit
`19498e62e6745cf9ec9b69a44b826caa467eac31`. It is not gated. The dataset card gives
the licence as Apache-2.0.

The repository's `NOTICE` file also states that the dataset incorporates
Emotional-Speech-Data (ESD, MIT licence) and notes that "this database can only be
used for research purpose". It does not say which clips come from ESD. This recipe
treats the corpus as Apache-2.0 because no ESD material could be found in it
(checked on 2026-10-23):

- Audio: no clip is a near-duplicate of any of the 35,000 ESD clips (clips of the
  same length compared by waveform; parts of clips were not compared).
- Text: ESD's Chinese speakers all read the same 350 sentences (transcribed here
  with Whisper from one speaker). None of the 1,296 CSEMOTIONS sentences is close
  to any of them (at most 19% of character pairs shared).
- The Marco-Voice report describes CSEMOTIONS as newly recorded by its ten voice
  actors and lists ESD separately as another training set.

The research-only note may therefore come from the Marco-Voice project as a whole,
which also trained on ESD, but the authors do not say so. Check the `NOTICE` before
any non-research use.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `CSEMOTIONS` — the directory holding the eight parquet shards, or where to download them to
- `CSEMOTIONS_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads the eight parquet shards (3.08 GB in total) and checks each
SHA-256. Measured on one run: about 2 minutes to download; `create_dataset`
(parquet to 16 kHz wav) took about 249 seconds, `remove_long_short` 18 seconds and
`prepare_labels` under 1 second.

## Splits

The corpus publishes no partition. Whole speakers are held out
(`n_test_speakers`, `n_valid_speakers`, `split_seed` in `dataset/config.yaml`): with
seed 0, `male003` and `male004` form the test set, `female002` the valid set, and the
other seven speakers the training set. Scores are therefore speaker-independent, but
the sentences are shared: 848 of the 849 test sentences and all 378 valid sentences
are also read in the training set. The test speakers are both male.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 2,932 | 2,917 |
| valid | 378 | 378 |
| test | 850 | 850 (not filtered) |

`remove_long_short` drops the 15 train clips longer than 30 s.

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
