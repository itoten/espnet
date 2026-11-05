# WHiSER speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on WHiSER alone and scores it on WHiSER's own test split, classifying each
utterance into one of eight emotions.

WHiSER (White House Tapes Speech Emotion Recognition corpus; Naini et al.,
Interspeech 2024) is built by the MSP lab (UT Dallas), the developers of
MSP-Podcast, from President Nixon's Oval Office and other recorded conversations
of 1971 to 1973 (the declassified Nixon tapes). It is not a podcast corpus: the
speech is natural, with distant microphones, reverberation and background noise,
which makes the domain very unusual compared with the other corpora here. Each of
the 5,427 segments has a consensus emotion class (the plurality vote of at least five
annotators from a pool of 33 students) and arousal / valence / dominance scores (1 to 7). Per-annotator
primary and secondary emotion votes are in `Labels/labels_detailed.csv` of the corpus
repository; this recipe uses only the consensus class. All files are mono WAV
(5,255 at 44.1 kHz and 172 at 22.05 kHz), 3.01 to 10.99 seconds long, 6.35 hours in
total. Speaker ids and gender are `Unknown` for every file, and the paper reports the
neutral class at 64% of the segments.

The label is the consensus class `EmoClass` of `labels_consensus.csv`
(`A` angry, `C` contempt, `D` disgust, `F` fear, `H` happy, `N` neutral, `S` sad,
`U` surprise). As in the msppodcast recipe, `O` (free-text other, 47 files) and `X`
(no agreed label, 916 files) have no usable class and are dropped, leaving 4,464
utterances. Of these, 57 are left out as duplicates (below), so 4,407 are used. The
label counts of the 4,407 are very uneven: neutral 3,447, happy 341, angry 290,
sad 279, surprise 33, contempt 10, fear 6, disgust 1.

Some segments are the same sound under different names (53 clusters, 107 files,
mostly the same tape in names that differ in one letter, for example `rmn_e655a`
and `rmn_e655b`, although the paper says duplicates were removed). In a cluster
whose files have one label, only the first file by name is kept; the 3 clusters whose
files carry different labels are left out entirely. The 57 files are listed in
`exclude_files` in `dataset/config.yaml`.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `WHISER` — the directory holding `wav/` and `labels_consensus.csv`, or where to download them to
- `WHISER_OUTPUT` — where the converted 16 kHz audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the WHiSER corpus: the audio at `lab-msp.com/WHiSER/Annotation_wav_files/` and `Labels/labels_consensus.csv` of the GitHub repository `msplabresearch/WHiSER` (downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download of the wav files and the consensus table, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 5,427 | unknown | 6.35 |

How these were counted: every row of `labels_consensus.csv` (equal to the wav files; mono, 44.1 kHz in 5,255 files and 22.05 kHz in 172, stored as 32-bit float; 3.01 to 10.99 s); hours from the file lengths. `SpkrID` and `Gender` are `Unknown` in every row, so the number of speakers is unknown.

### Files per label

| `EmoClass` code in `labels_consensus.csv` | read as (corpus documentation) | samples | hours |
|---|---|---|---|
| N | neutral | 3,492 | 4.105 |
| X | no agreed label | 916 | 1.089 |
| H | happy | 345 | 0.407 |
| A | angry | 293 | 0.308 |
| S | sad | 284 | 0.319 |
| O | other (free text) | 47 | 0.065 |
| U | surprise | 33 | 0.038 |
| C | contempt | 10 | 0.017 |
| F | fear | 6 | 0.006 |
| D | disgust | 1 | 0.001 |

How these were counted: the `EmoClass` column. The table also holds arousal, valence and dominance scores (`EmoAct`, `EmoVal`, `EmoDom`) and a `Split_Set` column. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- The audio is not 16 kHz 16-bit as the paper states.
- 53 clusters (107 files) among the 4,464 files with a letter class other than `O` and `X` are the same sound under names that often differ in one letter only (for example `rmn_e655a` and `rmn_e655b`); in 3 of them the labels differ (neutral/sad, sad/neutral, angry/neutral). The paper says duplicates were removed. Four pairs (8 files) are byte-identical.
- The repository README names a table linking the segments to the original recordings; the link returns HTTP 404. The repository states no license or citation terms.
- The recordings are conversations from the declassified Nixon tapes (natural speech with distant microphones, reverberation and background noise); the paper does not distribute splits (the corpus is meant as an independent test set).

### Listening test ratings in the release

The corpus repository holds `Labels/labels_detailed.csv` with each annotator's primary and secondary emotion votes (the paper: at least five annotators from a pool of 33 students per segment); that file was not part of the downloaded release.


## Download

`prepare_source` fetches the label table from the corpus GitHub repository and the
5,427 WAV files from the lab's public directory listing (no login, no archive),
with 4 parallel workers. Measured on one run, the audio download took about 16
minutes (3.7 GB); `create_dataset`, `remove_long_short` and `prepare_labels`
together took about 3 minutes 47 seconds. No licence is stated (neither the
repository nor the paper gives terms). It is to be asked of the authors (not yet
done); until then use the data only for local experiments and do not redistribute it.

## Splits

The release publishes no partition (`Split_Set` is `NA`). Files sharing the
prefix before the first `.` of the file name (1,089 prefixes, e.g. `001-105`) are
kept in the same split, as a guess at "same source recording" that was not checked
against the corpus metadata. Prefixes are shuffled with `split_seed` and filled
to 70/10/20.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 3,060 | 3,060 |
| valid | 458 | 458 |
| test | 889 | 889 (not filtered) |

Because of the class imbalance, the rare classes are not in every split: train has
7 contempt, 1 disgust and 3 fear, valid has no contempt or disgust, and test has no
disgust. Macro-F1 over the classes present
in the test labels is therefore dominated by a few samples for contempt, fear
and surprise.

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
