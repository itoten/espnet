# Dusha speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Dusha alone and scores it on Dusha's own test split, classifying each
recording into one of four emotions.

[Dusha](https://github.com/salute-developers/golos/tree/master/dusha) (Kondratenko et
al., arXiv:2212.12266) is Russian speech, annotated on a
crowd-sourcing platform. The release has a crowd
subset (`crowd.tar`) and a podcast subset (audio is not distributed for the
podcasts, only links, for licensing reasons). According to the paper, the crowd
subset is acted: non-professional actors on Yandex Toloka read phrases taken
from requests to a voice assistant (chit-chat, pre-selected with a text
classifier so that emotional phrases are over-represented), each phrase with one
of two suggested emotions; 201,850 recordings from 2,068 speakers, with
differing equipment and background noise (children, animals, street). Only the
crowd subset is used. The labels are `angry`, `sad`,
`neutral` and `happy` (the release calls happiness `positive`; it is renamed to
match the other recipes). There is no
`disgust`, `fear` or `surprise`.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Dusha crowd subset (`crowd.tar` with `crowd_train/`, `crowd_test/` and `paper_setups.tgz` from the `salute-developers/golos` project; downloaded around 2026-10-15). The podcast subset is not distributed as audio (links only) and was not obtained. All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from the project's file server, no sign-in. Only the recordings named by the two official lists were extracted from the roughly 30 GB `crowd.tar`.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 160,366 (extracted) | 1,870 train and 190 test `source_id` values | 202.5 |

How these were counted: every `.wav` under `crowd_train/wavs` and `crowd_test/wavs` that the official lists name (16 kHz mono 16-bit, 0.86 to 29.88 s, median 4.34 s); hours from the file lengths. The rest of `crowd.tar` (the paper counts 201,850 crowd recordings) was not extracted. The speakers are the `source_id` values of the raw tables (a unique ID of the speaker or podcast; none in train/test overlap; 1,517 control-task recordings have none); the paper counts 2,068 speakers.

### Files per label

| label (the `emotion` field of the official lists) | samples | hours |
|---|---|---|
| neutral | 106,065 | 128.296 |
| sad | 23,190 | 32.837 |
| angry | 15,719 | 21.133 |
| positive | 15,392 | 20.222 |

How these were counted: the `emotion` field of `paper_setups/crowd_large.jsonl` (146,715 recordings, train) and `paper_setups/test/crowd_test.jsonl` (13,651, test), obtained with the Dawid-Skene aggregation of the crowd annotations. The paper's table lists 164,387 crowd utterances including an `other` class and the lists 160,366; whether the lists follow the paper's rule exactly (agreement threshold 0.9) was not checked. The release has no disgust, fear or surprise class. Hours per label are measured in the same way as the total hours above and summed over the rows of that label.

### Issues in the release

- 9 recordings in 4 groups (and 10 more in 5 groups) are the same sound: in 4 groups (9 files) the labels differ (sad and neutral); in 5 groups the labels agree.
- The aggregated label and the actor's suggested emotion (`speaker_emo` in the raw tables) agree for 71.5% of the recordings: 51% of those suggested sad, 34% of angry and 35% of positive are aggregated as neutral.
- 1,517 control-task recordings with known correct answers and many annotations each are in the training list; they have no sentence and no speaker ID.
- The license is a custom text, not a Creative Commons license.

### Listening test ratings in the release

`crowd_train/raw_crowd_train.tsv` and `crowd_test/raw_crowd_test.tsv` hold every single annotation (898,136 rows for train and 78,205 for test; the annotator's emotion among neutral, sad, positive, angry and other, an annotator ID, and for the control tasks the golden answer). Annotators listened to the audio only, without the text (paper).


## What is used

The labels are the release's own aggregated labels, read from the two files in its
`paper_setups.tgz` (`crowd_large.jsonl` for train, `crowd_test.jsonl` for test).
The paper aggregates the annotations with the Dawid-Skene algorithm and an
agreement threshold of 0.9; whether these two files follow exactly that rule was
not checked (the paper's table lists 164,387 crowd utterances including an
`other` class, the files 160,366). These files list 146,715 train and 13,651 test utterances (160,366,
202.5 hours); 9 of them are left out as same-sound utterances with conflicting
labels (see "Duplicates"). The rest of `crowd.tar` (utterances that are in neither file) is not
extracted. Durations range from 0.86 to 29.88 seconds (median 4.34), so
`remove_long_short` drops nothing. Every recording is already 16 kHz mono 16-bit
WAV, so `build()` references the files where they lie and converts nothing. A file
in any other format would be converted into `$DUSHA_OUTPUT/wav/`.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `DUSHA` — the directory holding the extracted `crowd_train/`, `crowd_test/` and `paper_setups/`, or where to download it to
- `DUSHA_OUTPUT` — where the manifests are written

## Download

`prepare_source` downloads `paper_setups.tgz` (79 MB) and `crowd.tar` (29.97 GB,
about 28.6 GiB) from the release's CDN, checks each by byte count (no checksum is
published), and extracts only the listed WAVs and the two annotation files
(`raw_crowd_train.tsv`, `raw_crowd_test.tsv`). The archive is deleted afterwards
(`remove_archive`). Make sure the disk has about 30 GB free during the download and
23 GB afterwards. An interrupted download (`crowd.tar.part`) is resumed.

**Expired certificate (since 2026-11-04).** The CDN's TLS certificate (Let's Encrypt)
expired on 2026-11-04 and has not been renewed (checked 2026-11-10), so the download
stops with "certificate has expired". To download anyway, set
`allow_expired_certificate: true` in `dataset/config.yaml`, or
`DUSHA_ALLOW_EXPIRED_CERTIFICATE=1` in the environment. This skips only the check of
the certificate's validity dates: the issuer chain and the host name are still
verified, so a certificate for another host or from an untrusted issuer is refused
(checked against `self-signed.badssl.com` and `untrusted-root.badssl.com`). Only the
download is affected; the log warns each time the setting is used. Once the CDN has a
valid certificate again, leave the setting off.

The release is under the "Public license with attribution and conditions
reserved". It reworks CC BY-SA 4.0 and is not a Creative Commons license. It has no
non-commercial clause; it requires attribution and an indication of changes, and
requires adapted material to be shared under the same license elements or a
BY-SA-compatible license. Please read the license text in the repository
before redistributing anything derived from it, and cite Kondratenko, Sokolov, Karpov,
Kutuzov, Savushkin and Minkin, "Large Raw Emotional Dataset with Aggregation
Mechanism" (arXiv:2212.12266).

## Annotation and label quality

Measured on the files used here (160,366 utterances; not
taken from the paper).

- The aggregated label is the annotators' perception, not the actor's intention.
  The intended emotion (`speaker_emo` in the markup) equals the aggregated label
  for 71.5% of the utterances. Intended `sad` ended up `neutral` 23,610 times and
  `sad` 20,635 times; intended `angry` ended up `neutral` 7,644 times and `angry`
  14,235 times; intended `positive` ended up `neutral` 6,436 times and `positive`
  11,746 times.
- Annotators passed a training and an exam (above 80%) and received control
  tasks with known answers (the paper). The markup has 2 to 149 annotations per
  utterance (median 4); on average 90.6% of an utterance's annotations agree with
  its aggregated label. Annotators heard the audio only.
- 1,517 train utterances are the control tasks themselves (`golden_emo` set, no
  text, no `source_id`); they carry many annotations each.
- The phrases are not independent of the emotions, but a phrase does not predict
  the label across speakers: for test utterances whose phrase also occurs in
  train, the train majority label is right 49.9% of the time (always `neutral`:
  59.3% on the same utterances).
- `source_id` is documented as the speaker (podcast) id; train has 1,870 values
  and test 190, with no overlap.
- Four groups of utterances with the same sound carry different labels (sad vs
  neutral) and are left out; see "Duplicates".

## Duplicates

Nine groups of recordings have the same sound (identical after decoding, or a
waveform correlation of 0.998 or more). Five groups are pairs with the same label;
with `exclude_duplicates: true` the later one of each pair (`duplicate_ids`) is
dropped, `false` (the default) keeps both. Four groups (9 utterances) carry
different labels (sad against neutral); none of them is used (`conflicting_ids`).

## Splits

The release's train/test partition is kept. The validation set is carved out of
train: whole `source_id` groups (the last column of the annotation files, documented in the
paper as the unique id of the speaker) are moved to valid until 10% of the train
utterances are reached (`valid_ratio`, `split_seed` in `dataset/config.yaml`), so valid shares no
`source_id` with train. Utterances without a `source_id` (1,517 of the 146,715
listed train utterances; none in test) are treated as groups of one. The listed
train utterances carry 1,870 distinct `source_id`s, test carries 190, and the two
sets share none.

| split | utterances | after `remove_long_short` | angry | happy | neutral | sad |
|---|---|---|---|---|---|---|
| train | 131997 | 131997 | 12790 | 12558 | 87218 | 19431 |
| valid | 14709 | 14709 | 1597 | 1619 | 9894 | 1599 |
| test | 13651 | (not filtered) | 1332 | 1215 | 8948 | 2156 |

`neutral` is 66% of train, so the class weights (written by `prepare_labels`)
range from 0.378 (neutral) to 2.63 (happy). Because the data are unbalanced, report
Macro-F1 rather than accuracy.

## Time needed

Measured on this machine (CPU only, `CUDA_VISIBLE_DEVICES=""`, 16 threads). The
download depends on the network.

| step | time |
|---|---|
| `prepare_source` (download of 30 GB, size check, extraction of 160,366 WAVs) | 23 min 43 s |
| `create_dataset` (checks the format of each file; nothing is converted) | 32.6 s |
| `remove_long_short` | 22.7 s |
| the whole `create_dataset remove_long_short prepare_labels` command | 1 min 3 s (includes starting `pixi` and Python) |

The download and the extraction were not timed separately in the recipe run. A
separate manual `curl` of `crowd.tar` took 24 min 42 s, and a plain `tar xf` of the
whole archive took 31 s. The extracted audio takes 23 GB and the manifests 50 MB.

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
