# JNV speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on JNV alone and scores it on JNV's own test split, classifying each
utterance into one of six emotions.

[JNV](https://sites.google.com/site/shinnosuketakamichi/research-topics/jnv_corpus)
(Japanese Nonverbal Vocalization corpus) holds Japanese emotional nonverbal
vocalizations such as laughter, sobbing and screams, not words. 4 speakers (`F1`,
`F2`, `M1`, `M2`) each produced 105 vocalizations, 420 in total, for six emotions:
`angry`, `disgust`, `fear`, `happy`, `sad` and `surprise`. Measured class counts are
68 angry, 54 disgust, 50 fear, 76 happy, 40 sad and 132 surprise. Clips are 0.42 to
9.6 seconds long, 0.395 hours in total. Every speaker has 33 surprise, 19 happy, 17 angry and 10 sad clips;
disgust is 13 to 15 and fear 12 to 13 per speaker. The recordings are 48 kHz
stereo (one clip is 22.05 kHz mono), so `create_dataset` converts them to 16 kHz
mono with ffmpeg.

The label is the intended emotion, taken from the file name
`<speaker>_<emotion>_<id>_<session>.wav`; `dataset/config.yaml` lists the six
emotions. The session letter is `R` (phrase designated) or `F` (phrase chosen by the
speaker). The release also holds per-file authenticity and recognizability scores,
which this recipe does not use. All clips are used, including 42 that no listener
recognised as the intended emotion (forced-choice accuracy 0 in
`file_forced_choice_accuracy.csv`); the label stays the intended emotion.

**Read this before using the scores**: the vocalizations are written down as
phrases (for example `あは` for a laugh, `ぐすん` for sobbing, `こら` for anger), and the
corpus designates them per emotion: `phrases.txt` lists 87 phrases for the regular
(`R`) session, with 16 for happy, 30 for surprise, 14 for angry, 10 for fear, 10 for
disgust and 7 for sad. Only 9 of the phrase strings (for example `いや`, `えー`,
`ああ`) occur under two emotions. The phrase, the sound itself, therefore points to
the emotion, and the number of clips per class follows the number of phrases (in
the `R` session 120 surprise against 28 sad). For the 73 `F` clips the speaker chose
the phrase and the phrases are not recorded, so whether they lean towards
particular emotions is not known. Treat a high score as partly a result of this
link between phrase and emotion, not only of prosody.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `JNV` — the directory holding the extracted `JNV/` directory, or where to download it to
- `JNV_OUTPUT` — where the converted audio and manifests are written

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the JNV corpus (Japanese Nonverbal Vocalization corpus; folders `F1`, `F2`, `M1`, `M2`, `phrases.txt`, `readme.md` and three rating files, downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from the corpus page, no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 420 | 4 | 0.395 |

How these were counted: every `.wav` under the four speaker folders (48 kHz stereo 24-bit in 419 files, 22.05 kHz mono in 1; 0.42 to 9.62 s); hours from the file lengths. Speakers: `F1`, `F2`, `M1`, `M2` (two women, two men; `M2` is a professional, the others amateurs, per the release's readme).

### Files per label

| label (emotion in the file name) | samples |
|---|---|
| angry | 68 |
| disgust | 54 |
| fear | 50 |
| happy | 76 |
| sad | 40 |
| surprise | 132 |

How these were counted: the emotion field of each file name `<speaker>_<emotion>_<id>_<session>.wav`. The session letter is `R` (phrase designated) or `F` (phrase chosen by the speaker). Every speaker has 33 surprise, 19 happy, 17 angry and 10 sad clips; disgust 13 to 15 and fear 12 to 13.

### Issues in the release

- The release's readme states 105 clips per speaker (87 `R`, 15 `F`) and a total length of 406.9 s; the files give 104 to 106 clips per speaker (`R` 86 to 87, `F` 17 to 20) and 1,422.0 s. `file_forced_choice_accuracy.csv` has 418 rows (2 clips have no row).
- The vocalizations are written down as phrases (for example `あは` for a laugh, `ぐすん` for sobbing); `phrases.txt` lists 87 phrases for the `R` session: 16 for happy, 30 for surprise, 14 for angry, 10 for fear, 10 for disgust and 7 for sad. Only 9 of the phrase strings occur under two emotions. For the 73 `F` clips the phrases are not recorded.

### Listening test ratings in the release

The release contains `file_rating_scale.csv`, `file_authenticity.csv` and `file_forced_choice_accuracy.csv`, the listeners' per-clip ratings (crowdsourced). In `file_forced_choice_accuracy.csv` the mean forced-choice accuracy is 0.734; 42 clips have accuracy 0 (no listener chose the intended emotion); the emotion with the highest rating equals the intended emotion for 284 of the 420 clips.


## Download

`prepare_source` downloads `jnv_corpus_ver3.zip` (326 MB, byte size checked; the
site publishes no checksum) from the author's site and extracts the wav files.
Measured on one run: about 66 seconds to download; `create_dataset`,
`remove_long_short` and `prepare_labels` together took about 44 seconds.

## Splits

The release publishes no partition. Utterances are split 70/10/20 at random
within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed` in
`dataset/config.yaml`). Speakers are not held out, so all four speakers appear in
every split and scores are optimistic for unseen speakers.

| split | utterances | after `remove_long_short` |
|---|---|---|
| train | 294 | 294 |
| valid | 42 | 42 |
| test | 84 | 84 (not filtered) |

`remove_long_short` drops nothing.

## License and citation

CC BY-SA 4.0 (stated in the corpus readme and on the corpus page). Cite: Detai Xin,
Shinnosuke Takamichi and Hiroshi Saruwatari, "JNV corpus: A corpus of Japanese
nonverbal vocalizations with diverse phrases and emotions", Speech Communication
156 (2024) 103004.

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
