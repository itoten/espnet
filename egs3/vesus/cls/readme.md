# VESUS speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on VESUS alone and scores it on held-out actors, classifying each recording
into one of six instructed emotions.

[VESUS](https://www.nsa-lab.org/resources/vesus) (Varied Emotion in Syntactically
Uniform Speech) was recorded by the NSA lab at Johns Hopkins University: ten
English-speaking actors (five female, five male) read the same semantically
neutral script of words, phrases and sentences in each of six emotional styles,
neutral, monotone, happy, sad, angry and fearful. The release also holds crowd
ratings of the perceived emotion of most recordings (`Tools/VESUS_Key.csv`) and
evaluation protocols built on them (`Challenges/`); this recipe does not use them.

How the labels were made (paper, Sec. 2-3): the actors read the whole script once
in a neutral voice and then once in each of happiness, sadness, anger and fear, and
rated their own confidence in each portrayal; the folder is this instructed
emotion. Separately, ten Amazon Mechanical Turk workers per utterance chose the
emotion they perceived among the five (not monotone); individual raters agreed
with the instructed emotion 57% of the time, and the majority vote of the ten 65%
(paper, Table 3). The paper does not describe the monotone recordings; the
release's `README.txt` lists monotone as the second of six readings.

The label of a recording is its folder, the instructed emotion. All six are kept,
lowercased: `angry`, `fearful`, `happy`, `monotone`, `neutral`, `sad`.
`create_dataset` converts every recording to 16 kHz mono 16-bit PCM with ffmpeg,
reading `VESUS.zip` in place (nothing is extracted).

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `VESUS` — the directory holding `VESUS.zip` (or its unpacked `Audio/` folder)
- `VESUS_OUTPUT` — where the converted audio and manifests are written

## Download

VESUS cannot be downloaded by a script. Request it with the form linked from
[the VESUS page](https://www.nsa-lab.org/resources/vesus) (name, affiliation,
department, intended use, and agreement not to disseminate the database); a
download link for `VESUS.zip` is sent afterwards. Put the archive in `$VESUS`.
`prepare_source` only checks that it is there.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-25 against `VESUS.zip` as sent after the request form (4,244,472,686 bytes, md5 `a6eaeba1a2a8d62e36f8ca7d15d5c1bc`). All counts cover every recording in the release, before any filtering by this recipe.

### Access

Request form (name, affiliation, department, intended use, agreement not to disseminate the database independently); a download link is sent afterwards.

### Size of the release

| recordings | speakers | total hours |
|---|---|---|
| 15,041 | 10 | 7.21 |

How these were counted: every `.wav` under `Audio/` in the archive (macOS resource forks under `__MACOSX/` and `.DS_Store` files left out); hours from the WAV headers (0.61 to 5.46 s). All are 44.1 kHz mono. Speakers are the actor folders 1 to 10; `README.txt` describes actors 1 to 5 as female and 6 to 10 as male.

### Files per label

| emotion folder | recordings | hours |
|---|---|---|
| Angry | 2,519 | 1.282 |
| Fearful | 2,520 | 1.175 |
| Happy | 2,519 | 1.183 |
| Monotone | 2,446 | 1.116 |
| Neutral | 2,520 | 1.202 |
| Sad | 2,517 | 1.250 |

How these were counted: the folder of each recording. Hours per label are measured in the same way as the total hours above and summed over the recordings of that label.

### Issues in the release

- The script has 252 items, but not every actor has 252 recordings in every folder: actor 4 Monotone 179 (`README.txt` says 73 utterances are missing from that actor's monotone recording), actor 6 Sad 249, actor 1 Monotone, actor 6 Happy and actor 7 Angry 251 each.
- `Audio/4/Monotone/x.wav` is not numbered like the other files, so its script item is not known; the recipe keeps it (id `vesus-4-monotone-x`).
- `Audio/5/Fearful/152 .wav` has a space before the extension.
- No two recordings have byte-identical content.

### Not stated in the documentation

- `Tools/VESUS_Key.csv` holds 12,592 rows: the instructed emotion of each rated recording, the emotion chosen by each of 10 listeners (five classes, coded 0 neutral, 1 angry, 2 happy, 3 sad, 4 fearful), their confidence and response time, and the script item. Monotone recordings are not in it.
- The `Challenges/` protocols (`Leave one out`, `Longitudinal`, `Lexical Complexity`, each also in `5:10` and `7:10` variants) are NumPy files with train/validation/test lists of recordings and integer labels; in the first leave-one-out file, the labels of the listed recordings are not always the instructed emotion of their folder, so they appear to be listener-based labels.


## Splits

Held out by actor (`split_speakers` in `dataset/config.yaml`): actor 8 is the
test set, actor 9 the validation set, the other eight are the training set. The
actors were shuffled with seed 0 and the first two taken; both are male. The
release's `Challenges/` protocols cover only the listener-rated recordings (no
monotone) and appear to use listener-based labels (see "The data"), so they do not
fit this six-class recipe.

| split | recordings | angry | fearful | happy | monotone | neutral | sad |
|---|---|---|---|---|---|---|---|
| train | 12,017 | 2,015 | 2,016 | 2,015 | 1,942 | 2,016 | 2,013 |
| valid | 1,512 | 252 | 252 | 252 | 252 | 252 | 252 |
| test | 1,512 | 252 | 252 | 252 | 252 | 252 | 252 |

## License and citation

The VESUS page says the database is made freely available for academic use and
asks that any work using VESUS cite the paper below; the request form asks the
user to agree not to disseminate the database independently. The release itself
states no other terms. Please cite:

```
@inproceedings{sager2019vesus,
  title     = {{VESUS}: A Crowd-Annotated Database to Study Emotion Production and Perception in Spoken {English}},
  author    = {Sager, Jacob and Shankar, Ravi and Reinhold, Jacob and Venkataraman, Archana},
  booktitle = {Proc. Interspeech},
  pages     = {316--320},
  year      = {2019},
  doi       = {10.21437/Interspeech.2019-1413},
}
```

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
