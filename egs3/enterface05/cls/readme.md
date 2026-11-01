# enterface05 speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on enterface05 alone, and scores it on enterface05's own test split
(6 classes: `angry`/`disgust`/`fear`/`happy`/`sad`/`surprise`).

**Read this before using the scores**: subjects did not read a fixed script,
so the words alone give the emotion away; a text-aware frontend can read the
label off the content without modelling prosody. Do not compare this
recipe's numbers with a corpus whose script is fixed across emotions without
saying so.

[eNTERFACE'05](https://enterface.net/enterface05/emotion.html) is an
audio-visual database: the recordings are AVI files carrying 48 kHz stereo
PCM, so `create_dataset` extracts the audio and resamples it to 16 kHz mono.
The site's certificate chain is incomplete, so scripts cannot download it;
fetch `project2_database.zip` by hand and place it under `$ENTERFACE05` (or
let `$ENTERFACE05` hold the already-unpacked tree) for `prepare_source` to
extract. Labels and speakers come from directory names, not file names,
which carry typos.

The release holds 1,293 AVI files in 44 subject directories (the corpus page
reports 42 subjects). 42 subjects have the expected 30 recordings (5 sentences in
each of 6 emotions). Subject 23 lacks 3 (happiness, sentences 1 to 3) and its 27 files
are used. Subject 6 is not segmented: it has one file per emotion, 56 to 107
seconds each, against 1.1 to 5.4 seconds for the others. A Whisper transcript
(not checked by hand) shows that each file is a whole session, not one sentence:
the subject repeats the acted lines several times (for example "That's horrible! I
will never eat noodles again!" for disgust), and the experimenter's voice is mixed
in, with instructions such as "from the beginning" or "you are not looking at the
camera" and retakes. The files could be cut into sentences, but that is
laborious, so all 6 are left out. 1,287 recordings from 43 subjects are used.

The corpus publishes no partition. Speaker ids are sorted and dealt out to
train/valid/test (33, 5 and 5 speakers).

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the eNTERFACE'05 audio-visual emotion database (`project2_database.zip` from enterface.net, fetched by hand because the site's certificate chain is incomplete; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Manual download from the project site (scripts cannot fetch it: the certificate chain is incomplete), no sign-in.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 1,293 | 44 (subject folders) | 1.14 |

How these were counted: every `.avi` under the `subject N/` folders (48 kHz stereo PCM audio; 1.12 to 106.92 s); hours from the container durations. There are 44 subject folders; the corpus page reports 42 subjects. The release has no description file, only the AVI files and `Thumbs.db` files.

### Files per label

| label (emotion folder in the release) | samples |
|---|---|
| anger | 216 |
| disgust | 216 |
| fear | 216 |
| happiness | 213 |
| sadness | 216 |
| surprise | 216 |

How these were counted: the emotion folder of each `.avi` file (`subject N/<emotion>/sentence k/`).

### Issues in the release

- Subject 6 is not segmented: one file per emotion (6 files), 56 to 107 s each, against 1.1 to 5.4 s for the other subjects.
- Subject 23 lacks 3 recordings (happiness, sentences 1 to 3). 42 subjects have the expected 30 recordings (5 sentences x 6 emotions), 1,260 files in all.
- File names carry typos; labels and speakers must be taken from the directory names.
- Whisper transcripts (not checked by hand) of the subject 6 files show whole sessions: the acted lines repeated several times, the experimenter's voice with instructions such as "from the beginning" and "you are not looking at the camera", and retakes.

### Not stated in the documentation

- The subjects did not read a fixed script, so the spoken words differ between recordings of the same emotion.


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

`ENTERFACE05_OUTPUT` must already point at the prepared data; `create_dataset`
writes it there if it does not.
