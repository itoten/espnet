# emozionalmente speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling +
FC head) on emozionalmente alone, and scores it on emozionalmente's own test
split (7 classes: `angry`/`disgust`/`fear`/`happy`/`neutral`/`sad`/
`surprise`, spelled to match the other corpora here; the corpus's own
spelling is `anger`/`joy`/`neutrality`/`sadness`).

[Emozionalmente](https://doi.org/10.5281/zenodo.12616095) is 6,902 Italian
utterances crowdsourced from 431 non-professional actors, already 16 kHz
mono WAV. `EMOZIONALMENTE` points at an existing corpus tree, or where to
download it to; the Zenodo archive is checked against its published MD5.

The corpus publishes a speaker-independent train/dev/test split, stratified
by emotion, gender and age, and this recipe reads it from
`metadata/split/*.csv` rather than rebuilding it.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-01 against the Zenodo record 12616095 (Emozionalmente: `audio/`, `metadata/samples.csv`, `evaluations.csv`, `users.csv` and `split/`; downloaded around 2026-10-15). All counts cover every sample in the release, before any filtering by this recipe.

### Access

Direct download from Zenodo, no sign-in (the archive is checked against its published MD5).

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 6,902 | 431 | 6.30 |

How these were counted: every row of `metadata/samples.csv` (equal to the `.wav` files in `audio/`; 16 kHz mono 16-bit, 0.76 to 13.87 s); hours from the file lengths. Speakers are the 431 distinct `actor` values (non-professional actors; demographics in `users.csv`).

### Files per label

| label (the `emotion_expressed` column) | samples |
|---|---|
| anger | 986 |
| disgust | 986 |
| fear | 986 |
| joy | 986 |
| neutrality | 986 |
| sadness | 986 |
| surprise | 986 |

How these were counted: the `emotion_expressed` column of `samples.csv`: the emotion the actor was asked to express (18 sentences per actor in seven intentions).

### Issues in the release

- The recordings were mostly made with non-professional equipment by the actors themselves (stated in the documentation).

### Listening test ratings in the release

`metadata/evaluations.csv` holds the 34,510 evaluations (five per clip) by 829 evaluators: the recognised emotion (`emotion_recognized`) and an audio-quality flag. Counted on the file: a single evaluation equals the expressed emotion in 66.4% of the evaluations, and the strict majority of the five equals it for 5,025 of the 6,902 clips.


## Data and ratings

The 6,902 clips (0.76 to 13.9 seconds, 3.81 seconds on average, 16 kHz mono) were acted
by 431 amateur actors, each saying 18 sentences with the seven emotional intentions,
and were mostly recorded with non-professional equipment. The label is the intention
of the actor, not a vote. Besides the audio, the corpus holds the demographics of the
actors and of the 829 evaluators, and the 34,510 evaluations (five per clip, with an
audio-quality flag). A listener's single answer equals the intention for 66.4% of the
judgements and the strict majority of the five equals it for 5,025 of 6,902 clips; the
ratings are not used here.

## License and citation

The Zenodo record
[10.5281/zenodo.12616095](https://doi.org/10.5281/zenodo.12616095) is licensed CC BY 4.0.
Cite F. Catania, J. W. Wilke and F. Garzotto, "Emozionalmente: A Crowdsourced Corpus of
Simulated Emotional Speech in Italian", IEEE Transactions on Audio, Speech and Language
Processing 33, pp. 1142-1155, 2025.

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

`EMOZIONALMENTE_OUTPUT` must already point at the prepared data;
`create_dataset` writes it there if it does not.
