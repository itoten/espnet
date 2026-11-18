# Dialogs speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Dialogs alone and scores it on Dialogs' own test split, classifying each
utterance into one of seven emotions.

[Dialogs](https://huggingface.co/datasets/langswap/dialogs-ru-emotional-conversations)
is a studio-quality corpus of expressive, conversational Russian speech, recorded
by three professional theatre actors (two female, one male) performing scripted
dialogs face to face. It was built for dialog-oriented and emotional
text-to-speech. The dataset card states 11,796 utterances (20.6 hours, 44.1 kHz,
16-bit) and 12 "style/emotion" labels per utterance, in the `emotion` column of
the pipe-delimited split files `train.csv`, `val.csv` and `test.csv`.

What the recipe uses and leaves out:

| label (`emotion` column) | used | reason |
|---|---|---|
| `neutral`, `happy`, `surprise`, `sad`, `angry`, `disgust`, `fear` | yes | the seven emotion classes |
| `sgorogovrki` (the card calls it `tongue-twister`) | no | a reading task, not an emotion |
| `poem` | no | a reading style, not an emotion |
| `whisper` | no | a phonation mode, not an emotion |
| `laughing` | no | a vocal event; the emotion behind it is not labelled |
| `arrogance` | no | a judgement call: an attitude rather than a basic emotion, with no counterpart among the seven classes used by the other recipes |

The tongue-twister label is spelled `sgorogovrki` (a transliteration of the
Russian word) in the split files, not `tongue-twister` as on the card; both
spellings are listed in `drop_emotions` in `dataset/config.yaml`. The log
reports the number of utterances dropped for each label. An unknown label stops
the build with an error.

`create_dataset` converts the kept recordings to 16 kHz mono with ffmpeg.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `DIALOGS_RU` — the directory holding `train.csv`, `val.csv`, `test.csv` and `wavs/`, or where to download them to
- `DIALOGS_RU_OUTPUT` — where the converted audio and manifests are written

## Download

`prepare_source` downloads `wavs/*.wav`, the three split files, `README.md` and
`LICENSE.md` from Hugging Face at a pinned commit
(`a74ff44d8d27ca3d21098ea3ed4d31a32923fbdd`) with `huggingface_hub.snapshot_download`;
no login is needed. The audio takes several GB. `metadata.csv` (all splits in one
file) and `preview.parquet` (60 sample clips) are not needed and not downloaded.
Files already present are not fetched again, so an interrupted download can be
resumed by running `prepare_source` again.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-11-18 against the Hugging Face dataset `langswap/dialogs-ru-emotional-conversations` (repository revision `a74ff44d8d27ca3d21098ea3ed4d31a32923fbdd`, last modified 2026-10-12). All counts cover every file in the release, before any filtering by this recipe.

### Access

Hugging Face, direct download, no sign-in.

### Size of the release

| utterances listed | utterances with audio | speakers | hours of audio |
|---|---|---|---|
| 11,796 | 9,996 | 3 | 17.42 |

How these were counted: rows of `train.csv` (11,428), `val.csv` (180) and `test.csv` (188); files under `wavs/` (every one is listed in a split file); hours from the WAV headers (0.41 to 36.73 s). All are 44.1 kHz mono. Speakers are the `speaker_id` values `M`, `S` and `D`.

### Files per label

| `emotion` column | files with audio | hours |
|---|---|---|
| neutral | 5,264 | 9.023 |
| happy | 2,603 | 4.972 |
| surprise | 716 | 1.121 |
| sad | 389 | 0.813 |
| angry | 181 | 0.266 |
| disgust | 152 | 0.247 |
| fear | 105 | 0.130 |
| poem | 187 | 0.131 |
| laughing | 151 | 0.261 |
| arrogance | 115 | 0.187 |
| sgorogovrki | 81 | 0.177 |
| whisper | 52 | 0.091 |

How these were counted: the `emotion` column of the split files, for the files present under `wavs/`. Hours per label are measured in the same way as the total hours above and summed over the files of that label. The last five labels are speaking styles and are not used by this recipe.

### Issues in the release

- The audio of 1,800 listed utterances is missing: 1,742 of `train.csv` (1,664 with an emotion label, mostly neutral 961 and happy 440), 32 of `val.csv` and 26 of `test.csv`. `wavs/` holds 9,996 files. The split files' `duration` column puts the missing audio at 3.2 hours. The recipe skips these utterances.
- The dataset card calls one label `tongue-twister`; the split files spell it `sgorogovrki`.
- `masha_sveta_part10_1765` and `masha_sveta_part10_1821` (both neutral, same transcript, both in train) have byte-identical content; the second is listed in `duplicate_ids`.


## Splits

The published split is used as it is, with `val` as the validation set. The
dataset card gives 11,428 / 180 / 188 utterances for train / validation / test,
before this recipe drops the speaking-style labels, and says that validation
and test were drawn stratified, five utterances per speaker per label. All
three speakers appear in every split, so the split is not speaker-independent,
and validation and test are small. After the speaking styles and the
utterances without audio in the release (see "The data") are left out:

| split | listed in the split file | used by this recipe |
|---|---|---|
| train | 11,428 | 9,185 |
| valid | 180 | 106 |
| test | 188 | 119 | `remove_long_short` drops train and valid
clips outside 0.2 to 30 s; test stays unfiltered.

## License and citation

OpenRAIL (see `LICENSE.md` of the dataset): use, modification, redistribution and
commercial use are allowed, subject to the use restrictions listed in the license.
Please cite:

```
@inproceedings{shigabeev2026dialogs,
  title     = {Dialogs: A Studio-Quality Expressive Conversational Russian Speech Corpus for Dialog Assistants},
  author    = {Shigabeev, Ilya and Latyshev, Ilya},
  booktitle = {Proc. Interspeech},
  url       = {https://arxiv.org/abs/2607.14310},
  year      = {2026}
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
