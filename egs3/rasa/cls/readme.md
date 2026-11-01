# Rasa speaking-style recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on Rasa alone and scores it on Rasa's own test split, classifying each
utterance into one of 17 speaking styles: the six Ekman emotions plus eleven
non-emotional styles (reading of Wikipedia, news and books, conversation, voice
commands, proper nouns, and so on).

[Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) (AI4Bharat) is an expressive
text-to-speech corpus for the 22 scheduled languages of India. Each language has one
female and one male professional voice artist (44 speaker-language pairs), who read
text of many kinds in a studio. The audio is 48 kHz mono 16-bit; `create_dataset`
converts it to 16 kHz. The recipe uses all 22 languages and all 17 styles.

## The data

> These counts and observations were made by the recipe author while building this recipe,
> as a practical reference for users. They are not peer-reviewed measurements and should not be
> cited as research results; cite the corpus documentation instead.

Checked on 2026-10-29 against the Hugging Face dataset `ai4bharat/Rasa` (1,317 parquet files,
387.65 GB, downloaded 2026-10-28; repository revision
`632f55c7ac590219d41cd7adffce5b440e4604f5`, last modified 2026-06-06).
All counts cover every sample in the release, before any filtering by this recipe.

### Access

Hugging Face, gated: after agreeing on the dataset page to share the username and email with
the authors, all files downloaded with a token. No other form or review.

### Size of the release

| samples | speakers | total hours |
|---|---|---|
| 640,946 | 44 | 1,145.40 |

How these were counted: every row of the 1,317 parquet files (one row = one utterance); hours
from the decoded audio length. Speakers: distinct pairs of the `language` and `gender` columns
(one female and one male voice per language). The release has no speaker ID, so whether an
artist recorded more than one language cannot be told; 44 is the number of voices per
language and gender.

### Files per label

| label (as in the release) | samples |
|---|---|
| WIKI | 144,250 |
| CONV | 124,578 |
| PROPER NOUN | 104,068 |
| BOOK | 55,788 |
| NEWS | 41,782 |
| HAPPY | 21,975 |
| ANGER | 21,467 |
| FEAR | 20,712 |
| SURPRISE | 20,484 |
| DISGUST | 20,134 |
| SAD | 20,070 |
| ALEXA | 13,292 |
| INDIC | 12,852 |
| BB | 6,769 |
| DIGI | 5,452 |
| UMANG | 5,436 |
| SANGRAH | 1,837 |

How these were counted: the `style` column of every row. The 70 duplicated filenames below
are counted under both of their styles.

### Issues in the release

- Only 6 of the 17 styles are emotions (`HAPPY`, `ANGER`, `FEAR`, `SURPRISE`, `DISGUST`,
  `SAD`: 124,842 samples, 280.4 hours). The other 11 are kinds of text or voice-command
  domains and make up 516,104 of the 640,946 samples (80.5%) and 865.0 of the 1,145.4 hours.
- 640,946 rows against 640,950 utterances in the dataset card's statistics table: the
  Gujarati female and male voices have 2 fewer each.
- 70 filenames of the Tamil male voice (`TAM_M_NAMES_*`) occur twice, with byte-identical
  audio (MD5) but a different text and style (`DIGI` and `PROPER NOUN`).
- Observation by the recipe author (2026-10-29): of those 70, one clip
  (`TAM_M_NAMES_00106`) was listened to. It says "Digital India Mission", the text of its
  `PROPER NOUN` entry, not "nine hundred" of its `DIGI` entry. The other 69 were not
  listened to.
- 2,529 pairs of byte-identical recordings (MD5), all within one speaker and style.
  1,644 pairs have consecutive filenames; 464 pairs have one file in the published train
  and the other in test.
- Observation by the recipe author (2026-10-29): two of those pairs were listened to
  (`SND_F_WIKI_01563`/`01564` and `SND_F_WIKI_03613`/`03612`). The second pair sounded the
  same; the first sounded slightly shifted, although the two files are byte-identical. The
  cause was not checked; the copies that were played are also byte-identical.
- One file is empty (`ASM_F_INDIC_00001`). Durations range from 0 s (that file) to
  332.6 s.
- Observation by the recipe author (2026-10-29): one clip each of `ANGER` (Konkani),
  `SAD` (Bodo), `WIKI` (Malayalam) and `NEWS` (Assamese) was listened to. The `ANGER` and
  `SAD` clips sounded different in vocal expression from the `WIKI` and `NEWS` clips.

### Not stated in the documentation

- `SANGRAH` occurs only for the Bengali male voice, and `INDIC` is absent from 10 of the
  22 languages (`style`, `language` and `gender` columns).

## Labels

The label is the `style` column of the dataset: the kind of text the artist was asked to
read. It names the text source or the way of speaking, not a perceived emotion. Lower
case, spaces replaced by `_`.

- Neutral reading from text sources: `wiki`, `indic`, `sangrah`
- Other text types: `conv` (conversation), `news`, `book`, `proper_noun` (very short
  clips, median 1.8 s)
- Voice-command domains: `alexa`, `bb`, `umang`, `digi`
- The six Ekman emotions: `happy`, `anger`, `fear`, `surprise`, `disgust`, `sad`

For the emotion styles the artist read sentences written to express that emotion. The
paper (Varadhan et al., Interspeech 2024, covering Assamese, Bengali and Tamil) describes
the sentences as generated by a large language model and kept when human annotators
certified that the English text conveys the intended emotion; translators were asked to
keep the emotion over the literal meaning. The label is therefore the intended
emotion of the text the artist performed; no listener label of the recordings exists.

The recipe uses all 17 styles.

Label counts after the recipe's filtering (see Splits):

| label | train | valid | test |
|---|---|---|---|
| wiki | 116,666 | 12,963 | 14,409 |
| conv | 100,906 | 11,211 | 12,460 |
| proper_noun | 84,228 | 9,359 | 10,400 |
| book | 45,072 | 5,011 | 5,559 |
| news | 33,758 | 3,751 | 4,170 |
| happy | 17,800 | 1,977 | 2,198 |
| anger | 17,387 | 1,933 | 2,147 |
| fear | 16,778 | 1,861 | 2,073 |
| surprise | 16,590 | 1,844 | 2,050 |
| disgust | 16,308 | 1,813 | 2,013 |
| sad | 16,258 | 1,806 | 2,006 |
| alexa | 10,763 | 1,197 | 1,330 |
| indic | 10,393 | 1,157 | 1,286 |
| bb | 5,484 | 609 | 676 |
| umang | 4,379 | 489 | 542 |
| digi | 4,290 | 475 | 529 |
| sangrah | 1,484 | 164 | 183 |

## Listening study

The paper reports no perception test of the recordings. Its two listening experiments
use speech synthesized by models trained on the corpus: a MUSHRA expressiveness test with
22 raters, and a test in which listeners classify the emotion of the synthesized speech.
The human checks reported for the corpus itself are the certification of the English
emotion sentences described above and the authors' quality control (paper Section 2.3):
2,075 script sentences that did not convey the intended emotion were rejected, and
recordings with irregular pauses, stuttering or mispronunciations not aligned with the
intended emotion were re-recorded or removed. No counts or agreement figures are given
for this. The authors also trimmed silence and normalised the amplitude by hand.

## Recording

The paper (for the three languages it describes) states that the recordings were made
in a studio by professional voice artists who agreed in writing to research and
commercial release.

## Environment variables

Both are optional.

- `RASA` - where the Hugging Face files are placed while downloading (default `download/`
  under the recipe)
- `RASA_OUTPUT` - where the converted audio, progress files and manifests are written
  (default `data/`); about 125 GB

## Download

The Hugging Face dataset `ai4bharat/Rasa` is gated. **Before the first run, sign in to
Hugging Face, open the dataset page and agree to share your username and email with the
authors**, and make your token available (`huggingface-cli login` or `$HF_TOKEN`).
Without this, `prepare_source` stops with an error that says so.

`prepare_source` goes through the 22 languages and the 1,317 parquet files (388 GB in
total): 12 files in parallel are downloaded, converted to 16 kHz wav and deleted, so the
raw data is never on disk at once. A progress file per parquet file lets an interrupted
run continue. Measured: download and conversion took about 5 hours in total;
`collect_stats` took about 15 minutes.

## Splits

The published train/test partition is used (576,854 / 64,092 utterances before
filtering). Valid is a random 10% of train, drawn with `split_seed` for each combination
of language, speaker gender and style. All speakers appear in train, valid and test.

`remove_long_short` drops utterances shorter than 0.2 s or longer than 30 s from all
three splits: 58 test clips are longer than 30 s (up to 332.6 s) and do not fit in GPU
memory at inference. Both entries of each of the 70 duplicated filenames are always left out,
since it cannot be told which text is right. The 2,529 byte-identical pairs are kept by
default; `exclude_duplicates: true` in `dataset/config.yaml` leaves out one copy of each
(the one in train when the pair spans train and test). The result is 518,544 / 57,620 /
64,031 utterances.

## License

CC BY 4.0, as stated on the Hugging Face dataset card. Cite:

```
@inproceedings{ai4bharat2024rasa,
  author={Praveen Srinivasa Varadhan and Ashwin Sankar and Giri Raju and Mitesh M. Khapra},
  title={{Rasa: Building Expressive Speech Synthesis Systems for Indian Languages in Low-resource Settings}},
  year=2024,
  booktitle={Proc. INTERSPEECH 2024},
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
