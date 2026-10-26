# EmoSpeech (keyword) speech emotion recognition (single-corpus baseline)

Trains the `cls` system (fine-tuned WavLM Large + attentive statistics pooling + FC
head) on EmoSpeech (keyword) alone and scores it on its own test split, classifying
each utterance into one of four emotions.

The Indian EmoSpeech Command dataset ([Banga et al., 2019](https://arxiv.org/abs/1910.13801))
holds short spoken commands (`help`, `bacho`, `stop`, `no`, `go`, `yes`, plus
`unknown` and `background` clips) recorded with an emotion and an environment tag.
The Kaggle archive has 7,735 wav files in folders named after the recording place.
The label is the emotion in the file name
`<keyword>-<environment>-<emotion>-<stamp>-<user hash>.wav`.

Clips left out (counts from the build log): 1,117 clips are dropped by label or
keyword, and 88 more are dropped as duplicates (identical md5) of a clip that is kept,
so 6,530 utterances remain, all 16 kHz mono and 1.96 to 3.75 seconds long, 5.54 hours
in total. The four labels are `angry`, `calm`, `fear` (`Fearful` in the file names)
and `happy`.

- **`background` clips are left out (1,116 files).** The paper and README describe
  `background` as the absence of a keyword: sounds that are not words, such as
  birds, road noise or white noise. Although these files carry emotion tags
  (calm 880, fearful 53, happy 19, angry 1 and `None` 163), listening to samples
  (2026-10-24) found only background sound and no speech, so the tags describe no
  speech emotion. The `background` files also account for 163 of the 164 files
  tagged `None` (the other one is an `unknown` clip); all files tagged `None` are
  dropped. The paper does not explain what `None` means; it is not the `calm` class.
- **`unknown` clips are kept (1,191 files).** They are spoken words that are not
  among the six keywords (for example other words and Hindi sentences, from a
  Whisper transcription of 25 clips).
- The remaining clips are the six keywords (`help`, `bacho`, `stop`, `no`, `go`,
  `yes`) and `unknown`, spoken in `calm`, `happy`, `angry` and `fearful` styles, each
  also tagged with an environment (`safe`, `scream`, `glass break`, `gunshots`).

Environment variables (both optional; defaults are `download/` and `data/` under the
recipe):

- `EMOSPEECH_KEYWORDS` — the directory holding the extracted `emospeech-dataset-v1/`
- `EMOSPEECH_KEYWORDS_OUTPUT` — where the copied audio and manifests are written

## Download

The Kaggle dataset ([ujjwal9/emospeech](https://www.kaggle.com/datasets/ujjwal9/emospeech))
needs a Kaggle login, so the recipe does not download it. Download the zip by hand,
extract it and point `$EMOSPEECH_KEYWORDS` at the folder that holds
`emospeech-dataset-v1/`; otherwise `create_dataset` stops with the URL. On one run
`create_dataset`, which copies the audio, took 342 seconds. The download time was
not measured.

## Splits

The archive publishes no partition. 5,709 of the 7,735 recordings carry the single
"unknown user" hash, so speakers cannot be held out. Utterances are split 70/10/20 at
random within each label (`train_ratio`, `valid_ratio`, `test_ratio`, `split_seed`
in `dataset/config.yaml`), so the splits share speakers and scores are optimistic.

| split | utterances | hours | after `remove_long_short` | angry | calm | fear | happy |
|---|---|---|---|---|---|---|---|
| train | 4,570 | 3.88 | 4,570 | 777 | 2,039 | 1,084 | 670 |
| valid | 653 | 0.55 | 653 | 111 | 291 | 155 | 96 |
| test | 1,307 | 1.11 | 1,307 (not filtered) | 222 | 583 | 310 | 192 |

## License and citation

The archive is distributed under CC BY-NC-SA 4.0 (`License.txt` in the archive), so
use is non-commercial and derivatives must be shared alike. Please cite

```
@misc{1910.13801,
  Author = {Subham Banga and Ujjwal Upadhyay and Piyush Agarwal and Aniket Sharma and Prerana Mukherjee},
  Title = {Indian EmoSpeech Command Dataset: A dataset for emotion based speech recognition in the wild},
  Year = {2019},
  Eprint = {arXiv:1910.13801},
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

**Duplicates.** `exclude_duplicates` in `dataset/config.yaml` (default `true`) keeps one recording per group of byte-identical files with one label (88 extra copies are dropped); `false` keeps every copy. Copies with conflicting labels are always dropped.
