# Multi-dataset speech emotion recognition (multidataset_ser_03)

Experiment `multidataset_ser_03` studies how to train speech emotion
recognition models that hold up on corpora never seen in training, in
particular models trained from scratch. It uses the corpora and labels of
`multidataset_ser_02`, with three changes:

- **Validation on held-out corpora.** Five training corpora are held out in
  full as the validation set, instead of 5 % of every training corpus, so
  that validation measures the same thing as evaluation: performance on
  corpora the model has never seen.
- **Log-mel models normalised as in ESPnet2's ASR recipes.** The log-mel
  features are normalised with the mean and variance of the training set
  (`GlobalMVN`), after SpecAugment. Per-utterance statistics, used before,
  include the frames SpecAugment zeroes, so that the unmasked frames are
  shifted in training and not at inference.
- **Noise from FSD50K.** Additive noise comes from the FSD50K clips under
  CC0 or CC BY, without voice, crowds, breathing or music, so that the
  license of every noise file is known.

One `cls` system (encoder, attentive statistics pooling and a fully
connected head) is trained on the corpora whose license allows commercial
use and scored on all the others:

| Model | Encoder | Input | Initial weights |
|---|---|---|---|
| `hubert_large` | HuBERT Large (24 layers, width 1024), convolutional feature encoder frozen | waveform | pre-trained (`facebook/hubert-large-ll60k`) |
| `conformer` | Conformer, 12 blocks of width 1024 (about 297M parameters, the size of HuBERT Large) | 80-dim log-mel | random |
| `conformer_m` | Conformer, 12 blocks of width 512 (the encoder of ESPnet2's LibriSpeech recipes) | 80-dim log-mel | random |
| `conformer_s` | Conformer, 12 blocks of width 256 (the encoder of ESPnet2's LibriSpeech-100 recipe) | 80-dim log-mel | random |
| `ebranchformer` | E-Branchformer, 10 blocks of width 1024 (about 312M parameters) | 80-dim log-mel | random |

It uses the 72 corpora that the dataset filter of ser_dataset_db keeps for
`multidataset_ser_01` (`local/dataset_filtering/output/multidataset_ser_01/`,
generated 2026-11-09 from DB commit 3ff353c). They are listed in
`dataset/config.yaml`; one of them is on hold (see below), so 71 are
prepared. Every corpus is downloaded and converted by its own
`egs3/<corpus>/cls` recipe; corpora already prepared for another recipe are
reused as they are.

## Training and evaluation data

| Role | Corpora | Rows |
|---|---|---:|
| Training | 40 corpora whose license allows commercial use with no share-alike condition | 266,701 |
| Validation | 5 more such corpora, held out in full (see below) | 13,519 |
| Evaluation, one test set per corpus | the 26 others: aesdd, brytserc, cafe, cbesd, cremad, dailytalk, dusha, emospeech_keywords, emotiontalk, emovdb, jnv, jvnv, legov2, mcaespps, meld, multidialog, nemo, nonverbaltts, pavoque, ravdess, shemo, tess, thaiser, turevdb, utemo, vivae | about 483,000 |

Rows are those with one of the seven labels below. eNTERFACE'05 (MIT) is left
out while its site is down.

- **Roles.** Training uses the corpora whose license allows commercial use and
  puts no condition on what is made from them (CC0, CC BY, MIT, Apache,
  ODC-By), so that the trained model can be released under any license.
  Every other corpus is used for evaluation only: non-commercial and
  research-only licenses, and those with a share-alike condition, which are
  treated the same way (CC BY-SA: cbesd, dailytalk, jnv, jvnv, thaiser;
  Dusha's own license, a modified CC BY-SA; GPL-3.0: meld; ODbL: cremad).
  The licenses are those recorded in ser_dataset_db (discussion 0003). The
  lists are `train_corpora` and `eval_corpora` in `dataset/config.yaml`;
  every corpus must have exactly one role.
- **Every split of a corpus is used.** Each corpus recipe's `train`, `valid`
  and `test` manifests are read unfiltered and pooled.
- **Duration.** `remove_long_short` keeps the utterances of 0.5 to 30 s, in
  training, validation and every evaluation set alike
  (`manifest_filtered/`). Some evaluation corpora hold clips of a few
  milliseconds, too short for the encoders, and others of several minutes.
- **Labels.** Only seven classes are used, in training and evaluation alike:
  neutral, happy, sad, angry, fear, surprise and disgust (`labels`). A row
  with any other label (calm, boredom, RASA's text styles, ...) is left out of
  every manifest; the build log counts them per corpus and label. An
  evaluation corpus that lacks some of the seven is scored over the ones it
  has. `label_map` renames labels that plainly mean one of the seven: RASA's
  `anger` becomes `angry`, EmoGator's `surprise_negative` and
  `surprise_positive` become `surprise`, LEGOv2's `slightly_angry` and
  `very_angry` become `angry`, EARS's `sadness` becomes `sad` (this is the only change in the data from
  `multidataset_ser_01`).
- **Validation.** Taken out of the training corpora as `validation.method`
  says:

  | `method` | Validation rows |
  |---|---|
  | `held_out_corpora` (used here) | every row of the corpora listed in `corpora`: berst (English), csemotions (Chinese), eseravd (Indonesian), thorsten (German) and uftbesd (Bengali), 13,519 rows before the duration filter. They are not trained on, so 40 corpora remain for training |
  | `random_per_corpus` | `ratio` (0.05) of each training corpus's rows, chosen uniformly at random with `seed` (0), regardless of speakers; at least one row per corpus (`multidataset_ser_02`) |
  | `corpus_valid_split` | each corpus recipe's own `valid` split (how each recipe splits differs: by speaker, at random, ...) |

  The five corpora were chosen from the 45 training corpora by a fixed rule
  without random numbers: close to 5 % of the rows, labels as balanced as
  possible, spread over several corpora. No evaluation corpus is used for
  validation. They are all acted speech, while the evaluation corpora also
  hold conversations and recordings in the wild, so validation is a rough
  guide to evaluation, not a copy of it. Another method is a function in
  `dataset/builder.py` registered in `VALIDATION_METHODS`, with its own block
  under `validation`.

`create_dataset` writes `manifest/train.tsv`, `manifest/valid.tsv` and
`manifest/<eval corpus>.tsv` under `$MULTIDATASET_SER_03_OUTPUT`, and the
settings they were built with to `manifest/build_settings.json`. If the
roles, labels or validation settings change, the next `create_dataset`
rewrites the manifests.

## Model and training

Shared (`conf/training.yaml`): attentive statistics pooling, one hidden
layer of 1024 units (dropout 0.5) and seven outputs; cross-entropy weighted
by inverse class frequency; AdamW (weight decay 0.01) with TristageLR
(warm-up over the first 10 % of the updates, hold for 40 %, decay over the
last 50 %); batches of up to 21 minutes of audio (`numel`, padding
included), one GPU, bf16 mixed precision. 20 epochs by default
(`MULTIDATASET_SER_03_EPOCHS`), with no early stopping; the validation
accuracy is measured every second epoch and the best checkpoint
(`valid.acc.ave_1best.pth`) is kept and scored. The weights of every tenth
epoch are kept under `exp/<tag>/epochs/`. Inference runs in fp32.

`lr_schedule.max_steps` counts the optimizer updates of the whole run: the
number of epochs times the updates of one epoch (see "Epochs and updates"
below). Set `MULTIDATASET_SER_03_MAX_STEPS` together with
`MULTIDATASET_SER_03_EPOCHS`.

Per model (`conf/training_<model>.yaml`, `conf/frontend/<model>.yaml`):

| | `hubert_large` | `conformer` | `conformer_m` | `conformer_s` | `ebranchformer` |
|---|---|---|---|---|---|
| Input normalisation | waveform, training-set statistics | log-mel, `GlobalMVN` | same | same | same |
| Positions | HuBERT's convolutional positions | absolute (sinusoidal), plain self-attention on SDPA | same | same | same |
| Peak lr (default) | 3e-5 | 5e-5 | 5e-4 | 1e-3 | 5e-5 |
| Batches | 21 min | two half batches per update | 21 min | 21 min | two half batches per update |

Set `MULTIDATASET_SER_03_LR` to try another peak lr and
`MULTIDATASET_SER_03_SEED` to train with another seed.

## Data augmentation

Chosen with `MULTIDATASET_SER_03_AUGMENT` (`conf/augment/<name>.yaml`; none
by default), in training only. Noise, reverberation and speed perturbation
run in ESPnet2's `CommonPreprocessor`; SpecAugment is ESPnet2's `SpecAug`.

| Name | What |
|---|---|
| `noise` | additive noise at 5 to 20 dB SNR on half of the utterances, from `${data_dir}/augment/noise.scp` |
| `rir` | reverberation on half of the utterances, from `${data_dir}/augment/rir.scp` |
| `speed` | speed perturbation (0.9 or 1.1) on two thirds of the utterances |
| `specaug_logmel` | SpecAugment on the log-mel features, with the settings most ESPnet2 Conformer recipes use (time warping, two frequency masks of up to 30 bins, two time masks of up to 40 frames) |
| `specaug_hubert` | HuBERT's own masking, stronger than its default |
| `all_logmel`, `all_hubert` | noise, reverberation and the masking of the model |

The lists are made once, after `create_dataset`:

```bash
# Noise: FSD50K (https://zenodo.org/records/4060432), about 31 GB, then the
# CC0 and CC BY clips without voice, crowds, breathing or music, as 16 kHz WAV
# under ${MULTIDATASET_SER_03_OUTPUT}/augment/ (with noise_attribution.tsv,
# the credits CC BY asks for).
local/download_fsd50k.sh <fsd50k_dir>
python local/make_noise_list.py <fsd50k_dir> ${MULTIDATASET_SER_03_OUTPUT}/augment
# Reverberation: the measured RIRs of BUT ReverbDB (CC BY 4.0), unpacked from
# BUT_ReverbDB_rel_19_06_RIR-Only.tgz
# (https://speech.fit.vut.cz/software/but-speech-fit-reverb-database).
local/make_rir_list.sh <reverbdb_dir> ${MULTIDATASET_SER_03_OUTPUT}
```

## Up-sampling small corpora

`MULTIDATASET_SER_03_SAMPLER=corpus_pow050` (or `corpus_pow075`,
`corpus_pow025`) draws each epoch corpus by corpus, with each corpus's share
of the audio raised to the power β, so that small corpora are seen more
often. It needs `${MULTIDATASET_SER_03_OUTPUT}/corpus2utt`, written by
`local/make_corpus2utt.sh ${MULTIDATASET_SER_03_OUTPUT}` after
`remove_long_short`.

## What `create_dataset` does

For each corpus in turn, it calls the corpus recipe's `DatasetBuilder`:
`prepare_source` downloads the corpus and `build` writes the 16 kHz audio and
manifests. A corpus already prepared is skipped, so the stage can be run again
after a failure. When every corpus is ready, this recipe's own merged
manifests are written (see above); with `MULTIDATASET_SER_03_ONLY` set they
are not.

- **One corpus at a time**, so that no server receives many downloads at once.
- **Retries.** A network error that can pass (connection reset, timeout, HTTP
  5xx/408/429, a failed `git` or `kaggle` call) is retried after 60 s, then
  120 s and 240 s (`retry` in `dataset/config.yaml`). HTTP 4xx, an untrusted or
  expired server certificate, a missing token and the like fail at once.
- **A failure does not stop the others.** The stage ends with a summary of the
  corpora that are ready and, for each one that is not, its error; it then
  fails, so the run shows that something is missing.

Downloading all 71 takes hours and needs no GPU; run it where long network
transfers and CPU-heavy audio conversion are allowed.

## Where the data goes

Each corpus keeps the locations of its own recipe, set by two environment
variables named after it (see each `egs3/<corpus>/cls/readme.md`):

- `<CORPUS>` — the downloaded corpus, or where to download it to
  (e.g. `RAVDESS`); default `download/` under that corpus's recipe
- `<CORPUS>_OUTPUT` — where the converted audio and manifests are written
  (e.g. `RAVDESS_OUTPUT`); default `data/` under that corpus's recipe

The variable name is the corpus name in upper case. No audio is copied into
this recipe: the merged manifests point at the files the corpus recipes wrote.

One more variable belongs to this recipe:

- `MULTIDATASET_SER_03_OUTPUT` — where this recipe writes its stage logs, its
  merged manifests, the label list and the class weights; default `data/`
  under this recipe. Use a different place from `MULTIDATASET_SER_01_OUTPUT`:
  the label mapping differs, so the merged manifests differ.

## Requirements

Tools some corpus recipes call:

| Tool | Needed by |
|---|---|
| `cryptography` (Python) | aesdd (MEGA) |
| `git-lfs`, after `git lfs install` | cremad |
| `bsdtar` (libarchive) | oreau (RAR) |
| `kaggle` (Python package; some builds also need `six`) | baser, cbesd, emospeech_keywords, eseravd, jlcorpus, saraiki_ser, ses_ed, ses_sd, telugu_emotion_kaggle |
| `ffmpeg` | most recipes |

Accounts and tokens:

- **Kaggle** (the 9 corpora above): a free account and an API token from
  https://www.kaggle.com/settings/api, in `~/.kaggle/access_token` or
  `$KAGGLE_API_TOKEN`.
- **Hugging Face, gated** (emotiontalk and rasa): see below.

### Hugging Face: emotiontalk and rasa

Both datasets are public but gated: the files download only for an account that
has agreed on the dataset page. Access is granted at once, with no review
(`gated: auto` in the Hugging Face API, checked 2026-11-09). Do this once per
account:

1. Sign in to Hugging Face (https://huggingface.co/login; create an account
   at https://huggingface.co/join if needed).
2. Open each dataset page and, in the box at the top that asks you to agree to
   share your contact information, accept the conditions:
   - emotiontalk: https://huggingface.co/datasets/BAAI/Emotiontalk (CC BY-NC-SA 4.0)
   - rasa: https://huggingface.co/datasets/ai4bharat/Rasa (CC BY 4.0)

   Once accepted, the page shows that you have been granted access, and the
   "Files and versions" tab lists the files.
3. Create a token at https://huggingface.co/settings/tokens. A token of type
   **Read** works. A **Fine-grained** token works only with "Read access to
   contents of all public gated repos you can access" ticked.
4. Make the token available on the machine that runs the recipe, in either of
   the places the recipes read:
   - `hf auth login` (from the `huggingface_hub` package; older versions call it
     `huggingface-cli login`), which writes `~/.cache/huggingface/token`, or
   - the environment variable `HF_TOKEN`.

If the agreement or the token is missing, the recipe fails with
"Access to ... was refused" and names the page to agree on.

### dusha: expired certificate

The certificate of dusha's CDN expired on 2026-11-04 and had not been renewed on
2026-11-10. Until it is, set `DUSHA_ALLOW_EXPIRED_CERTIFICATE=1` when running
`create_dataset`; otherwise dusha fails with "certificate has expired" (the other
corpora are not affected). The setting skips only the check of the certificate's
validity dates; the issuer chain and the host name are still verified. Details are in
`egs3/dusha/cls/readme.md`.

## Corpora on hold (as of 2026-11-10)

This one is commented out in `dataset/config.yaml`. Once its server works
again, put it back in `corpora` and add it to `train_corpora` (MIT).

| Corpus | Why |
|---|---|
| enterface05 | https://enterface.net/enterface05/emotion.html answered no request on 2026-11-09, from three different networks; it was reachable on 2026-11-01. On 2026-11-10 the server accepted the TLS connection but sent no page within 90 s. The server also sends a wrong certificate chain (its certificate is issued by Let's Encrypt YR2, but it sends a GEANT/HARICA chain), which is why `egs3/enterface05/cls` asks for a browser download; the chain does verify once the missing certificates are fetched from the addresses in the certificates (`http://yr2.i.lencr.org/`, then `http://yr.i.lencr.org/`, which is cross-signed by ISRG Root X1). When the site is back, download `project2_database.zip` in a browser and put it in `$ENTERFACE05` |

## Troubleshooting

**Zenodo downloads fail with `SSL: UNEXPECTED_EOF_WHILE_READING` after about
60 s.** OpenSSL 3.5 and later offer a post-quantum key exchange (ML-KEM) by
default, which makes the TLS ClientHello about 1.5 KB, more than one packet. On
some networks a ClientHello of that size does not get through to zenodo.org,
while a small one does (seen on 2026-11-09: `openssl s_client` with its default
groups failed after 65 s, with `-groups X25519` it connected in 1 s). If this
happens, give Python an OpenSSL configuration without ML-KEM:

```
openssl_conf = openssl_init

[openssl_init]
ssl_conf = ssl_sect

[ssl_sect]
system_default = system_default_sect

[system_default_sect]
Groups = X25519:secp256r1:X448:secp384r1
```

and set `OPENSSL_CONF` to that file when running `create_dataset`. Check first
with `openssl s_client -connect zenodo.org:443` (with and without
`-groups X25519`).

## Epochs and updates

`lr_schedule.max_steps` must equal the number of epochs times the updates of
one epoch. With the data of `dataset/config.yaml` (266,701 training
utterances, 449.7 hours after the duration filter) and batches of 21
minutes on one GPU:

| Sampler | Updates per epoch | 20 epochs | 40 epochs | 60 epochs |
|---|---:|---:|---:|---:|
| `numel` (default) | 1,283 | 25,660 | 51,320 | 76,980 |
| `corpus_pow075` | 1,314 | 26,280 | 52,560 | 78,840 |
| `corpus_pow050` | 1,323 | 26,460 | 52,920 | 79,380 |
| `corpus_pow025` | 1,335 | 26,700 | 53,400 | 80,100 |

The models that take two half batches per update (`conformer`,
`ebranchformer`) make the same number of updates. Count again after a
change of the data or the batch, e.g. with ESPnet2's
`NumElementsBatchSampler` on `exp/stats/train/feats_shape`.

## Running

Run each line as its own command. Data preparation needs no GPU; training
and inference use one GPU each. `collect_stats` writes the statistics of
the waveforms (HuBERT) and of the log-mel features (the log-mel models),
and the utterance lengths; it is run once with a log-mel model's config.

```bash
python run.py --stages create_dataset remove_long_short prepare_labels collect_stats \
    --training_config conf/training_conformer.yaml
local/make_corpus2utt.sh ${MULTIDATASET_SER_03_OUTPUT:-data}   # for up-sampling only
# A model: conformer, conformer_m, conformer_s, ebranchformer or hubert_large
python run.py --stages train --training_config conf/training_conformer.yaml
python run.py --stages infer measure \
    --training_config conf/training_conformer.yaml \
    --inference_config conf/inference_conformer.yaml \
    --metrics_config conf/metrics.yaml
# e.g. SpecAugment, 40 epochs
MULTIDATASET_SER_03_EXP_TAG=conformer_specaug_e40 MULTIDATASET_SER_03_AUGMENT=specaug_logmel \
MULTIDATASET_SER_03_EPOCHS=40 MULTIDATASET_SER_03_MAX_STEPS=51320 \
    python run.py --stages train --training_config conf/training_conformer.yaml
```

Results go to `exp/<exp_tag>/inference/` (`exp_tag` defaults to
`train_<model>`). Set `MULTIDATASET_SER_03_EXP_TAG` to use another directory
under `exp/` (for `train` and for `infer measure` alike).

To prepare only some corpora, name them in `MULTIDATASET_SER_03_ONLY` (the
merged manifests are then not written):

```bash
MULTIDATASET_SER_03_ONLY="ravdess emodb" \
    python run.py --stages create_dataset --training_config conf/training_conformer.yaml
```
