# Multi-dataset speech emotion recognition (multidataset_ser_01)

Experiment `multidataset_ser_01` uses the 72 corpora that the dataset filter of
ser_dataset_db keeps (`local/dataset_filtering/output/multidataset_ser_01/`,
generated 2026-11-09 from DB commit 3ff353c). They are listed in
`dataset/config.yaml`; one of them is on hold (see below), so 71 are prepared.

The `cls` system (a pre-trained speech encoder, attentive statistics pooling
and a fully connected head) is trained on the corpora whose license allows
commercial use and scored on all the others, with WavLM Large as the encoder.
Every corpus is downloaded and converted by its own `egs3/<corpus>/cls`
recipe, the way `egs3/multicorpus_ood20/cls` does.

## Training and evaluation data

| Role | Corpora | Rows |
|---|---|---:|
| Training and validation | 45 corpora whose license allows commercial use with no share-alike condition | about 278,000 |
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
- **Labels.** Only seven classes are used, in training and evaluation alike:
  neutral, happy, sad, angry, fear, surprise and disgust (`labels`). A row
  with any other label (calm, boredom, RASA's text styles, ...) is left out of
  every manifest; the build log counts them per corpus and label. An
  evaluation corpus that lacks some of the seven is scored over the ones it
  has. `label_map` only fixes spellings (RASA's `anger` becomes `angry`).
- **Validation.** Taken out of the training corpora as `validation.method`
  says:

  | `method` | Validation rows |
  |---|---|
  | `random_per_corpus` (default) | `ratio` (0.05) of each training corpus's rows, chosen uniformly at random with `seed` (0), regardless of speakers; at least one row per corpus. The choice depends only on the seed and the corpus's own rows |
  | `corpus_valid_split` | each corpus recipe's own `valid` split (how each recipe splits differs: by speaker, at random, ...) |

  Another method is a function in `dataset/builder.py` registered in
  `VALIDATION_METHODS`, with its own block under `validation`.

`create_dataset` writes `manifest/train.tsv`, `manifest/valid.tsv` and
`manifest/<eval corpus>.tsv` under `$MULTIDATASET_SER_01_OUTPUT`, and the
settings they were built with to `manifest/build_settings.json`. If the
roles, labels or validation settings change, the next `create_dataset`
rewrites the manifests.

## Model and training

The settings are those of `egs3/multicorpus_ood20/cls` for WavLM Large
(`conf/frontend/wavlm_large.yaml`: `microsoft/wavlm-large`, waveform
standardised with the training set's statistics), with seven classes and
validation: AdamW at lr 1e-5, batch size 8 per GPU, after every epoch the
validation accuracy is measured, the best checkpoint
(`valid.acc.ave_1best.pth`) is kept and scored, and training stops after 3
epochs without improvement (at most 10). These values are provisional and not
tuned for this amount of data.

## What `create_dataset` does

For each corpus in turn, it calls the corpus recipe's `DatasetBuilder`:
`prepare_source` downloads the corpus and `build` writes the 16 kHz audio and
manifests. A corpus already prepared is skipped, so the stage can be run again
after a failure. When every corpus is ready, this recipe's own merged
manifests are written (see above); with `MULTIDATASET_SER_01_ONLY` set they
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

- `MULTIDATASET_SER_01_OUTPUT` — where this recipe writes its stage logs, its
  merged manifests, the label list and the class weights; default `data/`
  under this recipe

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

## Running

Run each line as its own command. Under multi-GPU DDP, Lightning re-runs the
whole command once per GPU, so data preparation or inference chained onto
`train` would run several times at once.

```bash
python run.py --stages create_dataset remove_long_short prepare_labels collect_stats \
    --training_config conf/training_wavlm_large.yaml
python run.py --stages train --training_config conf/training_wavlm_large.yaml
python run.py --stages infer measure \
    --training_config conf/training_wavlm_large.yaml \
    --inference_config conf/inference_wavlm_large.yaml \
    --metrics_config conf/metrics.yaml
```

Results go to `exp/train_wavlm_large/inference/`.

To prepare only some corpora, name them in `MULTIDATASET_SER_01_ONLY` (the
merged manifests are then not written):

```bash
MULTIDATASET_SER_01_ONLY="ravdess emodb" \
    python run.py --stages create_dataset --training_config conf/training_wavlm_large.yaml
```
