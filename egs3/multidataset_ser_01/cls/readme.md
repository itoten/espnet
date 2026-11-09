# Multi-dataset speech emotion recognition (multidataset_ser_01)

Experiment `multidataset_ser_01` uses the 72 corpora that the dataset filter of
ser_dataset_db keeps (`local/dataset_filtering/output/multidataset_ser_01/`,
generated 2026-11-09 from DB commit 25de395). They are listed in
`dataset/config.yaml`; two of them are on hold (see below), so 70 are prepared.

Only `create_dataset` can run yet. It downloads and converts every corpus with
that corpus's own `egs3/<corpus>/cls` recipe, the way `egs3/multicorpus_ood20/cls`
does. Which corpora are used for training and which for evaluation, and how
their labels are unified, are still being decided (ser_dataset_db
`docs/discussions/0003-license-classes.md` and `0004-label-mapping.md`); the
merged manifests, training and evaluation are added after that.

## What `create_dataset` does

For each corpus in turn, it calls the corpus recipe's `DatasetBuilder`:
`prepare_source` downloads the corpus and `build` writes the 16 kHz audio and
manifests. A corpus already prepared is skipped, so the stage can be run again
after a failure.

- **One corpus at a time**, so that no server receives many downloads at once.
- **Retries.** A network error that can pass (connection reset, timeout, HTTP
  5xx/408/429, a failed `git` or `kaggle` call) is retried after 60 s, then
  120 s and 240 s (`retry` in `dataset/config.yaml`). HTTP 4xx, an untrusted or
  expired server certificate, a missing token and the like fail at once.
- **A failure does not stop the others.** The stage ends with a summary of the
  corpora that are ready and, for each one that is not, its error; it then
  fails, so the run shows that something is missing.

Downloading all 70 takes hours and needs no GPU; run it where long network
transfers and CPU-heavy audio conversion are allowed.

## Where the data goes

Each corpus keeps the locations of its own recipe, set by two environment
variables named after it (see each `egs3/<corpus>/cls/readme.md`):

- `<CORPUS>` — the downloaded corpus, or where to download it to
  (e.g. `RAVDESS`); default `download/` under that corpus's recipe
- `<CORPUS>_OUTPUT` — where the converted audio and manifests are written
  (e.g. `RAVDESS_OUTPUT`); default `data/` under that corpus's recipe

The variable name is the corpus name in upper case. Nothing is copied into this
recipe: the merged manifests to come will point at the files under each
`<CORPUS>_OUTPUT`.

One more variable belongs to this recipe:

- `MULTIDATASET_SER_01_OUTPUT` — where this recipe writes its stage log
  (`create_dataset.log`) and, later, its merged manifests; default `data/`
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

## Corpora on hold (as of 2026-11-09)

These two are commented out in `dataset/config.yaml`. Put them back once their
servers work again.

| Corpus | Why |
|---|---|
| enterface05 | https://enterface.net/enterface05/emotion.html answered no request on 2026-11-09, from three different networks; it was reachable on 2026-11-01. The server also sends a wrong certificate chain (its certificate is issued by Let's Encrypt YR2, but it sends a GEANT/HARICA chain), which is why `egs3/enterface05/cls` asks for a browser download; the chain does verify once the missing certificates are fetched from the addresses in the certificates (`http://yr2.i.lencr.org/`, then `http://yr.i.lencr.org/`, which is cross-signed by ISRG Root X1). When the site is back, download `project2_database.zip` in a browser and put it in `$ENTERFACE05` |
| dusha | The CDN's certificate expired on 2026-11-04. Wait for the server to renew it; the recipe then works unchanged. The certificate check is not disabled: no checksum is published, so the 30 GB download could not be verified otherwise |

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

```bash
python run.py --stages create_dataset --training_config conf/training.yaml
```

To prepare only some corpora, name them in `MULTIDATASET_SER_01_ONLY`:

```bash
MULTIDATASET_SER_01_ONLY="ravdess emodb" \
    python run.py --stages create_dataset --training_config conf/training.yaml
```
