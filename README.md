# Codemate

Codemate is a local-first command-line assistant for codebases. It tracks a Git
working tree, builds a local chunk and vector index, and can answer questions
using retrieved code context and a configurable chat model.

## Install for development

```sh
python -m pip install -e .
```

## Track a repository

From a Git repository, run:

```sh
codemate track
```

Or point to a repository explicitly:

```sh
codemate track --repo /path/to/repository
```

Codemate uses read-only Git commands to list tracked files and non-ignored
untracked files, then fingerprints their current working-tree contents. It
stores hashes, byte sizes, line counts for text files, and a binary marker in
`.index/manifest.json`. Symlinks are recorded by their link target without
following them. The tracker reports files added, changed, unchanged, or removed
since the last run. The `.index/` folder is excluded from tracking. To keep the
local index out of Git, add `.index/` to the repository's `.gitignore`.

The tracker inventories binary and large files too; later RAG ingestion can
choose which file types to decode, chunk, and embed. The index is rebuildable
local metadata and should not be committed.

## Chunk and embed for RAG

Install the optional local embedding dependency:

```sh
python -m pip install -e '.[embeddings]'
```

Then run:

```sh
codemate index
```

The first indexing run downloads the default `BAAI/bge-small-en-v1.5` model if
it is not already cached. Embeddings are generated locally. Text chunks and
vectors are stored in `.index/vectors.sqlite3`; file paths and line ranges are
kept with each chunk for later citations. The tracker still records every
Git-visible file, while embedding skips symlinks, binary or non-UTF-8 files,
and files larger than 1 MB. Only changed files are chunked and embedded again.
Chunks target 1,200 characters with 160 characters of overlap and prefer
newline boundaries; these are initial defaults, not language-aware syntax
boundaries. Changing the model or chunk settings rebuilds the derived vector
index.

## Ask questions

Gemma 4 is the default chat model, served through Google's Gemini API. Create a
`.env` file in the repository root from the example and add your Google AI
Studio API key. Ensure `.env` is listed in that repository's `.gitignore`;
Codemate skips `.env` files during embedding:

```sh
cp .env.example .env
```

The example uses `gemma-4-26b-a4b-it` and the Gemini API endpoint. Google also
lists `gemma-4-31b-it` as a hosted Gemma 4 model. Set `GEMINI_API_KEY` in
`.env`. Shell environment variables override `.env`, and `--model` or
`--base-url` can override settings for one command. To use another
OpenAI-compatible provider, set `CODEMATE_PROVIDER=openai-compatible`, its base
URL and model, and choose its key variable with `--api-key-env`.

Then ask a question:

```sh
codemate ask "Where is authentication handled?"
```

Codemate embeds the question locally with the same model used to build the
index, retrieves the closest chunks, and sends only those excerpts and the
question to the configured chat endpoint. The model is instructed to treat
repository text as untrusted and cite the retrieved source labels. Codemate
also prints the exact source paths and line ranges it retrieved. The first
version uses a local cosine scan of the SQLite vectors; it does not yet use an
approximate-nearest-neighbor index or hybrid keyword search.
