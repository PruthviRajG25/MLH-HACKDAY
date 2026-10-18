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
newline and common function/class boundaries. Each chunk records its language
and nearest recognized symbol where available. Changing the model or chunk
settings rebuilds the derived vector index.

## Ask questions

Gemma 4 is the default chat model, served through Google's Gemini API. Configure
the provider once for your user account:

```sh
codemate init
```

The setup prompts for provider, model, endpoint, and API key. Settings are
saved outside the repository in `~/.config/codemate/config.json` (or
`$XDG_CONFIG_HOME/codemate/config.json`), with private file permissions. To
review or change settings later:

```sh
codemate config list
codemate config set model gemma-4-31b-it
codemate config set api-key
codemate config unset api-key
```

The API key prompt is hidden. Avoid passing a key as a command argument because
shell history may retain it. The example model is `gemma-4-26b-a4b-it`; another
hosted Gemma 4 model is `gemma-4-31b-it`. CLI flags override saved settings,
and environment variables override saved settings. Existing repository `.env`
settings remain a fallback for compatibility.

Then ask a question:

```sh
codemate ask "Where is authentication handled?"
```

To run a command and diagnose it automatically if it fails, use `codemate run`:

```sh
codemate run pytest -q
codemate run -- npm test
```

Codemate prints the command output and error first, then retrieves relevant code
and asks the configured model for a diagnosis. Before diagnosing, it updates an
existing local index so newly added or changed files can be retrieved. The
wrapper returns the original command's exit status. It executes arguments directly without a shell; for
shell syntax, pass a shell explicitly (for example,
`codemate run sh -c 'make check'`). On failure, Codemate sends the command and up to the final 12,000
characters of its output to the configured AI provider along with retrieved
repository excerpts. Avoid including secrets in command arguments or logs.
After showing a diagnosis in an interactive terminal, Codemate asks whether to
save a Markdown report containing the command, captured output, diagnosis, and
retrieved excerpts. Reports go into the repository's `report/` directory and
are ignored by Git and future indexing by default.

Failed commands are also recorded in a local Failure Atlas at
`.index/failures.sqlite3`. It groups failures by a normalized output signature
and the exact command, tracks occurrence counts and the latest Git `HEAD` commit, and
stores the latest diagnosis. When a matching failure returns, Codemate gives
the previous diagnosis to the model as historical context and asks it to
re-check that diagnosis against current evidence. Saving a report links it to
the incident. A successful run resolves active incidents for that exact
command; normal successful runs remain quiet. View active incidents with
`codemate failures`; include resolved incidents with:

```sh
codemate failures --all
```

The history stays local alongside the index and can be removed by
deleting `.index/failures.sqlite3`.

Codemate embeds the question locally with the same model used to build the
index, retrieves the closest chunks, and sends only those excerpts and the
question to the configured chat endpoint. The model is instructed to treat
repository text as untrusted and cite the retrieved source labels. Codemate
renders common Markdown in the terminal, including headings, lists, code, and
tables; it uses color only for interactive terminals and honors `NO_COLOR`.
Codemate also prints the exact source paths and line ranges it retrieved. Retrieval
combines vector similarity with keyword matching, filters overlapping duplicate
chunks, and reports when no result clears its relevance threshold. Large
indexes use batched NumPy scoring when NumPy is available; retrieval still
compares against the full vector set, so an approximate-nearest-neighbor index
may be worthwhile for very large repositories.

Use `codemate doctor` to inspect configuration and index status and make a
read-only API model check. It reports only whether a key is configured, never
the key itself. The check sends an authenticated GET request to the configured
provider, without generating a model response.

To measure retrieval changes, keep representative questions and the files that
should answer them in `.codemate-eval.json`:

```json
{
  "cases": [
    {
      "question": "How are tracked files discovered?",
      "expected_paths": ["codemate/tracker.py"]
    }
  ]
}
```

Run `codemate evaluate` to see Recall@k and mean reciprocal rank (MRR@k).
Codemate includes a starter evaluation set for its own codebase; use
`codemate evaluate --file path/to/cases.json` for another set.
