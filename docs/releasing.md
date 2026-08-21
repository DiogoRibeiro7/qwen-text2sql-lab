# Releasing and archiving

A release of this repository does two things: it tags a state of the code, and it
deposits that state with [Zenodo](https://zenodo.org), which mints a DOI so the
work can be cited from a paper.

## Before anything else: Zenodo needs a public repository

**Zenodo cannot archive a private repository.** Its GitHub integration reads the
release tarball through the GitHub API using permissions a private repository
does not grant, so the toggle for a private repository either does not appear or
does nothing.

This repository is currently private. Until it is made public, the metadata below
is prepared and validated but no DOI can be minted. Making it public is the same
decision that would re-enable CI (see [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)),
so the two are worth taking together.

Before making it public, note that the commit author email becomes visible in the
history of every commit.

## One-time setup

1. Sign in to [zenodo.org](https://zenodo.org) with the GitHub account that owns
   the repository.
2. Go to **Account → GitHub**, and let Zenodo read the repository list.
3. Flip the toggle for `DiogoRibeiro7/qwen-text2sql-lab` to **On**.

Zenodo installs a webhook and will archive **releases created after the toggle**.
It does not backfill: a release published before the toggle is never archived, so
the toggle has to come first.

## Cutting a release

### 1. Agree the version with itself

The version appears in three files, and Zenodo reads `.zenodo.json` at the moment
of publication. A stale version there is archived permanently and cannot be
edited into agreement afterwards.

```bash
make release-check
```

That checks `pyproject.toml`, `CITATION.cff` and `.zenodo.json` agree on version,
title and licence, that `.zenodo.json` carries the fields Zenodo requires, and
that the changelog has somewhere for the release to go.

To bump, edit all three, plus `date-released` in `CITATION.cff`, then re-run it.

### 2. Close the changelog

Move the accumulated `## [Unreleased]` entries under `## [X.Y.Z] - YYYY-MM-DD`,
add a fresh empty `Unreleased`, and update the link definitions at the foot of
the file.

### 3. Verify the code

```bash
make check       # lint, format, typecheck, tests
make notebooks
poetry check --lock
```

CI does not run while the repository is private, so this is the only gate.

### 4. Tag and publish

```bash
git tag -a v0.1.0 -m "v0.1.0"
git push origin v0.1.0
gh release create v0.1.0 --title "v0.1.0" --notes-from-tag
```

Zenodo reacts to the **published release**, not to the tag. Pushing a tag alone
archives nothing.

### 5. Record the DOI

Zenodo mints two:

| DOI | Resolves to | Use it for |
|---|---|---|
| **Concept DOI** | always the newest version | citing the project in general |
| **Version DOI** | this release, permanently | citing the exact code behind a result |

A paper reporting a number should cite the **version** DOI. The concept DOI is
for a general reference to the software.

Add the concept DOI to `CITATION.cff`:

```yaml
doi: "10.5281/zenodo.XXXXXXX"
```

and the badge to the top of `README.md`:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

Then commit both, and check the Zenodo record renders the licence as MIT and the
authors correctly — the licence identifier is the field most likely to be
misread, and it is easier to fix on the draft than after publication.

## What is not archived

The deposit contains the repository at the tagged commit and nothing else. It
does **not** contain BIRD databases, Qwen model weights, trained adapters, or
generated predictions — all of those are gitignored, and the first two are not
ours to redistribute in any case. Anyone reproducing a result needs to obtain
them from their own sources, which is what
[`docs/data_contract.md`](data_contract.md) and the README's data section are
for.

## Adding an ORCID

`.zenodo.json` currently lists the author by name only. An ORCID makes the
deposit attributable across name variants and institutions:

```json
"creators": [
  {"name": "Ribeiro, Diogo", "orcid": "0000-0000-0000-0000"}
]
```

It is deliberately absent rather than guessed.
