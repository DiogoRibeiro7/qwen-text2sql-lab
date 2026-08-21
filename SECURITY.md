# Security Policy

## Supported versions

This is a research repository under active development. Security fixes are
applied to the `main` branch only.

| Version | Supported |
|---|---|
| `main` | Yes |
| tagged releases behind `main` | No |

## Reporting a vulnerability

**Do not open a public issue for a security vulnerability.**

Report privately through GitHub:

1. Open the [security advisory form](https://github.com/DiogoRibeiro7/qwen-text2sql-lab/security/advisories/new).
2. Describe the issue, the affected component, and how to reproduce it.

Expect an initial acknowledgement within 7 days and an assessment within 30
days. If a fix is warranted, the advisory is published once the fix lands on
`main`.

## Scope

This project executes model-generated SQL in order to score it. That makes the
execution path the most security-relevant part of the codebase.

In scope:

- Escaping the read-only SQLite execution sandbox in
  [`src/qwen_text2sql/evaluation/execution.py`](src/qwen_text2sql/evaluation/execution.py)
  — writes, `ATTACH`, file access, loading extensions, or otherwise mutating state.
- Path traversal through `db_path`, `--db-root`, or config-supplied paths.
- Deserialisation issues in prepared JSONL, configuration files, or downloaded
  artifacts.
- Arbitrary code execution triggered by loading an adapter or a configuration.
- Dependency vulnerabilities reachable through normal use of the CLI.

Out of scope:

- The model generating incorrect or dangerous SQL. Detecting that is the purpose
  of the evaluator, not a vulnerability in it.
- Vulnerabilities in BIRD data or Qwen model weights themselves. Report those
  upstream.
- Denial of service from an intentionally expensive query. Execution is
  time-limited by design, and the limit is a research parameter.

## Operating guidance

- Treat every generated query as untrusted input.
- Run evaluation against **copies** of benchmark databases, never originals.
- Do not point the evaluator at a database holding real or sensitive data.
- Prefer an isolated environment or container when evaluating adapters you did
  not train yourself.
