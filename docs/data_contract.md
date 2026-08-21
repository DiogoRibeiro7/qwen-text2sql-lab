# Data Contract

Prepared JSONL records contain exactly the fields required for reproducible training and execution evaluation:

| Field | Type | Meaning |
|---|---|---|
| `example_id` | string | stable hash of database, question and reference SQL |
| `db_id` | string | database identifier |
| `question` | string | natural-language request |
| `evidence` | string | optional benchmark-provided evidence |
| `gold_sql` | string | verified reference SQL |
| `schema` | string | deterministic SQLite schema text |
| `db_path` | string | local database path used for evaluation |
| `difficulty` | string | optional benchmark difficulty label |
| `source_id` | string | optional benchmark-native example identifier |

`db_path` is environment-local and is therefore not a portable artifact by itself. Re-preparing the same benchmark on another machine reconstructs the paths from `db_id` and the supplied database root.
