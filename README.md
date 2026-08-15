# pi-zeromem

A [pi coding agent](https://github.com/earendil-works/pi-mono) extension that wires
the [`zm`](https://github.com/onibaku/zeromem) memory CLI into pi as zero-token,
provenance-preserving long-term memory.

Wraps the `zm` binary (Zero-Mem, arXiv 2607.29377): recall/ingest/stats make
**zero LLM calls and zero tokens** — retrieval is structured search over raw
conversation traces (entity-context graph + temporal hierarchy + BM25/dense
vectors), never generated summaries.

## Install

Symlink into pi's auto-discovered extensions dir (global), then `/reload`:

```sh
ln -s "$(pwd)/zeromem.ts" ~/.pi/agent/extensions/zeromem.ts
```

Alternative: list the file in `~/.pi/agent/settings.json`:

```json
{
  "extensions": ["/home/jhsu/code/pi-zeromem/zeromem.ts"]
}
```

Requires the `zm` binary on `PATH` (install from the [zeromem repo](https://github.com/onibaku/zeromem)):

```sh
just build   # produces target/release/zm; put it on PATH
```

## Tools exposed to the agent

| Tool | Purpose |
|------|---------|
| `zm_recall` | Ranked verbatim evidence from past turns with provenance (session/turn/speaker). Zero tokens. |
| `zm_ingest` | Persist turns (structured `{ session_id, speaker, text, ts }` or the `turns`/`session_id`/`speaker`/`text` form). |
| `zm_stats` | Store counters: turns, sessions, entities, windows, episodes. |

Plus an interactive `/zm query <text>` / `/zm stats` command.

## Automatic context preservation on compaction

On `session_before_compact`, the extension serializes the messages pi is about to
summarize and drop, splits them into speaker-tagged turns, and ingests them to
`zm` — so context that leaves the live window stays retrievable. Failures are
swallowed (never block compaction), and repeated auto-compactions are deduped by
`preparation.firstKeptEntryId`.

## Config

Environment variables (defaults shown):

| Var | Default | Meaning |
|-----|---------|---------|
| `ZM_DB` | `~/.zm/zeromem.db` | SQLite store path |
| `ZM_NO_MODEL` | *(unset)* | `1` = use the deterministic hash embedder (no onnxruntime / ~130MB BGE download, fully offline) |
| `ZM_PRESERVE_ON_COMPACT` | `1` | `0` = disable the auto-ingest-before-compaction hook |

To avoid the model download entirely, set `ZM_NO_MODEL=1` in your shell profile, or
build `zm` with `--no-default-features`.

## Layout

```
zeromem.ts   the extension (single file)
package.json optional pi-package metadata (installable via `pi install`)
```

## License

MIT
