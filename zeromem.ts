import {
  convertToLlm,
  serializeConversation,
  type ExtensionAPI,
} from "@earendil-works/pi-coding-agent";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { spawn } from "node:child_process";
import { promises as fs } from "node:fs";
import os from "node:os";
import path from "node:path";

/**
 * pi extension that integrates the zeromem (zm) memory tool.
 *
 * Wraps the `zm` CLI so the agent gets zero-token, provenance-preserving
 * recall/ingest over raw conversation traces. No LLM calls in the memory path.
 *
 * Tools exposed to the LLM:
 *   - zm_recall   query past turns, returns ranked verbatim evidence
 *   - zm_ingest   persist a JSONL file of turns ({"session_id","speaker","text","ts"})
 *   - zm_stats    store counters (turns, sessions, entities, windows, episodes)
 *
 * Commands:
 *   /zm query <text>   recall from the interactive prompt
 *   /zm stats          show store counters
 *   /zm ingest <jsonl> persist a turns file
 *
 * Config via env (defaults shown):
 *   ZM_DB        ~/.zm/zeromem.db   SQLite store path
 *   ZM_NO_MODEL  1                  use the deterministic hash embedder
 *                                   (no onnxruntime / ~130MB model download)
 *   ZM_PRESERVE_ON_COMPACT  0|1     auto-ingest pre-compaction context to zm.
 *                                   Default: 1 (on). Set to 0 to disable.
 */

// Tunables for compaction-preservation.
const PRESERVE_ON_COMPACT =
  process.env.ZM_PRESERVE_ON_COMPACT !== "0" &&
  process.env.ZM_PRESERVE_ON_COMPACT !== "false";
/** Skip really large/low-signal messages (e.g. huge tool dumps). */
const MAX_TURN_CHARS = 4000;

const DB =
  process.env.ZM_DB || path.join(os.homedir(), ".zm", "zeromem.db");
const NO_MODEL = process.env.ZM_NO_MODEL === "1" || process.env.ZM_NO_MODEL === "true";

const RECALL_PARAMS = Type.Object({
  query: Type.String({ description: "What to look up in past conversations" }),
  top_k: Type.Optional(
    Type.Integer({ description: "Max main evidence items", default: 5 }),
  ),
});
const INGEST_PARAMS = Type.Object({
  turns: Type.Array(
    Type.Object({
      session_id: Type.String({ description: "Session identifier" }),
      speaker: Type.String({ description: "user | assistant | human name" }),
      text: Type.String({ description: "The turn text / utterance" }),
      ts: Type.Optional(Type.Integer({ description: "Unix timestamp; defaults to ingest order" })),
    }),
    { description: "Conversation turns to persist" },
  ),
  session_id: Type.Optional(
    Type.String({ description: "Session id to stamp on turns missing one (defaults to 'pi')" }),
  ),
});
const STATS_PARAMS = Type.Object({});

/** Run `zm` with `--db` and optional `--no-model`, returning stdout. */
function zm(args: string[]): Promise<string> {
  return new Promise((resolve, reject) => {
    const fullArgs = [`--db`, DB];
    if (NO_MODEL) fullArgs.push("--no-model");
    fullArgs.push(...args);

    const child = spawn("zm", fullArgs, { stdio: ["ignore", "pipe", "pipe"] });
    let out = "";
    let err = "";
    child.stdout.on("data", (c: Buffer) => { out += c.toString(); });
    child.stderr.on("data", (c: Buffer) => { err += c.toString(); });
    child.on("error", (e) => reject(new Error(`zm: ${e.message}`)));
    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(err.trim() || `zm exited with code ${code}`));
      } else {
        resolve(out);
      }
    });
  });
}

/** Write a JSONL turns file to a temp dir and ingest it. */
async function ingestJsonl(turns: Array<Record<string, unknown>>): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), "pi-zeromem-"));
  const file = path.join(dir, "turns.jsonl");
  const lines = turns
    .map((t) => JSON.stringify(t))
    .join("\n");
  await fs.writeFile(file, lines + "\n");
  try {
    const out = await zm(["ingest", file]);
    return out.trim();
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
}

/**
 * Flatten one outgoing AgentMessage into clean zeromem turns.
 *
 * Serializes the message the same way pi does for compaction, then splits on
 * each `[Role]:` block so the store gets discrete, correctly-speaker-tagged
 * turns (better entity + temporal retrieval) rather than one blob.
 */
function messageToTurns(msg: AgentMessage, sessionId: string, baseTs: number): Array<Record<string, unknown>> {
  const serialized = serializeConversation(convertToLlm([msg]));
  const out: Array<Record<string, unknown>> = [];
  let n = 0;
  for (const block of serialized.split(/\n\s*\n/)) {
    const trimmed = block.trim();
    if (!trimmed) continue;
    const m = trimmed.match(/^\[([^\]]+)\]:\s?([\s\S]*)$/);
    const speaker = m ? m[1].toLowerCase() : "message";
    const text = m ? m[2].trim() : trimmed;
    if (!text || text.length <= 1) continue;
    // Cap the size of a single stored turn to keep embedding/indexing cheap.
    const stored = text.length > MAX_TURN_CHARS
      ? text.slice(0, MAX_TURN_CHARS) + "\n…(truncated)"
      : text;
    out.push({
      session_id: sessionId,
      speaker,
      text: stored,
      ts: baseTs + n++,
    });
  }
  return out;
}

/** Push turns to zeromem immediately (used by the compaction hook). */
async function persistTurns(turns: Array<Record<string, unknown>>): Promise<string | null> {
  if (turns.length === 0) return null;
  try {
    return await ingestJsonl(turns);
  } catch (error) {
    console.error("[zeromem] compact-persist failed:", error);
    return null;
  }
}

/** Parse `zm query` output into a route line + normalized evidence lines. */
function formatQueryOutput(out: string): string {
  const lines = out.split("\n").filter((l) => l.trim().length > 0);
  if (lines.length === 0) return "No memory matches.";
  const route = lines[0].startsWith("route:") ? lines[0] : `route: ${lines[0]}`;
  const evidence = lines
    .slice(1)
    .filter((l) => !l.startsWith("route:"))
    .map((l) => `- ${l}`);
  return evidence.length ? `${route}\nEvidence:\n${evidence.join("\n")}` : route;
}

export default function zeromem(pi: ExtensionAPI) {
  // Remember entry ids we've already preserved so repeat auto-compactions
  // (each is a new span) don't silently re-ingest identical prefixes. We key
  // on the current session file so it stays scoped to one conversation branch.
  let persistedUpToSession: string | undefined;
  const persistedPrefixes = new Set<string>();

  const summarizeEvidence = (out: string): string => {
    const idx = out.indexOf("\n");
    const route = idx >= 0 ? out.slice(0, idx) : out;
    const items = (idx >= 0 ? out.slice(idx + 1) : "").trim().split("\n").filter(Boolean);
    return `${route} — ${items.length} evidence item(s)`;
  };

  pi.registerTool({
    name: "zm_recall",
    label: "ZM Recall",
    description:
      "Retrieve ranked evidence from zeromem memory over past conversations. Returns verbatim " +
      "turns with provenance (session, turn, speaker), never summaries. Uses zero tokens and zero " +
      "LLM calls. Use this before answering when a user reference their own history or previous sessions.",
    promptSnippet: "Recall relevant past turns from long-term memory",
    promptGuidelines: [
      "Use zm_recall when the user references earlier conversations, personal facts, or prior decisions — it costs no tokens and makes no LLM calls.",
      "zm_recall returns source turns with provenance; answer from that recorded evidence, not from guesses.",
    ],
    parameters: RECALL_PARAMS,
    async execute(_toolCallId, params) {
      const k = params.top_k ?? 5;
      const out = await zm(["query", String(params.query), ...(k ? ["-k", String(k)] : [])]);
      return {
        content: [{ type: "text", text: formatQueryOutput(out) }],
        details: { route: formatQueryOutput(out).split("\n")[0] },
      };
    },
  });

  pi.registerTool({
    name: "zm_ingest",
    label: "ZM Ingest",
    description:
      "Persist conversation turns into zeromem memory. Turns become retrievable via zm_recall. " +
      "Every operation is token-free and makes no LLM calls. Provide turn records directly, or a " +
      "session_id + speaker + text conversation from the current session you want remembered.",
    promptSnippet: "Save turns to long-term zeromem memory",
    promptGuidelines: [
      "Use zm_ingest after a turn established durable facts the user will want recalled later, or to persist context before a compaction.",
      "zm_ingest expects at least one { session_id, speaker, text } turn; timestamps are optional.",
    ],
    parameters: INGEST_PARAMS,
    async execute(_toolCallId, params) {
      const session = params.session_id ?? "pi";
      const now = Math.floor(Date.now() / 1000);
      const turns: Array<Record<string, unknown>> = params.turns.map((t, i) => ({
        session_id: t.session_id ?? session,
        speaker: t.speaker,
        text: t.text,
        ts: t.ts ?? now + i,
      }));
      const result = await ingestJsonl(turns);
      return {
        content: [{ type: "text", text: `Persisted ${turns.length} turn(s) to zeromem.\n${result}` }],
        details: { turns: turns.length },
      };
    },
  });

  pi.registerTool({
    name: "zm_stats",
    label: "ZM Stats",
    description: "Zeromem memory store counters: turns, sessions, entities, windows, episodes.",
    parameters: STATS_PARAMS,
    async execute() {
      const out = await zm(["stats"]);
      return { content: [{ type: "text", text: out.trim() }], details: {} };
    },
  });

  pi.registerCommand("zm", {
    description: "Interact with zeromem memory: /zm query <text> | /zm stats",
    handler: async (args, ctx) => {
      const [cmd, ...rest] = args.trim().split(/\s+/);
      try {
        if (cmd === "query") {
          const text = rest.join(" ");
          if (!text) {
            ctx.ui.notify("Usage: /zm query <text>", "warning");
            return;
          }
          const out = await zm(["query", text]);
          ctx.ui.notify(summarizeEvidence(out), "info");
        } else if (cmd === "stats") {
          const out = await zm(["stats"]);
          ctx.ui.notify(out.trim(), "info");
        } else {
          ctx.ui.notify("Usage: /zm query <text> | /zm stats", "warning");
        }
      } catch (error) {
        ctx.ui.notify(error instanceof Error ? error.message : String(error), "error");
      }
    },
  });

  pi.on("session_start", (_event, ctx) => {
    ctx.ui.setStatus("zeromem", `zm: ${DB}`);
  });

  // Preserve pre-compaction context to zeromem BEFORE pi summarizes and drops
  // it. This is the zero-token safety net: once messages leave the live
  // context window they remain retrievable via zm_recall from disk.
  if (PRESERVE_ON_COMPACT) {
    pi.on("session_before_compact", async (event, ctx) => {
      const { preparation } = event;
      const sessionId = ctx.sessionManager.getSessionFile() ??
        ctx.sessionManager.getSessionId?.() ??
        "pi";

      // On repeated auto-compactions pi recomputes the span from the previous
      // kept boundary. Only ingest messages not already covered so the store
      // doesn't accrue duplicated turns.
      if (persistedUpToSession === sessionId && persistedPrefixes.has(preparation.firstKeptEntryId)) {
        return; // this span was already persisted
      }
      persistedUpToSession = sessionId;
      persistedPrefixes.add(preparation.firstKeptEntryId);

      const baseTs = Math.floor(Date.now() / 1000) - (preparation.messagesToSummarize?.length ?? 0);
      const turns: Array<Record<string, unknown>> = [];
      for (const msg of preparation.messagesToSummarize ?? []) {
        turns.push(...messageToTurns(msg, sessionId, baseTs));
      }
      for (const msg of preparation.turnPrefixMessages ?? []) {
        turns.push(...messageToTurns(msg, sessionId, baseTs));
      }

      if (turns.length === 0) return;

      const result = await persistTurns(turns);
      if (result) {
        ctx.ui.setStatus("zeromem", `zm: ${DB} · +${turns.length} pre-compact`);
        ctx.ui.notify(
          `zeromem: preserved ${turns.length} turn(s) before compaction (${event.reason})`,
          "info",
        );
      }
    });
  }
}
