# Offline CAN Bus / Embedded Firmware Q&A Assistant

A fully **offline** retrieval-augmented (RAG) Q&A assistant that answers CAN bus / embedded
firmware questions from a small local document corpus (protocol notes, MCU peripheral
references, transceiver behavior), running entirely on-device via **Microsoft Foundry Local**.
No internet is used in the query path.

The assistant retrieves the most relevant document chunks for a question, then a local LLM
answers **grounded in those chunks only** — it cites the source document and refuses when the
answer is not in the corpus.

---

## How it works

```
user question ──► answer_query(question)
                        │
      ┌─────────────────┼────────────────────┐
      ▼                 ▼                     ▼
 embed query      SQLite: load all       build grounded prompt
 (NVIDIA/CUDA)    chunk embeddings       (system + context + Q)
      │                 │                     │
      └──► cosine similarity, top-K=3 ────────┤
                                              ▼
                              chat (Intel iGPU / OpenVINO)
                                    phi-4-mini
                                              │
                                              ▼
                                 answer + cited source(s)
```

Every answer falls into one of **three tiers**, decided by the retrieval scores:

1. **Grounded** — a retrieved chunk clears `MIN_SCORE`, so the model answers from that chunk and
   cites the source, e.g. `[can_fd_basics.md]`. Unchanged from before.
2. **General knowledge** — no retrieved chunk covers the question, but the *top* retrieval score
   clears the stricter `DOMAIN_SCORE` floor — i.e. the question reads as clearly on-topic for CAN
   bus / embedded firmware, just not something the corpus discusses. The model may then answer
   from its own general engineering knowledge, prefixed with `[General knowledge — not from your
   documents]` and never cited. Disable this tier with `RAG_GENERAL_KNOWLEDGE=0` to restore the
   original strict grounded/refuse-only behavior. Enabling this tier does **not** loosen fidelity
   on covered questions: the prompt still requires the model to answer *only* from the context —
   preserving its exact terms, names, and numeric values, and citing the source — whenever the
   context actually has the answer. The general-knowledge fallback only ever triggers on a
   genuine context miss, never as a substitute for a grounded answer that's already available.
3. **Refusal** — nothing clears the floor, or the question is off-topic / genuinely unanswerable,
   so the model replies with the exact refusal string. The domain gate only ever *adds* a labeled
   fallback for on-topic gaps — it never lets the model fabricate an answer to something outside
   CAN bus / embedded firmware.

- **Interface:** CLI (`src/cli.py`) and Streamlit (`app_streamlit.py`).
- **Pipeline:** `src/pipeline.py` → `answer_query(question)` = retrieve + generate + cite.
- **Data:** `data/kb.sqlite` — chunk text + L2-normalized float32 embeddings (SQLite only, no
  external vector DB).
- **Inference:** the running Foundry Local server (OpenAI-compatible HTTP endpoint) — chat on
  the Intel iGPU (OpenVINO), embeddings on the NVIDIA GPU (CUDA).

### Model placement

| Role | Model ID (pinned, full variant) | Device / EP | Notes |
|---|---|---|---|
| Chat / generation | `phi-4-mini-instruct-openvino-gpu` | Intel iGPU / OpenVINO | Uses shared RAM, not the 4 GB NVIDIA VRAM. Chosen after testing: refuses out-of-context questions correctly, concise, fast. |
| Embedding | `qwen3-embedding-0.6b-cuda-gpu` | NVIDIA RTX 3050 Ti / CUDA | ~478 MB. |

The two models sit on **different processors**, so there is no 4 GB VRAM contention and no OOM
risk. Context is kept ≤ 4096 tokens and retrieval uses top-K = 3. Fallback models (if needed)
are listed in `src/config.py` and the build spec.

---

## Setup

### 1. Install Foundry Local + the SDK

Install [Microsoft Foundry Local](https://learn.microsoft.com/azure/foundry-local/) first,
then the Python dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` selects the Foundry SDK per platform automatically:

- **Windows (recommended, hardware-accelerated):** `foundry-local-sdk-winml`
- **macOS / Linux (cross-platform):** `foundry-local-sdk`

### 2. Download the models (one time)

The two models are referenced by their full variant IDs so device placement is deterministic.
Inspect the catalog and cache state with:

```bash
foundry model list --variants
```

If they are not already cached:

```powershell
foundry model download phi-4-mini-instruct-openvino-gpu
foundry model download qwen3-embedding-0.6b        # cached as ...-cuda-gpu
```

After this one-time download, **nothing in the query path touches the network.**

### 3. Start the Foundry Local server

The app talks to the running Foundry server over its OpenAI-compatible HTTP endpoint, so start
it first (from a shell where the `foundry` CLI is on PATH — the **same user context** you'll
run Python in; don't mix an elevated daemon with a normal-user app):

```bash
foundry server start
foundry server status          # note the "Web URLs http://127.0.0.1:<port>"
```

The endpoint is auto-discovered from `foundry server status`. If the `foundry` CLI isn't on
PATH for your Python process, set it explicitly instead:

```powershell
setx FOUNDRY_LOCAL_ENDPOINT http://127.0.0.1:54163   # use your actual port
```

Models are **loaded into the server on demand** — the first ingest/query triggers
`foundry model load <id>` automatically, so no manual load step is normally needed.

### 4. Build the knowledge base

```bash
python -m src.ingest            # read data/raw/ → chunk → embed → data/kb.sqlite
python -m src.ingest --debug    # also print a sample of parsed chunks
```

This prints the total chunk count when finished.

---

## Usage

### CLI

```bash
python -m src.cli               # ask questions in a loop (remembers the conversation)
python -m src.cli --debug       # also show retrieved chunks + similarity scores
python -m src.cli --mode explain  # start in "explain" mode (default: short)
python -m src.cli --no-stream   # print each answer at once instead of streaming it
```

Answers **stream token-by-token** by default, so a long/slow reply appears as it's generated
instead of after a blank wait (disable with `--no-stream` or `RAG_STREAM=0`).

Greetings ("hi", "hey", "howdy", ...), "how are you", thanks, and "what can you do?" get a
friendly, **English-only** reply instead of a refusal — matched exactly (never as a substring),
so real questions are never intercepted and these never touch the grounded pipeline or the
conversation memory.

The CLI **remembers the conversation**, so follow-ups work ("explain that", "what about at
250 kbps?" — the previous question is folded into retrieval when a follow-up is elliptical).
In-session commands: `:short` / `:explain` to switch answer mode, `:general on` / `:general off`
(bare `:general` flips it) to toggle the general-knowledge fallback tier live for the rest of the
session — no restart or env var needed, `:reset` (`:clear`) to forget the conversation,
`:examples` (`:ex`) to print the sample questions from `EXAMPLE_QUESTIONS`, `:help` for the list.
The startup line reports the tier's initial state (from `RAG_GENERAL_KNOWLEDGE`), e.g.
`Ready — 41 chunks indexed. Mode: short. General knowledge: on.` Type `quit` or submit an empty
line to exit. Example:

```
Q[short]> What is the maximum CAN bus length at 500 kbps?
About 100 meters at 500 kbit/s. [can_2_0_basics.md]
Sources: ['can_2_0_basics.md']
(2.3s · short · grounded · top match 0.71)

Q[short]> :explain
[mode: explain]

Q[explain]> How do I choose a watchdog timeout for a safety-critical ECU?
[General knowledge — not from your documents] ...
(i) General knowledge — not grounded in your documents.
Sources: []
(1.8s · explain · general · top match 0.31)

Q[explain]> What's the WiFi setup procedure?
I don't have that information in the provided documents.
Sources: []
(0.1s · explain · refusal · top match 0.09)
```

Each answer shows how long it took, the mode, the answer **kind** (`grounded` / `general` /
`refusal`), and the best retrieval score, so you can see when a match is weak (and why an
off-topic question was refused) or when an answer came from the general-knowledge tier instead of
the corpus. Output is colorized (green/amber/dim by kind, green/yellow/red by score) when stdout
is an interactive terminal — auto-disabled when piped/captured, or by setting `NO_COLOR`. See
**Answer modes** and **Tuning retrieval** below.

### Answer modes

Two response *styles*, orthogonal to the answer *tier* (grounded / general knowledge / refusal —
see **How it works**):

- **short** (default) — a direct, definite answer in one or two sentences.
- **explain** — a fuller answer that states the result then explains the relevant details from
  the context.

Switch with `--mode` at launch, `:short` / `:explain` in the CLI, or the sidebar radio in
Streamlit. Set the default with `RAG_MODE`. The exact-refusal contract holds in both modes.

The **general-knowledge tier** (see **How it works** above) applies in both modes too — `short`
still gives a direct sentence or two, `explain` still gives a fuller write-up — the only
difference from a grounded answer is the `[General knowledge — not from your documents]` prefix
and the absence of a citation.

### Streamlit

```bash
streamlit run app_streamlit.py
```

The UI is **themed**: `.streamlit/config.toml` sets a warm terracotta accent (`primaryColor =
"#C96442"`) and deliberately leaves `base` unset so Streamlit derives a full light/dark palette
around it and auto-follows the OS's light/dark setting rather than locking into one theme. Chat
turns render as rounded, theme-agnostic bubbles with user/assistant avatars (🧑‍💻 / 🔧), and
default Streamlit chrome (main menu, footer) is hidden for a cleaner look. All of it is inline CSS
using the system font stack — no external fonts or other resources are loaded, so the app stays
fully offline.

The main pane is a **chat with memory** — ask a question, then follow up ("explain that")
and it uses the conversation so far. Before the first question, it shows a row of clickable
**example-question chips** (from `config.EXAMPLE_QUESTIONS`) so you can try the assistant without
typing. Answers **stream in live**. Each answer shows its sources, mode, latency, and an expander
with the retrieved chunks, each annotated with its similarity score and a progress bar. A
**general-knowledge** answer (see **How it works**) shows an amber warning banner in place of
sources, so it's never mistaken for a cited, corpus-grounded answer. The sidebar has the
**answer-mode** selector (short / explain), a **General-knowledge fallback** toggle (live on/off,
mirrors `RAG_GENERAL_KNOWLEDGE` but takes effect immediately without restarting the app), a
**Clear conversation** button that clears the active chat's messages, live **Documents** /
**Chunks** metrics for the indexed corpus, and a **"How it works"** expander summarizing the
three answer tiers. The Foundry client and KB are cached, so each query is fast.

**Conversations (sidebar).** The app manages multiple, independent chat histories: **New chat**
starts a fresh, empty conversation and makes it current; each past chat appears as its own
full-width button, labeled by its first question (or "New chat" until one is asked), with the
active chat highlighted in the accent color — click a different one to switch to it. A **Chat
name** box below the switcher lets you **rename** the active conversation; a renamed chat is
pinned, so the auto-generated title (derived from its first question) never overwrites the name
you chose. **Delete current chat** removes the active conversation, falling back to the most
recently created one left, or opening a new empty one if none remain. Every conversation —
questions, answers, sources, mode, latency, retrieved chunks, and title (including a rename) — is
saved to `data/chats.json` after each turn or edit, so **switching chats or restarting the app
never loses history**. This is separate from the document knowledge base: clearing, renaming, or
deleting a chat only affects that conversation's messages/title, not the indexed corpus in
`data/kb.sqlite`.

**Upload documents (sidebar).** Drop embedded-systems **PDF / Markdown / text** files into the
uploader and click *Add to knowledge base*. Each file's text is extracted (PDFs are converted
and reflowed automatically), chunked, embedded on the NVIDIA GPU, and added to the same SQLite
KB — then you can ask questions over it with citations. The sidebar also lists every indexed
document with its chunk count and a ✕ to remove it.

- Uploads **add on top** of the existing corpus (re-uploading the same filename replaces it,
  idempotently) and are stored in `data/kb.sqlite`, so they persist across restarts.
- The Foundry Local **server must be running** to upload (it embeds the new chunks).
- Note: a full `python -m src.ingest` rebuild repopulates the KB from `data/raw/` only, so it
  drops UI-uploaded documents. To keep a document permanently, also place its text file in
  `data/raw/`.
- Scanned/image-only PDFs (no text layer) can't be extracted — the UI warns you (OCR is out of
  scope).

---

## Corpus

Source documents live in `data/raw/` as `.md`/`.txt` files. The filename becomes the citation
label. The starter corpus covers:

| File | Topic |
|---|---|
| `can_2_0_basics.md` | CAN 2.0 frames, arbitration, 11-bit vs 29-bit IDs, bus length vs bit rate |
| `can_fd_basics.md` | CAN FD: 64-byte payload, BRS / dual bit rate, CRC changes |
| `can_error_handling.md` | Error detection, TEC/REC, error-active / error-passive / bus-off |
| `stm32_bxcan_bit_timing.md` | bxCAN/FDCAN bit timing: BRP, TSEG1/BS1, TSEG2/BS2, SJW, sample point |
| `j1939_overview.md` | J1939: 29-bit IDs, PGN/SPN, PDU1/PDU2, transport protocol, address claiming |
| `canopen_overview.md` | CANopen: Object Dictionary, PDO/SDO, NMT states, heartbeat |
| `can_transceiver_basics.md` | ISO 11898-2 transceiver: bus levels, 120 Ω termination, modes |

To grow the corpus, drop more text-heavy `.md`/`.txt` files into `data/raw/` and re-run
`python -m src.ingest`. Prefer clean prose sections over raw multi-column tables and figures —
chunking quality is the main retrieval-quality lever.

---

## Evaluation

`tests/eval_set.jsonl` holds answerable and unanswerable (refusal) questions, plus **multi-turn**
items. Each item is one of:
- `{"expected_behavior": "answer", "must_include_any": [...], "expect_source": "..."}`
- `{"expected_behavior": "refuse"}`
- multi-turn: add `"history_questions": [...]` — those are asked first (building real conversation
  memory) and then the `"question"` is graded, so follow-up retrieval + memory are exercised.

Run the full functional eval against the real pipeline (needs Foundry + the models):

```bash
python -m tests.run_eval                 # human-readable pass/fail + latency
python -m tests.run_eval --mode explain  # evaluate in "explain" mode (default: short)
python -m tests.run_eval --out results.md  # also write a Markdown results table
python -m tests.run_eval --json          # machine-readable results
```

Targets (per the build spec): answerable questions return the right fact **with a citation**;
out-of-corpus questions return the exact refusal string; latency ~1–3 s per answer on the
target laptop. Use `--out` to record a results table for the README/PR.

### Offline unit tests (no GPU / models needed)

The pure pipeline logic — chunking, vector serialization, cosine search, the SQLite
round-trip, prompt assembly, context fitting, and the refusal/citation logic — is covered by a
deterministic test suite that injects fake embed/chat functions. It runs anywhere:

```bash
pip install pytest
python -m pytest tests/ -q          # 41 tests
```

These run in CI too — `.github/workflows/ci.yml` runs the whole suite on every push and pull
request (only `numpy` + `pytest` needed, no GPU/runtime).

### Tuning retrieval

Retrieved chunks below `MIN_SCORE` (cosine similarity) are dropped before generation, so an
off-topic question reaches the model with no context and gets a clean, deterministic refusal
instead of a guess over weak matches. Watch the `top match` score printed with each answer (or
the per-chunk scores under `--debug`):

- If off-topic questions still get answered, **raise** the floor:
  `setx RAG_MIN_SCORE 0.3` (typical strict range 0.25–0.35).
- If legitimate questions get wrongly refused, **lower** it (e.g. `0.05`, or `0` to disable).

The default is a conservative `0.1`.

A second, stricter floor gates the **general-knowledge tier** (see **How it works**):
`DOMAIN_SCORE` (default `0.25`, override `RAG_DOMAIN_SCORE`) is the top-match score above which a
question is considered clearly on-topic enough to let the model answer from general engineering
knowledge when the corpus itself doesn't cover it. It's deliberately set above `MIN_SCORE` —
`MIN_SCORE` only decides whether a chunk is worth feeding to the model at all, while
`DOMAIN_SCORE` gates the stronger, uncited claim "this question is clearly about CAN bus /
embedded firmware." In practice an on-topic question's top match tends to land around ~0.5–0.7,
while an off-topic one stays below ~0.1, so the default sits comfortably in the gap.

- **Raise** `RAG_DOMAIN_SCORE` to make the general-knowledge fallback stricter (only the most
  confidently on-topic questions get it).
- **Lower** it to offer the fallback more readily.
- Set `RAG_GENERAL_KNOWLEDGE=0` to disable the tier entirely and restore the original strict
  grounded/refuse-only behavior — covered questions behave exactly as before either way, and
  off-topic questions are always refused regardless of this setting.

`RAG_GENERAL_KNOWLEDGE` sets the tier's default at process startup only — it can also be flipped
live, per session, without restarting: the CLI's `:general on` / `:general off` (bare `:general`
toggles) and the Streamlit sidebar's **General-knowledge fallback** toggle both override the env
var's value for that running session via `Pipeline.answer(..., general_enabled=...)`.

Also tunable: `RAG_CHAT_MODEL`, `RAG_EMBED_MODEL`, `RAG_MODE` (short/explain), `RAG_HISTORY_TURNS`,
`RAG_STREAM` (0 to disable streaming), `RAG_REQUEST_TIMEOUT`, `FOUNDRY_LOCAL_ENDPOINT`,
`RAG_DB_PATH`, `RAG_GENERAL_KNOWLEDGE` (0 to disable the general-knowledge tier), `RAG_DOMAIN_SCORE`
(default `0.25`; the on-topic gate for that tier), `RAG_CHATS_PATH` (default `data/chats.json`;
where the Streamlit UI persists its multi-conversation chat history).

---

## Troubleshooting (Windows)

- **`foundry server start` times out / `foundry server status` says "Not running" but shows a
  PID and Web URL.** The daemon is owned by a different session's integrity level. Run the
  Foundry server **and** the Python app from the **same terminal** — if you started Foundry
  from an elevated (Administrator) shell, run `python -m src.ingest` / `src.cli` there too. A
  per-user named pipe backs the daemon, so a non-elevated shell can't control a daemon started
  elevated (and vice-versa). If a stale daemon is stuck, clear it and restart:

  ```powershell
  foundry server stop
  Get-Process -Name foundrylocald -ErrorAction SilentlyContinue | Stop-Process -Force
  foundry server start
  foundry server status
  ```

- **`Model '...-openvino-gpu' not found via the Foundry Local SDK` / `Cached models the SDK
  reports: (none)`.** The in-process SDK core doesn't share the running daemon's model cache or
  execution-provider packs, so it can't see your cached GPU variants. That's why the app talks
  to the server over HTTP instead. Make sure `foundry server start` is running and the endpoint
  is discoverable (see "Could not find the endpoint" below).

- **`Could not find the Foundry Local server endpoint`.** Either the server isn't running
  (`foundry server start`) or the `foundry` CLI isn't on PATH for your Python process. Fix by
  setting the endpoint explicitly: `setx FOUNDRY_LOCAL_ENDPOINT http://127.0.0.1:<port>` using
  the port from `foundry server status`, then open a new terminal so the variable takes effect.

- **`Model '...' is not loaded`.** The server lists cached models but loads them on demand. The
  app auto-runs `foundry model load <id>` on the first request; if that can't run (CLI not on
  PATH), load them yourself once per server session:

  ```powershell
  foundry model load qwen3-embedding-0.6b-cuda-gpu
  foundry model load phi-4-mini-instruct-openvino-gpu
  ```

- **First chat hangs / freezes (tens of seconds to minutes), or times out.** OpenVINO compiles
  `phi-4-mini` on the iGPU on first load, which can be very slow. The CLI now warms the model up
  at startup (with a message) and every server request has a bounded timeout, so it errors with
  guidance instead of freezing forever. If it stays too slow, switch to the fast cached
  NVIDIA/TensorRT model — no code edit needed, just an env var (then reopen the terminal):

  ```powershell
  setx RAG_CHAT_MODEL phi-3.5-mini-instruct-trtrtx-gpu
  ```

  Related overrides: `RAG_EMBED_MODEL`, and `RAG_REQUEST_TIMEOUT` (seconds; default 300) to wait
  longer for a slow first compile.

## Design decisions

- **SQLite + brute-force cosine, no vector DB.** For a corpus of dozens–low-hundreds of
  chunks, a NumPy dot product over all rows is correct and fast. Embeddings are L2-normalized
  at ingestion so query-time cosine similarity is a plain dot product.
- **Grounded prompt with a strict refusal contract.** The system prompt forbids outside
  knowledge and mandates an exact refusal string when the context lacks the answer; low
  temperature favors faithful answers over guessing.
- **Deterministic device placement.** Full variant IDs are pinned in `src/config.py`, keeping
  the chat model on the Intel iGPU and the embedding model on the NVIDIA GPU with no VRAM
  contention.
- **Dependency injection for testability.** `ingest`, `retrieve`, `generate`, and `pipeline`
  accept the embed/chat callables, so the whole pipeline can be unit-tested without the
  Foundry runtime.

### Foundry Local integration note

The app talks to the running `foundry server` over its **OpenAI-compatible HTTP endpoint**
rather than the in-process SDK core (which, in practice, doesn't share the daemon's model cache
or execution-provider packs and so can't resolve the on-device GPU variants). This also avoids
Windows admin/normal-user cache-visibility problems, since it's just localhost HTTP:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:54163/v1", api_key="foundry-local")
answer = client.chat.completions.create(
    model="phi-4-mini-instruct-openvino-gpu", messages=messages
).choices[0].message.content
vecs = client.embeddings.create(
    model="qwen3-embedding-0.6b-cuda-gpu", input=texts
).data
```

The endpoint is auto-discovered from `foundry server status` (override with
`FOUNDRY_LOCAL_ENDPOINT`), and models are loaded on demand via `foundry model load`. All
Foundry calls are isolated in `src/foundry_client.py`; that's the only file to adjust if the
integration changes.

---

## Project structure

```
├─ README.md
├─ requirements.txt
├─ .streamlit/config.toml     # Streamlit theme: warm terracotta accent, auto light/dark
├─ .github/workflows/ci.yml   # runs the offline test suite on push / PR
├─ data/
│  ├─ raw/                 # source documents (.md / .txt)
│  ├─ kb.sqlite            # generated: chunks + embeddings
│  └─ chats.json           # generated: Streamlit multi-conversation history (gitignored)
├─ src/
│  ├─ config.py            # model IDs, endpoint, chunk params, top_k, ctx limit, prompt, paths
│  ├─ foundry_client.py    # Foundry Local server (HTTP): chat (iGPU) + embeddings (NVIDIA)
│  ├─ vectors.py           # L2 normalize + float32 blob (de)serialization
│  ├─ documents.py         # extract text from uploaded PDF / Markdown / text files
│  ├─ ingest.py            # chunk → embed → write/upsert SQLite (full build + incremental add)
│  ├─ retrieve.py          # embed query → cosine over stored vectors → top-K chunks
│  ├─ generate.py          # build grounded prompt + call chat → answer
│  ├─ pipeline.py          # answer_query(question): retrieve + generate + cite
│  ├─ conversations.py     # multi-conversation persistence + rename/title-pinning (data/chats.json)
│  ├─ smalltalk.py         # non-grounded greeting / "what can you do?" replies
│  └─ cli.py               # Phase 1 CLI
├─ app_streamlit.py        # Phase 2 web UI (Q&A + document upload)
└─ tests/
   ├─ eval_set.jsonl       # answerable + unanswerable test questions
   ├─ run_eval.py          # eval runner (real pipeline)
   ├─ test_pipeline.py     # offline unit tests (fake embed/chat)
   ├─ test_documents.py    # offline tests: extraction + incremental upload
   ├─ test_cli.py          # offline tests: CLI loop + streaming + client error mapping
   └─ test_smalltalk.py    # offline tests: greeting / meta shortcut
```

---

## Requirements recap

- Python 3.11+
- Microsoft Foundry Local + `foundry-local-sdk[-winml]`
- `numpy`, `openai`, `streamlit`
- The web UI is themed (`.streamlit/config.toml`) and fully offline — no external fonts or
  other remote resources.
- Target hardware for the reference setup: Windows, NVIDIA RTX 3050 Ti Laptop (4 GB VRAM),
  Intel i5-12500H (Iris Xe iGPU), 16 GB RAM. Runs on other machines with compatible Foundry
  Local model variants.
