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
python -m src.cli               # ask questions in a loop
python -m src.cli --debug       # also show retrieved chunks + similarity scores
```

Type `quit` or submit an empty line to exit. Example session:

```
Q> What is the maximum CAN bus length at 500 kbps?
About 100 meters at 500 kbit/s. [can_2_0_basics.md]
Sources: ['can_2_0_basics.md']

Q> What's the WiFi setup procedure?
I don't have that information in the provided documents.
Sources: []
```

### Streamlit

```bash
streamlit run app_streamlit.py
```

A text box, a submit button, the grounded answer with its sources, and an expander showing the
retrieved chunks + scores. The Foundry client and KB are cached, so each query is fast.

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

`tests/eval_set.jsonl` holds answerable and unanswerable (refusal) questions. Each item is
either `{"expected_behavior": "answer", "must_include_any": [...]}` or
`{"expected_behavior": "refuse"}`.

Run the full functional eval against the real pipeline (needs Foundry + the models):

```bash
python -m tests.run_eval          # human-readable pass/fail + latency
python -m tests.run_eval --json   # machine-readable results
```

Targets (per the build spec): answerable questions return the right fact **with a citation**;
out-of-corpus questions return the exact refusal string; latency ~1–3 s per answer on the
target laptop.

### Offline unit tests (no GPU / models needed)

The pure pipeline logic — chunking, vector serialization, cosine search, the SQLite
round-trip, prompt assembly, context fitting, and the refusal/citation logic — is covered by a
deterministic test suite that injects fake embed/chat functions. It runs anywhere:

```bash
pip install pytest
python -m pytest tests/test_pipeline.py -q
```

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

- **First chat is very slow (tens of seconds or more).** OpenVINO compiles the model on first
  load. If `phi-4-mini` on the iGPU stays too slow for you, switch `CHAT_MODEL_ID` in
  `src/config.py` to the cached NVIDIA/TensorRT fallback `phi-3.5-mini-instruct-trtrtx-gpu`
  (fast; rely on the strict system prompt for refusals).

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
├─ data/
│  ├─ raw/                 # source documents (.md / .txt)
│  └─ kb.sqlite            # generated: chunks + embeddings
├─ src/
│  ├─ config.py            # model IDs, chunk params, top_k, ctx limit, prompt, paths
│  ├─ foundry_client.py    # Foundry Local: chat client (iGPU) + embed client (NVIDIA)
│  ├─ vectors.py           # L2 normalize + float32 blob (de)serialization
│  ├─ ingest.py            # read docs → chunk → embed → write SQLite
│  ├─ retrieve.py          # embed query → cosine over stored vectors → top-K chunks
│  ├─ generate.py          # build grounded prompt + call chat → answer
│  ├─ pipeline.py          # answer_query(question): retrieve + generate + cite
│  └─ cli.py               # Phase 1 CLI
├─ app_streamlit.py        # Phase 2 web UI
└─ tests/
   ├─ eval_set.jsonl       # answerable + unanswerable test questions
   ├─ run_eval.py          # eval runner (real pipeline)
   └─ test_pipeline.py     # offline unit tests (fake embed/chat)
```

---

## Requirements recap

- Python 3.11+
- Microsoft Foundry Local + `foundry-local-sdk[-winml]`
- `numpy`, `openai`, `streamlit`
- Target hardware for the reference setup: Windows, NVIDIA RTX 3050 Ti Laptop (4 GB VRAM),
  Intel i5-12500H (Iris Xe iGPU), 16 GB RAM. Runs on other machines with compatible Foundry
  Local model variants.
