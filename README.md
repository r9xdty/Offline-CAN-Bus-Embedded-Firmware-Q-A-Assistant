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
- **Inference:** Foundry Local — chat on the Intel iGPU (OpenVINO), embeddings on the NVIDIA
  GPU (CUDA).

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

### 3. Build the knowledge base

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

- **`Model '...-openvino-gpu' not found`, yet `foundry model list --variants` shows it cached.**
  The SDK's `get_model_variant()` / `list_models()` only expose the remote `*-generic-cpu`
  catalog; on-device GPU variants (OpenVINO / CUDA / TensorRT) are surfaced by
  `get_cached_models()` and each model's `.variants`. `src/foundry_client.py` searches all of
  those, so the pinned GPU IDs resolve. If you still see the error, its message now lists the
  cached IDs the SDK reports — update `CHAT_MODEL_ID` / `EMBED_MODEL_ID` in `src/config.py` to
  match.

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

### Foundry Local SDK note

This code targets the Foundry Local SDK v1.x native client API:

```python
from foundry_local_sdk import Configuration, FoundryLocalManager
from foundry_local_sdk.openai import ChatClientSettings

FoundryLocalManager.initialize(Configuration(app_name="rag-can-assistant"))
catalog = FoundryLocalManager.instance.catalog
model = catalog.get_model_variant("phi-4-mini-instruct-openvino-gpu")
model.download(); model.load()
answer = model.get_chat_client().complete_chat(messages).choices[0].message.content
```

All Foundry calls are isolated in `src/foundry_client.py`; if the SDK surface changes, that is
the only file to adjust.

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
