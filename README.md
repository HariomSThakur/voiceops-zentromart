# ZentroMart VoiceOps 2026 capstone

Python implementation for the VoiceOps brief: document-grounded support, conversation memory, verified order lookup, human handover tickets, Gnani STT/TTS, and a reproducible audio-folder runner.

The supplied dataset is expected at `data/dataset/Capstone_Project_Dataset_FTTT_2026/`. Its working copy is kept locally and is **excluded from the project bundle and submission ZIP** by `make_submission.py`. If setting up elsewhere, extract the ZIP to that path.

## 1. Set up

Use Python 3.10 or newer. From this folder:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Before building, the dataset's included offline validator can be run with:

```powershell
python data/dataset/Capstone_Project_Dataset_FTTT_2026/validate_dataset.py
```

Edit `.env` and set:

- `LLM_API_KEY`, `LLM_MODEL`: a hosted or OpenAI-compatible chat model.
- `EMBEDDING_MODEL` and, if needed, `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL`: an embeddings endpoint. The selected endpoint must implement the OpenAI-compatible embeddings API.
- `GNANI_API_KEY`: the individual Gnani key for both speech services.

For a compatible local model server, set `LLM_BASE_URL` and a non-empty placeholder `LLM_API_KEY`. If its embedding endpoint differs, configure `EMBEDDING_BASE_URL` and `EMBEDDING_API_KEY` separately. Never commit `.env` or put keys in source code.

## 2. Build the knowledge index

```powershell
python run.py index
```

This embeds current Markdown sections plus product catalog and pincode rows into a local SQLite vector index at `.cache/voiceops_index.sqlite3`. The builder preserves section IDs, document type, status, and `effective_from`; superseded and not-yet-effective documents are excluded for the fixed reference date, 20 October 2026. Support logs are not indexed as policy evidence.

## 3. Run text conversations

```powershell
python run.py ask --text "How long do wallet refunds take?" --session demo
python run.py ask --text "And what if it is late?" --session demo
```

Keep the same `--session` value for follow-up messages. The agent keeps verified order context only in that in-memory session. To compare the prompting-only baseline with retrieval:

```powershell
python run.py ask --mode prompt-only --text "How long do wallet refunds take?"
```

## 4. Evaluate on the supplied practice set

```powershell
python run.py evaluate
```

This runs all single questions in prompting-only and RAG modes, then all multi-turn conversations in RAG mode. It writes `sample_outputs/evaluation.jsonl` and `sample_outputs/evaluation_summary.json`. Add `--include-asr` to also run the `asr_text` variants. The run makes model and embedding API calls. The built-in checks compare required/forbidden facts and retrieved evidence; review decisions, multilingual wording, and policy nuance manually before reporting a score. These checks are a working aid, not the organizer's scoring system.
Single-question evaluation calls run in parallel by default (`--workers 4`); use `--workers 1` for fully sequential runs.

Each evaluation conversation gets its own session, and turns are processed in order. Verification values in user messages are redacted from saved inputs and are never passed to the LLM. The helper tool returns order details only after its own verification succeeds.

## 5. Run the required audio-folder command

The single batch command is:

```powershell
python run.py batch --audio-dir .\audio_in --output-dir .\sample_outputs\batch
```

Accepted input extensions are WAV, MP3, FLAC, M4A, OGG, and AMR. Files in each subfolder are treated as successive turns in one conversation; files directly in `audio_in` share one conversation. Keep separate calls in separate subfolders. Output includes one JSON record and one Gnani-generated WAV reply per file, plus `batch_results.jsonl` and `summary.json`. JSON records include the redacted transcript, reply text/audio path, retrieved section IDs, and sanitized tool action status. No other STT/TTS provider or fallback is used.

The archive has no audio clips. Record your own voice for the demo and practice audio flow; do not record another person without written consent. Gnani STT defaults to `en-IN`; override with `--language hi-IN`, `gu-IN`, etc. TTS defaults are `Kaveri`, `timbre-v2.5`, and `en-IN`, configurable in `.env`.

## Implementation notes

- IDs, phone values, pincodes, and SKUs remain strings in the supplied helper.
- Order details are never shown before successful helper verification. The agent asks for one proof value at a time and permits one retry. It never logs or repeats the value.
- Tickets are created through the supplied offline helper. The response only confirms creation when it receives `created=True`.
- Calendar calculations use `AS_OF_DATE=2026-10-20`; Sundays and `holidays.csv` are excluded from business-day counts. Return windows remain calendar days.
- The fixed dataset date is intentional; do not replace it with the machine's current date.
- `data/dataset/`, `.cache/`, `.env`, and ticket logs are excluded from the submission package. Do not include the dataset ZIP or an extracted copy in the ZIP you submit.

## Submission materials still requiring your run/results

`report/REPORT_TEMPLATE.md` is the report draft and `report/DEMO_SCRIPT.md` is a two-minute recording plan. Fill them with real evaluation results and record the demo before packaging. The project does not invent scores or claim a demo was recorded.

When `report.pdf` and `demo.mp4` are ready, create the requested ZIP:

```powershell
python make_submission.py --team-name your_team_name
```

The packager checks the required files and excludes credentials, environments, caches, generated ticket logs, and the dataset.
Add the public repository URL to the README before submission after you publish the project repository.

## References

- Gnani official Python SDK: <https://github.com/Gnani-AI-Mintlify/Gnani-Vachana>
- Gnani API documentation: <https://docs.gnani.ai>
