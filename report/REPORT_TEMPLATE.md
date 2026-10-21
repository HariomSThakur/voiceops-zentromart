# ZentroMart VoiceOps 2026 — short report draft

Keep the final report within four pages. Replace every bracketed prompt with the results from your own runs. Do not claim a score that was not measured.

## 1. What we built

- **Step 1 — Prompt-only:** [model, system prompt approach, baseline score]
- **Step 2 — RAG:** [chunk strategy, embedding model, index, top-k, current-policy filtering]
- **Step 3 — Conversation and tools:** [session memory, order verification, retry limit, ticket handover]
- **Step 4 — Voice:** [Gnani STT language/model, Gnani TTS voice/model, audio batch command]

## 2. Evaluation

| Measure | Prompt-only | RAG | Notes |
|---|---:|---:|---|
| Questions completed | [ ] | [ ] | [ ] |
| Required facts satisfied | [ ] | [ ] | Heuristic checks plus manual review |
| Forbidden claims avoided | [ ] | [ ] | [ ] |
| Evidence section hit rate | — | [ ] | Compare retrieved IDs with `gold_evidence` |
| Conversation/tool cases passed | — | [ ] | [ ] |
| Audio files completed end to end | — | [ ] | [ ] |

Describe how the score was computed, including any manual judgment. The local script's checks are not the organizers' score.

## 3. What failed and what we learned

- [A retrieval miss or confusing policy conflict and its cause]
- [A conversation, verification, or handover failure]
- [A speech/transcription issue, if observed]

## 4. What we would improve next

[One or two changes supported by the evaluation results.]

## 5. Reproduction

- Public repository: [URL]
- Setup: see `README.md`
- Index command: `python run.py index`
- Required audio batch command: `python run.py batch --audio-dir <folder> --output-dir sample_outputs/batch`
- Gnani STT/TTS: [language, model and voice]
