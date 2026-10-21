# Two-minute demo recording plan

Use your own recorded voice for the audio input. Keep the final video under two minutes and show the transcript, answer, and generated audio.

1. **0:00–0:15 — Introduce the agent.** Show the terminal and state that the caller is speaking to ZentroMart's AI assistant.
2. **0:15–0:55 — Grounded answer and follow-up.** Ask one question answerable from a policy; show the retrieved section ID. Ask a short follow-up and show that the answer uses the same context.
3. **0:55–1:30 — Verified order lookup.** Provide an order ID, show that the agent asks for either phone last four digits or delivery pincode, then provide a valid synthetic proof from the dataset. Do not show the proof in saved logs or the video overlay.
4. **1:30–1:55 — Handover or unknown answer.** Use a practice scenario that requires escalation or an unsupported request. Show the ticket tool result; claim a ticket only when `created=True`.
5. **1:55–2:00 — End.** Show the exact batch command briefly.

Select the scenario only after reviewing the corresponding practice row and current policy. Avoid exposing verification values, `.env`, API keys, or unrelated dataset rows on screen.
