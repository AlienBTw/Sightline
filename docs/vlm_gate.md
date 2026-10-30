# VLM evidence gate

Evidence only — never the final hazard verdict. Fields become HistGB features (`vlm_*`).

The gate calls any **OpenAI-compatible** chat endpoint with an image. The local default is **Muse Glimmer 30B through Ollama**:

```
VLM_BASE_URL=http://localhost:11434/v1
VLM_MODEL=muse-glimmer:30b
```

The base URL includes `/v1`; the gate posts to `{VLM_BASE_URL}/chat/completions`. Set `VLM_API_KEY` only for hosted endpoints (e.g. the Meta Model API). In the Meta build, Muse Spark + SAM 3.1 via the Meta Model API are the escalation path.

## When it runs

Configured in `config/thresholds.yaml`:

- `always_gate`: enable `fire_heat` / `electrical` when `VLM_BASE_URL` is set (see `config/thresholds.yaml`)
- Mid-band probability for other families (`gate_probability_low` … `high`)

## Fail-loud

| Condition | Error code |
|---|---|
| Empty `VLM_BASE_URL` on gated candidate | `vlm_not_configured` |
| Empty `VLM_MODEL` | `vlm_model_missing` |
| HTTP failure | `vlm_http_error` |
| Invalid / incomplete JSON | `vlm_invalid_json` / `vlm_schema_incomplete` |

Smoke: `python scripts/smoke_vlm.py` (requires a running endpoint, e.g. `ollama serve`).

Offline replay demos that cannot reach the VLM should avoid gated families or leave those candidates unlabeled until the gate is up — the API will not invent empty VLM fields.
