# Glimmer label eval (2026-10-31T12:49:56+01:00)

Earlier attempt: SGLang serving `RedHatAI/Muse-Glimmer-30B-INT4`. This is not the full labelling run.

| Config | F7 P / R / unc | F8 P / R / unc | F9 P / R / unc | Uncertain (all F1–F10) | Present (all) | s/call (mean) | calls/min |
|---|---|---|---|---|---|---|---|
| v2:none | 1.00 / 0.50 / 0.14 | 0.25 / 0.20 / 0.33 | 1.00 / 0.25 / 0.14 | 0.29 | 0.23 | 89.65 | 8.60 |

Clips per GT factor: {'F7': 7, 'F8': 9, 'F9': 7}. P = precision, R = recall (uncertain counts as a miss), unc = share of hinted clips labeled uncertain.
