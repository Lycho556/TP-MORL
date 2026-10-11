# P2 formal transfer test — summary

Runs: `results_p2_formal/test`  |  fields: field1, field2, field3, field4, field5, field6, field7, field8, field9, field10  |  seeds: [0, 1, 2, 3, 4]

| Rule | Mean ratio |
|---|---|
| persistence | 0.7685 |
| rh_announced | 0.8706 |
| trajectory | 0.9004 |
| rh_full | 0.9629 |
| **value-based policy** | **0.8730** |

- Difference to ranking on announced plans (mean over fields of seed-mean minus announced): **+0.0024**, 95% t-interval over 10 fields [-0.0064, +0.0112]
- Field--seed pairs above announced: 32/50; fields whose seed-mean is above announced: 6/10
- Pre-registered wording: **slightly above, not distinguishable**

| Field | Policy (seed mean) | Announced | Difference | Trajectory-informed |
|---|---|---|---|---|
| field1 | 0.8897 | 0.8869 | +0.0028 | 0.9164 |
| field2 | 0.8394 | 0.8550 | -0.0156 | 0.9000 |
| field3 | 0.8677 | 0.8566 | +0.0111 | 0.8711 |
| field4 | 0.8582 | 0.8597 | -0.0015 | 0.8601 |
| field5 | 0.8898 | 0.8722 | +0.0175 | 0.9228 |
| field6 | 0.8606 | 0.8777 | -0.0171 | 0.8945 |
| field7 | 0.8477 | 0.8363 | +0.0113 | 0.8777 |
| field8 | 0.8854 | 0.8832 | +0.0022 | 0.9160 |
| field9 | 0.9098 | 0.9133 | -0.0035 | 0.9356 |
| field10 | 0.8821 | 0.8653 | +0.0168 | 0.9100 |
