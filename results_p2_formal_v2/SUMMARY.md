# P2 formal transfer test — summary

Runs: `results_p2_formal_v2/test`  |  fields: field1, field2, field3, field4, field5, field6, field7, field8, field9, field10  |  seeds: [0, 1, 2, 3, 4]

| Rule | Mean ratio |
|---|---|
| persistence | 0.7685 |
| rh_announced | 0.8706 |
| trajectory | 0.9004 |
| rh_full | 0.9629 |
| **value-based policy** | **0.8649** |

- Difference to ranking on announced plans (mean over fields of seed-mean minus announced): **-0.0058**, 95% t-interval over 10 fields [-0.0173, +0.0058]
- Field--seed pairs above announced: 21/50; fields whose seed-mean is above announced: 4/10
- Pre-registered wording: **does not beat**

| Field | Policy (seed mean) | Announced | Difference | Trajectory-informed |
|---|---|---|---|---|
| field1 | 0.8833 | 0.8869 | -0.0036 | 0.9164 |
| field2 | 0.8294 | 0.8550 | -0.0256 | 0.9000 |
| field3 | 0.8651 | 0.8566 | +0.0085 | 0.8711 |
| field4 | 0.8493 | 0.8597 | -0.0104 | 0.8601 |
| field5 | 0.8884 | 0.8722 | +0.0162 | 0.9228 |
| field6 | 0.8388 | 0.8777 | -0.0389 | 0.8945 |
| field7 | 0.8349 | 0.8363 | -0.0014 | 0.8777 |
| field8 | 0.8832 | 0.8832 | +0.0000 | 0.9160 |
| field9 | 0.9070 | 0.9133 | -0.0063 | 0.9356 |
| field10 | 0.8694 | 0.8653 | +0.0040 | 0.9100 |
