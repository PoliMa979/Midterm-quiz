# Manual AP@5 verification

- Method: `association`
- User: `36938861791891792`
- Predictions: `2400480 240081 12001682 2400467 20088`
- Relevant set: `20088 2400480 6700482 12002856 12002896`

| Rank | Program | Relevant? | Hits so far | Precision | Contribution |
| ---: | ---: | :---: | ---: | ---: | ---: |
| 1 | 2400480 | yes | 1 | 1.000000 | 1.000000 |
| 2 | 240081 | no | 1 | 0.500000 | 0.000000 |
| 3 | 12001682 | no | 1 | 0.333333 | 0.000000 |
| 4 | 2400467 | no | 1 | 0.250000 | 0.000000 |
| 5 | 20088 | yes | 2 | 0.400000 | 0.400000 |

Numerator = `1.400000000000`.
Denominator = `min(5, 5) = 5`.
AP@5 = `1.400000000000 / 5 = 0.280000000000`.

## Independent student verification

Verified by the student on 2026-10-25. The student confirmed hits at ranks 1
and 5 and independently calculated:

`(1/1 + 2/5) / min(5, 5) = (1.0 + 0.4) / 5 = 0.28`.

The manual result matches the code result.
