# SWEET.TV TV Program Recommendation

This directory contains the reproducible implementation and evidence for the
LLM4Rec AVITO exam. It never reads user viewing behavior after week 30. The
weeks 31–42 schedule is used only for the eligible program set and static item
metadata.

## Reproduction

The implementation uses Python 3.12 and the standard library only.

```bash
cd exam_avito/repo
PYTHONPATH=. python3 -m unittest discover -s tests -v
PYTHONPATH=. python3 run_analysis.py
```

Raw Kaggle files must be placed in `../data/raw/` with their original names.
The first run creates a join cache under `../data/processed/`; both raw and
processed data are excluded from Git. A later run reuses the cache only when
the input sizes, modification times, join version, and viewing threshold match.

## Evaluation protocol

- Training period: weeks 11–20.
- Validation period: weeks 21–30.
- Metric: MAP@5 with denominator `min(5, number of relevant programs)`.
- Validation target: each user's five most frequently watched programs in the
  validation period, mirroring the competition target construction.
- Candidate pool: all nonzero `tv_show_id` values airing in the target schedule.
- Repeats: programs seen during training remain eligible for every method.
- Cold start: training interaction popularity, restricted to the target pool;
  target-schedule frequency and then program ID are deterministic tie-breakers.
- Viewing rule (our operational decision): a program is classified as watched
  when the union of the user's viewing intervals covers at least 80% of its
  scheduled duration on the same channel.
- `tv_show_id = 0`: removed before the join because it is unlabelled airtime.

The 0.80 threshold was fixed before validation and was not tuned. It aligns
with the Kaggle Data page's description of target construction: “Only TV shows
with high screen time (>= 0.8) were taken into account.” This quotation is
evidence about how the competition target was constructed, not a claim that
Kaggle defines a universal viewing rule.

The empirical overlap-fraction distribution for the 3,014,129 matched
user-airing pairs is: Q25 = 0.043, median = 0.195, Q75 = 0.656, Q80 = 0.809,
Q90 = 1.000, and Q95 = 1.000. Therefore, 0.80 is a selective high-coverage
cutoff just above the observed 75th percentile. It retains 611,760 matched
pairs, or 20.2964%, while excluding short and partial overlaps.

## Methods

1. Popularity baseline: watched-airing counts from the training period.
2. Content-based filtering: an IDF-weighted cosine model over category, genre,
   year, director, and actor features. Feature set and history weighting are
   selected on validation.
3. Association rules: co-view rules mined from fixed-length time baskets.
   Basket length, minimum pair support, and score type are selected on
   validation. A fixed recent-history limit and deterministic recency decay are
   documented in the evidence.
4. Hybrid: reciprocal-rank fusion of the selected content and association
   variants. Its weight is selected on validation.

All methods use the same metric, users, candidate pool, repeat policy, and
fallback. The selected configurations are retrained on weeks 11–30 before test
recommendations are generated.

## Outputs

- `submissions/`: fully validated Kaggle CSV files for the baseline, both
  standalone methods, and the hybrid. The student manually uploaded
  `submission_association.csv` as the designated final submission. The other
  files remain local and will not be uploaded.
- `evidence/`: machine-readable inputs for the report, including validation and
  tuning results, business metrics, runtime and memory, user examples, hashes,
  submission checks, and a manual AP@5 worksheet.
- `tests/`: unit tests for time parsing, interval-union coverage, MAP@5,
  candidate-bounded ranking, and submission validation.

Do not cite a number in the report unless it appears in `evidence/` or is
recomputed from those saved artifacts. The manual AP@5 worksheet must be
checked independently by the student before it is described as verified.
