"""Unit tests for the deterministic recommendation pipeline."""

from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from src.recsys import (
    Airing,
    K,
    PeriodSchedule,
    ScheduleBundle,
    ap_at_k,
    build_targets,
    business_metrics,
    fill_ranking,
    join_views_to_schedule,
    parse_iso_timestamp,
    parse_schedule_timestamp,
    validate_submission,
    write_submission,
)


class TimeParsingTest(unittest.TestCase):
    def test_two_formats_share_axis(self) -> None:
        iso = parse_iso_timestamp("2020-03-09 07:40:00")
        schedule, day = parse_schedule_timestamp("09.03.2020 07:40:00")
        self.assertEqual(iso, schedule)
        self.assertEqual(schedule // 86400, day)


class MetricTest(unittest.TestCase):
    def test_ap_at_5_known_example(self) -> None:
        prediction = [1, 2, 9, 3, 8]
        relevant = {1, 2, 3}
        expected = (1.0 + 1.0 + 3 / 4) / 3
        self.assertAlmostEqual(ap_at_k(prediction, relevant), expected)

    def test_duplicate_prediction_does_not_create_a_second_hit(self) -> None:
        self.assertAlmostEqual(ap_at_k([1, 1, 2, 3, 4], {1, 2}), (1.0 + 2 / 3) / 2)

    def test_targets_use_top_five_counts(self) -> None:
        interactions = {7: Counter({10: 9, 11: 8, 12: 7, 13: 6, 14: 5, 15: 4})}
        self.assertEqual(build_targets(interactions)[7], {10, 11, 12, 13, 14})


class RankingTest(unittest.TestCase):
    def test_fill_is_unique_and_candidate_bounded(self) -> None:
        result = fill_ranking([9, 9, 8, 100], [7, 6, 5], {5, 6, 7, 8, 9}, K)
        self.assertEqual(result, [9, 8, 7, 6, 5])

    def test_submission_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.csv"
            rankings = {1: [10, 11, 12, 13, 14], 2: [11, 12, 13, 14, 15]}
            write_submission(path, [1, 2], rankings)
            result = validate_submission(path, [1, 2], set(range(10, 16)))
            self.assertTrue(result["valid"])

    def test_novelty_uses_training_user_population(self) -> None:
        rankings = {1: [10, 11, 12, 13, 14]}
        histories = {user: Counter({10: 1}) for user in range(1, 101)}
        metrics = business_metrics(rankings, [1], set(range(10, 15)), histories)
        self.assertGreaterEqual(metrics["mean_self_information_bits"], 0.0)


class JoinTest(unittest.TestCase):
    def test_union_coverage_and_threshold(self) -> None:
        # Two overlapping view rows cover [0, 50] and [40, 80]. Union is 80%.
        from datetime import date

        day = date(2020, 3, 9).toordinal()
        base = day * 86400
        airing = Airing(base, base + 100, 42, "train", day, 0)
        bundle = ScheduleBundle(
            periods={"train": PeriodSchedule(candidates={42})},
            by_channel={3: [airing]},
            starts_by_channel={3: [airing.start]},
            prefix_max_end_by_channel={3: [airing.end]},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "views.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["user_id", "vsetv_id", "start_time", "stop_time", "duraton"])
                writer.writerow([1, 3, "2020-03-09 00:00:00", "2020-03-09 00:00:50", 50])
                writer.writerow([1, 3, "2020-03-09 00:00:40", "2020-03-09 00:01:20", 40])
            result = join_views_to_schedule(path, bundle, threshold=0.8)
        self.assertEqual(result.interactions["train"][1][42], 1)
        self.assertEqual(result.watched_user_airings, 1)
        self.assertEqual(result.matched_user_airings, 1)


if __name__ == "__main__":
    unittest.main()
