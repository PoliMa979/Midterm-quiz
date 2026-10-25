"""Deterministic utilities for the SWEET.TV recommendation exam.

The module intentionally depends only on the Python standard library so the
full experiment can be reproduced without downloading model packages.
"""

from __future__ import annotations

import bisect
import csv
import hashlib
import heapq
import json
import math
import os
import platform
import random
import re
import resource
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence


K = 5
DEFAULT_SEED = 20261025
WATCH_THRESHOLD = 0.80


def parse_iso_timestamp(value: str, day_cache: dict[str, int] | None = None) -> int:
    """Parse YYYY-MM-DD HH:MM:SS into seconds on a timezone-free axis."""
    cache = day_cache if day_cache is not None else {}
    date_part = value[:10]
    day = cache.get(date_part)
    if day is None:
        year = int(value[0:4])
        month = int(value[5:7])
        day_of_month = int(value[8:10])
        from datetime import date

        day = date(year, month, day_of_month).toordinal()
        cache[date_part] = day
    return day * 86400 + int(value[11:13]) * 3600 + int(value[14:16]) * 60 + int(value[17:19])


def parse_schedule_timestamp(value: str, day_cache: dict[str, int] | None = None) -> tuple[int, int]:
    """Parse DD.MM.YYYY HH:MM:SS and return (seconds, ordinal day)."""
    cache = day_cache if day_cache is not None else {}
    date_part = value[:10]
    day = cache.get(date_part)
    if day is None:
        year = int(value[6:10])
        month = int(value[3:5])
        day_of_month = int(value[0:2])
        from datetime import date

        day = date(year, month, day_of_month).toordinal()
        cache[date_part] = day
    seconds = day * 86400 + int(value[11:13]) * 3600 + int(value[14:16]) * 60 + int(value[17:19])
    return seconds, day


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "").casefold().strip()
    return re.sub(r"\s+", " ", value)


def split_people(value: str, limit: int = 20) -> list[str]:
    if not value:
        return []
    parts = re.split(r"[,;|/]", value)
    return [part for part in (normalize_text(x) for x in parts) if part][:limit]


def item_features(row: Mapping[str, str], full: bool) -> set[str]:
    features: set[str] = set()
    category = normalize_text(row.get("tv_show_category", ""))
    if category:
        features.add(f"category={category}")
    for key in ("tv_show_genre_1", "tv_show_genre_2", "tv_show_genre_3"):
        genre = normalize_text(row.get(key, ""))
        if genre:
            features.add(f"genre={genre}")
    if not full:
        return features

    year_text = row.get("year_of_production", "") or ""
    match = re.search(r"(?:19|20)\d{2}", year_text)
    if match:
        year = int(match.group(0))
        features.add(f"year={year}")
        features.add(f"decade={year // 10 * 10}")
    for director in split_people(row.get("director", ""), limit=10):
        features.add(f"director={director}")
    for actor in split_people(row.get("actors", ""), limit=20):
        features.add(f"actor={actor}")
    return features


@dataclass(slots=True, frozen=True)
class Airing:
    start: int
    end: int
    item: int
    period: str
    day: int
    uid: int


@dataclass
class PeriodSchedule:
    candidates: set[int] = field(default_factory=set)
    schedule_frequency: Counter[int] = field(default_factory=Counter)
    features_basic: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    features_full: dict[int, set[str]] = field(default_factory=lambda: defaultdict(set))
    titles: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    categories: dict[int, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    durations: list[int] = field(default_factory=list)
    labelled_rows: int = 0
    zero_id_rows: int = 0
    invalid_duration_rows: int = 0


@dataclass
class ScheduleBundle:
    periods: dict[str, PeriodSchedule]
    by_channel: dict[int, list[Airing]]
    starts_by_channel: dict[int, list[int]]
    prefix_max_end_by_channel: dict[int, list[int]]


def load_schedules(period_paths: Mapping[str, Path], include_airings: bool = True) -> ScheduleBundle:
    """Load labelled schedule rows and aggregate static item metadata."""
    periods = {period: PeriodSchedule() for period in period_paths}
    by_channel: dict[int, list[Airing]] = defaultdict(list)
    uid = 0
    day_cache: dict[str, int] = {}

    for period, path in period_paths.items():
        info = periods[period]
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                item = int(row["tv_show_id"] or 0)
                if item == 0:
                    info.zero_id_rows += 1
                    continue
                try:
                    duration = int(float(row["duration"]))
                except (TypeError, ValueError):
                    info.invalid_duration_rows += 1
                    continue
                if duration <= 0:
                    info.invalid_duration_rows += 1
                    continue

                start, day = parse_schedule_timestamp(row["start_time"], day_cache)
                channel = int(row["channel_id"])
                info.labelled_rows += 1
                info.candidates.add(item)
                info.schedule_frequency[item] += 1
                info.durations.append(duration)
                info.features_basic[item].update(item_features(row, full=False))
                info.features_full[item].update(item_features(row, full=True))
                title = (row.get("tv_show_title") or "").strip()
                category = (row.get("tv_show_category") or "").strip()
                if title:
                    info.titles[item][title] += 1
                if category:
                    info.categories[item][category] += 1
                if include_airings:
                    by_channel[channel].append(
                        Airing(start, start + duration, item, period, day, uid)
                    )
                    uid += 1

    starts_by_channel: dict[int, list[int]] = {}
    prefix_max_end_by_channel: dict[int, list[int]] = {}
    for channel, airings in by_channel.items():
        airings.sort(key=lambda airing: (airing.start, airing.end, airing.uid))
        starts_by_channel[channel] = [airing.start for airing in airings]
        running = -1
        prefix: list[int] = []
        for airing in airings:
            running = max(running, airing.end)
            prefix.append(running)
        prefix_max_end_by_channel[channel] = prefix
    return ScheduleBundle(periods, dict(by_channel), starts_by_channel, prefix_max_end_by_channel)


@dataclass
class JoinResult:
    interactions: dict[str, dict[int, Counter[int]]]
    events: dict[str, dict[int, list[tuple[int, int]]]]
    overlap_histogram: list[int]
    matched_user_airings: int
    watched_user_airings: int
    view_rows: int
    skipped_view_rows: int
    sorted_violations: int
    runtime_seconds: float


def join_views_to_schedule(
    view_path: Path,
    schedule: ScheduleBundle,
    threshold: float = WATCH_THRESHOLD,
    progress_every: int = 1_000_000,
) -> JoinResult:
    """Join views to airings and classify watched airings by union coverage.

    The view log is required to be globally sorted by start_time. Active
    user-airing pairs are finalized when their airing ends, which keeps memory
    proportional to concurrent viewing rather than to the entire dataset.
    """
    started = time.perf_counter()
    interactions: dict[str, dict[int, Counter[int]]] = {
        period: defaultdict(Counter) for period in schedule.periods
    }
    events: dict[str, dict[int, list[tuple[int, int]]]] = {
        period: defaultdict(list) for period in schedule.periods
    }
    active: dict[tuple[int, int], list[int | Airing]] = {}
    expiry_heap: list[tuple[int, tuple[int, int]]] = []
    overlap_histogram = [0] * 1001
    matched_user_airings = 0
    watched_user_airings = 0
    day_cache: dict[str, int] = {}
    view_rows = 0
    skipped_view_rows = 0
    sorted_violations = 0
    previous_start = -1

    def finalize(key: tuple[int, int]) -> None:
        nonlocal matched_user_airings, watched_user_airings
        record = active.pop(key, None)
        if record is None:
            return
        airing = record[0]
        assert isinstance(airing, Airing)
        covered = int(record[1])
        duration = airing.end - airing.start
        fraction = min(1.0, covered / duration)
        overlap_histogram[min(1000, int(fraction * 1000 + 1e-12))] += 1
        matched_user_airings += 1
        if fraction + 1e-12 >= threshold:
            user = key[0]
            interactions[airing.period][user][airing.item] += 1
            events[airing.period][user].append((airing.day, airing.item))
            watched_user_airings += 1

    def expire(before_or_at: int) -> None:
        while expiry_heap and expiry_heap[0][0] <= before_or_at:
            _, key = heapq.heappop(expiry_heap)
            finalize(key)

    with view_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            view_rows += 1
            if progress_every and view_rows % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(
                    f"join progress: {view_rows:,} view rows, "
                    f"{watched_user_airings:,} watched airings, {elapsed:.1f}s",
                    flush=True,
                )
            try:
                start = parse_iso_timestamp(row["start_time"], day_cache)
                stop = parse_iso_timestamp(row["stop_time"], day_cache)
                user = int(row["user_id"])
                channel = int(row["vsetv_id"])
            except (KeyError, TypeError, ValueError):
                skipped_view_rows += 1
                continue
            if stop <= start:
                skipped_view_rows += 1
                continue
            if start < previous_start:
                sorted_violations += 1
            previous_start = max(previous_start, start)
            if sorted_violations:
                raise ValueError("View log is not globally sorted by start_time")
            expire(start)

            airings = schedule.by_channel.get(channel)
            if not airings:
                continue
            starts = schedule.starts_by_channel[channel]
            prefix_max = schedule.prefix_max_end_by_channel[channel]
            first = bisect.bisect_right(prefix_max, start)
            index = first
            while index < len(airings) and starts[index] < stop:
                airing = airings[index]
                overlap_start = max(start, airing.start)
                overlap_end = min(stop, airing.end)
                if overlap_end > overlap_start:
                    key = (user, airing.uid)
                    record = active.get(key)
                    if record is None:
                        active[key] = [airing, overlap_end - overlap_start, overlap_end]
                        heapq.heappush(expiry_heap, (airing.end, key))
                    else:
                        last_end = int(record[2])
                        if overlap_end > last_end:
                            record[1] = int(record[1]) + overlap_end - max(overlap_start, last_end)
                            record[2] = overlap_end
                index += 1

    expire(sys.maxsize)
    return JoinResult(
        interactions={period: dict(users) for period, users in interactions.items()},
        events={period: dict(users) for period, users in events.items()},
        overlap_histogram=overlap_histogram,
        matched_user_airings=matched_user_airings,
        watched_user_airings=watched_user_airings,
        view_rows=view_rows,
        skipped_view_rows=skipped_view_rows,
        sorted_violations=sorted_violations,
        runtime_seconds=time.perf_counter() - started,
    )


def merge_interactions(*parts: Mapping[int, Counter[int]]) -> dict[int, Counter[int]]:
    merged: dict[int, Counter[int]] = defaultdict(Counter)
    for part in parts:
        for user, counts in part.items():
            merged[user].update(counts)
    return dict(merged)


def merge_events(*parts: Mapping[int, Sequence[tuple[int, int]]]) -> dict[int, list[tuple[int, int]]]:
    merged: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for part in parts:
        for user, rows in part.items():
            merged[user].extend(rows)
    for rows in merged.values():
        rows.sort()
    return dict(merged)


def merge_feature_maps(*maps: Mapping[int, set[str]]) -> dict[int, set[str]]:
    result: dict[int, set[str]] = defaultdict(set)
    for mapping in maps:
        for item, features in mapping.items():
            result[item].update(features)
    return dict(result)


def aggregate_item_counts(interactions: Mapping[int, Counter[int]]) -> Counter[int]:
    total: Counter[int] = Counter()
    for counts in interactions.values():
        total.update(counts)
    return total


def build_targets(interactions: Mapping[int, Counter[int]], k: int = K) -> dict[int, set[int]]:
    """Mirror the competition target: top-k watched programs by airing count."""
    targets: dict[int, set[int]] = {}
    for user, counts in interactions.items():
        ranked = sorted(counts, key=lambda item: (-counts[item], item))[:k]
        if ranked:
            targets[user] = set(ranked)
    return targets


def ap_at_k(predictions: Sequence[int], relevant: set[int], k: int = K) -> float:
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    seen: set[int] = set()
    for rank, item in enumerate(predictions[:k], start=1):
        if item in seen:
            continue
        seen.add(item)
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / min(k, len(relevant))


def evaluate_rankings(
    rankings: Mapping[int, Sequence[int]],
    targets: Mapping[int, set[int]],
    users: Sequence[int],
    history: Mapping[int, Counter[int]],
) -> dict[str, float | int]:
    eligible = [user for user in users if targets.get(user)]
    scores = [ap_at_k(rankings[user], targets[user]) for user in eligible]
    hit_users = sum(bool(set(rankings[user][:K]) & targets[user]) for user in eligible)
    recall_values = [
        len(set(rankings[user][:K]) & targets[user]) / len(targets[user]) for user in eligible
    ]
    cold = [user for user in eligible if not history.get(user)]
    warm = [user for user in eligible if history.get(user)]

    def mean_ap(group: Sequence[int]) -> float:
        if not group:
            return 0.0
        return sum(ap_at_k(rankings[user], targets[user]) for user in group) / len(group)

    return {
        "map_at_5": sum(scores) / len(scores) if scores else 0.0,
        "hit_rate_at_5": hit_users / len(eligible) if eligible else 0.0,
        "mean_recall_at_5": sum(recall_values) / len(recall_values) if recall_values else 0.0,
        "eligible_users": len(eligible),
        "all_requested_users": len(users),
        "cold_start_users": len(cold),
        "warm_users": len(warm),
        "cold_map_at_5": mean_ap(cold),
        "warm_map_at_5": mean_ap(warm),
        "zero_filled_all_user_map_at_5": sum(scores) / len(users) if users else 0.0,
    }


def popularity_order(
    train_interactions: Mapping[int, Counter[int]],
    candidates: set[int],
    target_schedule_frequency: Mapping[int, int],
) -> list[int]:
    popularity = aggregate_item_counts(train_interactions)
    return sorted(
        candidates,
        key=lambda item: (-popularity[item], -target_schedule_frequency.get(item, 0), item),
    )


def fill_ranking(
    scored_items: Iterable[int], fallback: Sequence[int], candidates: set[int], topn: int
) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for item in list(scored_items) + list(fallback):
        if item in candidates and item not in seen:
            result.append(item)
            seen.add(item)
            if len(result) == min(topn, len(candidates)):
                break
    return result


def popularity_rankings(users: Sequence[int], fallback: Sequence[int], topn: int) -> dict[int, list[int]]:
    top = list(fallback[:topn])
    return {user: top.copy() for user in users}


def content_rankings(
    users: Sequence[int],
    interactions: Mapping[int, Counter[int]],
    train_features: Mapping[int, set[str]],
    candidate_features: Mapping[int, set[str]],
    candidates: set[int],
    fallback: Sequence[int],
    history_weighting: str,
    topn: int = 100,
) -> dict[int, list[int]]:
    """Rank by cosine similarity between an IDF-weighted user profile and item features."""
    feature_df: Counter[str] = Counter()
    for features in train_features.values():
        feature_df.update(features)
    catalog_size = max(1, len(train_features))
    idf = {
        feature: math.log((catalog_size + 1) / (count + 1)) + 1.0
        for feature, count in feature_df.items()
    }

    postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for item in candidates:
        weighted = [(feature, idf[feature]) for feature in candidate_features.get(item, set()) if feature in idf]
        norm = math.sqrt(sum(weight * weight for _, weight in weighted))
        if norm:
            for feature, weight in weighted:
                postings[feature].append((item, weight / norm))

    rankings: dict[int, list[int]] = {}
    for user in users:
        profile: dict[str, float] = defaultdict(float)
        for item, count in interactions.get(user, {}).items():
            if history_weighting == "binary":
                item_weight = 1.0
            elif history_weighting == "log_count":
                item_weight = math.log1p(count)
            else:
                raise ValueError(f"Unknown history weighting: {history_weighting}")
            for feature in train_features.get(item, set()):
                if feature in idf:
                    profile[feature] += item_weight * idf[feature]
        norm = math.sqrt(sum(value * value for value in profile.values()))
        scores: dict[int, float] = defaultdict(float)
        if norm:
            for feature, value in profile.items():
                coefficient = value / norm
                for item, candidate_weight in postings.get(feature, ()):
                    scores[item] += coefficient * candidate_weight
        scored = sorted(scores, key=lambda item: (-scores[item], item))
        rankings[user] = fill_ranking(scored, fallback, candidates, topn)
    return rankings


@dataclass
class RuleCounts:
    window_days: int
    basket_count: int
    item_support: Counter[int]
    pair_support: Counter[tuple[int, int]]


def build_rule_counts(events: Mapping[int, Sequence[tuple[int, int]]], window_days: int) -> RuleCounts:
    baskets: dict[tuple[int, int], set[int]] = defaultdict(set)
    for user, rows in events.items():
        for day, item in rows:
            baskets[(user, day // window_days)].add(item)
    item_support: Counter[int] = Counter()
    pair_support: Counter[tuple[int, int]] = Counter()
    for basket in baskets.values():
        ordered = sorted(basket)
        item_support.update(ordered)
        for left_index in range(len(ordered)):
            left = ordered[left_index]
            for right in ordered[left_index + 1 :]:
                pair_support[(left, right)] += 1
    return RuleCounts(window_days, len(baskets), item_support, pair_support)


def association_rankings(
    users: Sequence[int],
    events: Mapping[int, Sequence[tuple[int, int]]],
    counts: RuleCounts,
    candidates: set[int],
    fallback: Sequence[int],
    min_pair_support: int,
    metric: str,
    history_limit: int = 50,
    recency_decay: float = 0.95,
    topn: int = 100,
) -> dict[int, list[int]]:
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    basket_count = max(1, counts.basket_count)
    for (left, right), pair_count in counts.pair_support.items():
        if pair_count < min_pair_support:
            continue
        left_support = counts.item_support[left]
        right_support = counts.item_support[right]
        left_confidence = pair_count / left_support
        right_confidence = pair_count / right_support
        lift = pair_count * basket_count / (left_support * right_support)
        if metric == "confidence":
            left_score = left_confidence
            right_score = right_confidence
        elif metric == "confidence_lift":
            multiplier = math.log1p(lift)
            left_score = left_confidence * multiplier
            right_score = right_confidence * multiplier
        else:
            raise ValueError(f"Unknown association metric: {metric}")
        adjacency[left].append((right, left_score))
        adjacency[right].append((left, right_score))

    rankings: dict[int, list[int]] = {}
    for user in users:
        last_seen: dict[int, int] = {}
        for day, item in events.get(user, ()):
            last_seen[item] = max(day, last_seen.get(item, day))
        recent_items = sorted(last_seen, key=lambda item: (-last_seen[item], item))[:history_limit]
        scores: dict[int, float] = defaultdict(float)
        for rank, antecedent in enumerate(recent_items):
            weight = recency_decay**rank
            for consequent, rule_score in adjacency.get(antecedent, ()):
                if consequent in candidates:
                    scores[consequent] += weight * rule_score
        scored = sorted(scores, key=lambda item: (-scores[item], item))
        rankings[user] = fill_ranking(scored, fallback, candidates, topn)
    return rankings


def reciprocal_rank_fusion(
    users: Sequence[int],
    left: Mapping[int, Sequence[int]],
    right: Mapping[int, Sequence[int]],
    candidates: set[int],
    fallback: Sequence[int],
    left_weight: float,
    rank_constant: int = 60,
    topn: int = 100,
) -> dict[int, list[int]]:
    rankings: dict[int, list[int]] = {}
    for user in users:
        scores: dict[int, float] = defaultdict(float)
        for rank, item in enumerate(left[user], start=1):
            scores[item] += left_weight / (rank_constant + rank)
        for rank, item in enumerate(right[user], start=1):
            scores[item] += (1.0 - left_weight) / (rank_constant + rank)
        scored = sorted(scores, key=lambda item: (-scores[item], item))
        rankings[user] = fill_ranking(scored, fallback, candidates, topn)
    return rankings


def business_metrics(
    rankings: Mapping[int, Sequence[int]],
    users: Sequence[int],
    candidates: set[int],
    train_interactions: Mapping[int, Counter[int]],
) -> dict[str, float | int]:
    recommended = [item for user in users for item in rankings[user][:K]]
    unique_recommended = set(recommended)
    item_user_count: Counter[int] = Counter()
    for counts in train_interactions.values():
        item_user_count.update(counts.keys())
    # Probabilities must use the same population as item_user_count. Using the
    # requested/evaluation users here can make p(item) > 1 when the training
    # set contains additional users, which would incorrectly yield negative
    # self-information.
    user_count = max(1, len(train_interactions))
    novelty = sum(
        -math.log2((item_user_count[item] + 1) / (user_count + 1)) for item in recommended
    ) / len(recommended)
    repeats = sum(
        item in train_interactions.get(user, {})
        for user in users
        for item in rankings[user][:K]
    )
    return {
        "catalog_coverage_count": len(unique_recommended),
        "catalog_coverage_fraction": len(unique_recommended) / len(candidates),
        "mean_self_information_bits": novelty,
        "repeat_recommendation_rate": repeats / len(recommended),
    }


def mean_top5_jaccard(
    left: Mapping[int, Sequence[int]], right: Mapping[int, Sequence[int]], users: Sequence[int]
) -> float:
    values = []
    for user in users:
        left_set = set(left[user][:K])
        right_set = set(right[user][:K])
        union = left_set | right_set
        values.append(len(left_set & right_set) / len(union) if union else 1.0)
    return sum(values) / len(values) if values else 0.0


def histogram_quantile(histogram: Sequence[int], probability: float) -> float:
    total = sum(histogram)
    if total == 0:
        return 0.0
    target = max(1, math.ceil(total * probability))
    cumulative = 0
    for index, count in enumerate(histogram):
        cumulative += count
        if cumulative >= target:
            return index / (len(histogram) - 1)
    return 1.0


def exact_quantile(values: Sequence[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def canonical_label(counters: Mapping[int, Counter[str]], item: int, default: str) -> str:
    values = counters.get(item)
    if not values:
        return default
    return min(values, key=lambda value: (-values[value], value))


def read_submission_users(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [int(row["user_id"]) for row in csv.DictReader(handle)]


def write_submission(path: Path, users: Sequence[int], rankings: Mapping[int, Sequence[int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["user_id", "tv_show_id"])
        for user in users:
            writer.writerow([user, " ".join(str(item) for item in rankings[user][:K])])


def validate_submission(
    path: Path, expected_users: Sequence[int], candidates: set[int]
) -> dict[str, int | bool | str]:
    errors: list[str] = []
    observed_users: list[int] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["user_id", "tv_show_id"]:
            errors.append(f"unexpected header: {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            user = int(row["user_id"])
            observed_users.append(user)
            try:
                items = [int(value) for value in row["tv_show_id"].split()]
            except ValueError:
                errors.append(f"line {line_number}: non-integer item")
                continue
            if len(items) != K:
                errors.append(f"line {line_number}: expected {K} items, got {len(items)}")
            if len(set(items)) != len(items):
                errors.append(f"line {line_number}: duplicate items")
            if any(item <= 0 for item in items):
                errors.append(f"line {line_number}: non-positive item")
            invalid = [item for item in items if item not in candidates]
            if invalid:
                errors.append(f"line {line_number}: items outside candidate pool {invalid}")
    if observed_users != list(expected_users):
        errors.append("user ids or row order differ from the template")
    return {
        "valid": not errors,
        "rows": len(observed_users),
        "expected_rows": len(expected_users),
        "error_count": len(errors),
        "errors": " | ".join(errors[:10]),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_memory_mib() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def environment_info(seed: int = DEFAULT_SEED) -> dict[str, object]:
    random.seed(seed)
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "seed": seed,
        "dependencies": "Python standard library only",
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
