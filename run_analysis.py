#!/usr/bin/env python3
"""Run the full SWEET.TV validation, model selection, and submission pipeline."""

from __future__ import annotations

import csv
import gc
import json
import math
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.recsys import (
    DEFAULT_SEED,
    K,
    WATCH_THRESHOLD,
    aggregate_item_counts,
    ap_at_k,
    association_rankings,
    build_rule_counts,
    build_targets,
    business_metrics,
    canonical_label,
    content_rankings,
    environment_info,
    evaluate_rankings,
    exact_quantile,
    histogram_quantile,
    join_views_to_schedule,
    load_schedules,
    mean_top5_jaccard,
    merge_events,
    merge_feature_maps,
    merge_interactions,
    peak_memory_mib,
    popularity_order,
    popularity_rankings,
    read_submission_users,
    reciprocal_rank_fusion,
    sha256_file,
    validate_submission,
    write_json,
    write_submission,
)


ROOT = Path(__file__).resolve().parent
EXAM_ROOT = ROOT.parent
RAW = EXAM_ROOT / "data" / "raw"
PROCESSED = EXAM_ROOT / "data" / "processed"
EVIDENCE = ROOT / "evidence"
SUBMISSIONS = ROOT / "submissions"

TRAIN_SCHEDULE = RAW / "export_arh_11-20-final.csv"
VALIDATION_SCHEDULE = RAW / "export_arh_21-30-final.csv"
TEST_SCHEDULE = RAW / "export_arh_31-42-final.csv"
VIEWS = RAW / "dataset11-30.csv"
TEMPLATE = RAW / "submission.csv"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def merge_counter_maps(*maps: dict[int, Counter[str]]) -> dict[int, Counter[str]]:
    result: dict[int, Counter[str]] = defaultdict(Counter)
    for mapping in maps:
        for item, counter in mapping.items():
            result[item].update(counter)
    return dict(result)


def cache_signature() -> dict[str, object]:
    paths = [VIEWS, TRAIN_SCHEDULE, VALIDATION_SCHEDULE]
    return {
        "threshold": WATCH_THRESHOLD,
        "inputs": {
            str(path.name): {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths
        },
        "join_version": 2,
    }


def main() -> None:
    total_started = time.perf_counter()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    print("Loading training and validation schedules...", flush=True)
    started = time.perf_counter()
    schedule = load_schedules(
        {"train": TRAIN_SCHEDULE, "validation": VALIDATION_SCHEDULE},
        include_airings=True,
    )
    timings["load_train_validation_schedules"] = time.perf_counter() - started

    cache_path = PROCESSED / "join_11_30_v2.pkl"
    signature = cache_signature()
    join_result = None
    if cache_path.exists():
        try:
            with cache_path.open("rb") as handle:
                cached = pickle.load(handle)
            if cached.get("signature") == signature:
                join_result = cached["result"]
                print("Using verified join cache.", flush=True)
        except (OSError, pickle.PickleError, EOFError, AttributeError):
            join_result = None
    if join_result is None:
        print("Joining 5.8M view rows to labelled airings...", flush=True)
        join_result = join_views_to_schedule(VIEWS, schedule, WATCH_THRESHOLD)
        with cache_path.open("wb") as handle:
            pickle.dump({"signature": signature, "result": join_result}, handle, protocol=5)
    timings["join_views_to_schedule"] = join_result.runtime_seconds

    print("Loading test schedule metadata and candidate pool (no test views)...", flush=True)
    started = time.perf_counter()
    test_bundle = load_schedules({"test": TEST_SCHEDULE}, include_airings=False)
    timings["load_test_schedule"] = time.perf_counter() - started

    train_info = schedule.periods["train"]
    validation_info = schedule.periods["validation"]
    test_info = test_bundle.periods["test"]
    train_interactions = join_result.interactions["train"]
    validation_interactions = join_result.interactions["validation"]
    train_events = join_result.events["train"]
    validation_events = join_result.events["validation"]
    users = read_submission_users(TEMPLATE)
    validation_targets = build_targets(validation_interactions, K)

    fallback_validation = popularity_order(
        train_interactions,
        validation_info.candidates,
        validation_info.schedule_frequency,
    )
    tuning_rows: list[dict[str, object]] = []

    print("Evaluating popularity baseline...", flush=True)
    started = time.perf_counter()
    popularity_validation = popularity_rankings(users, fallback_validation, topn=100)
    popularity_metrics = evaluate_rankings(
        popularity_validation, validation_targets, users, train_interactions
    )
    tuning_rows.append(
        {
            "method": "popularity",
            "variant": "interaction_count_then_target_schedule_frequency",
            "map_at_5": popularity_metrics["map_at_5"],
            "hit_rate_at_5": popularity_metrics["hit_rate_at_5"],
            "mean_recall_at_5": popularity_metrics["mean_recall_at_5"],
            "runtime_seconds": time.perf_counter() - started,
            "parameters": json.dumps({}, sort_keys=True),
        }
    )

    print("Tuning content-based variants on validation only...", flush=True)
    best_content: tuple[float, dict[str, object], dict[int, list[int]], dict[str, object]] | None = None
    for feature_set in ("basic", "full"):
        train_features = (
            train_info.features_basic if feature_set == "basic" else train_info.features_full
        )
        target_features = (
            validation_info.features_basic
            if feature_set == "basic"
            else validation_info.features_full
        )
        for weighting in ("binary", "log_count"):
            params = {"feature_set": feature_set, "history_weighting": weighting}
            started = time.perf_counter()
            rankings = content_rankings(
                users,
                train_interactions,
                train_features,
                target_features,
                validation_info.candidates,
                fallback_validation,
                history_weighting=weighting,
                topn=100,
            )
            runtime = time.perf_counter() - started
            metrics = evaluate_rankings(rankings, validation_targets, users, train_interactions)
            tuning_rows.append(
                {
                    "method": "content",
                    "variant": f"{feature_set}_{weighting}",
                    "map_at_5": metrics["map_at_5"],
                    "hit_rate_at_5": metrics["hit_rate_at_5"],
                    "mean_recall_at_5": metrics["mean_recall_at_5"],
                    "runtime_seconds": runtime,
                    "parameters": json.dumps(params, sort_keys=True),
                }
            )
            candidate = (float(metrics["map_at_5"]), params, rankings, metrics)
            if best_content is None or candidate[0] > best_content[0]:
                best_content = candidate
    assert best_content is not None

    print("Tuning association-rule variants on validation only...", flush=True)
    best_association: tuple[float, dict[str, object], dict[int, list[int]], dict[str, object]] | None = None
    rule_count_evidence: list[dict[str, object]] = []
    for window_days in (1, 3):
        build_started = time.perf_counter()
        rule_counts = build_rule_counts(train_events, window_days)
        build_runtime = time.perf_counter() - build_started
        rule_count_evidence.append(
            {
                "window_days": window_days,
                "basket_count": rule_counts.basket_count,
                "distinct_items": len(rule_counts.item_support),
                "distinct_pairs": len(rule_counts.pair_support),
                "build_runtime_seconds": build_runtime,
            }
        )
        for min_support in (2, 5):
            for metric in ("confidence", "confidence_lift"):
                params = {
                    "window_days": window_days,
                    "min_pair_support": min_support,
                    "metric": metric,
                    "history_limit": 50,
                    "recency_decay": 0.95,
                }
                started = time.perf_counter()
                rankings = association_rankings(
                    users,
                    train_events,
                    rule_counts,
                    validation_info.candidates,
                    fallback_validation,
                    min_pair_support=min_support,
                    metric=metric,
                    history_limit=50,
                    recency_decay=0.95,
                    topn=100,
                )
                runtime = time.perf_counter() - started
                metrics = evaluate_rankings(rankings, validation_targets, users, train_interactions)
                tuning_rows.append(
                    {
                        "method": "association",
                        "variant": f"w{window_days}_s{min_support}_{metric}",
                        "map_at_5": metrics["map_at_5"],
                        "hit_rate_at_5": metrics["hit_rate_at_5"],
                        "mean_recall_at_5": metrics["mean_recall_at_5"],
                        "runtime_seconds": runtime + build_runtime,
                        "parameters": json.dumps(params, sort_keys=True),
                    }
                )
                candidate = (float(metrics["map_at_5"]), params, rankings, metrics)
                if best_association is None or candidate[0] > best_association[0]:
                    best_association = candidate
        del rule_counts
        gc.collect()
    assert best_association is not None

    print("Tuning rank-fusion weights on validation only...", flush=True)
    best_hybrid: tuple[float, dict[str, object], dict[int, list[int]], dict[str, object]] | None = None
    for content_weight in (0.25, 0.50, 0.75):
        params = {"content_weight": content_weight, "rank_constant": 60}
        started = time.perf_counter()
        rankings = reciprocal_rank_fusion(
            users,
            best_content[2],
            best_association[2],
            validation_info.candidates,
            fallback_validation,
            left_weight=content_weight,
            rank_constant=60,
            topn=100,
        )
        runtime = time.perf_counter() - started
        metrics = evaluate_rankings(rankings, validation_targets, users, train_interactions)
        tuning_rows.append(
            {
                "method": "hybrid",
                "variant": f"rrf_content_{content_weight:.2f}",
                "map_at_5": metrics["map_at_5"],
                "hit_rate_at_5": metrics["hit_rate_at_5"],
                "mean_recall_at_5": metrics["mean_recall_at_5"],
                "runtime_seconds": runtime,
                "parameters": json.dumps(params, sort_keys=True),
            }
        )
        candidate = (float(metrics["map_at_5"]), params, rankings, metrics)
        if best_hybrid is None or candidate[0] > best_hybrid[0]:
            best_hybrid = candidate
    assert best_hybrid is not None

    validation_rankings = {
        "popularity": popularity_validation,
        "content": best_content[2],
        "association": best_association[2],
        "hybrid": best_hybrid[2],
    }
    validation_metrics = {
        "popularity": popularity_metrics,
        "content": best_content[3],
        "association": best_association[3],
        "hybrid": best_hybrid[3],
    }
    validation_params = {
        "popularity": {},
        "content": best_content[1],
        "association": best_association[1],
        "hybrid": best_hybrid[1],
    }
    # Highest validation MAP@5 wins. Ties prefer the simpler strategy in this order.
    strategy_priority = {"popularity": 3, "content": 2, "association": 1, "hybrid": 0}
    selected_strategy = max(
        validation_metrics,
        key=lambda method: (
            float(validation_metrics[method]["map_at_5"]),
            strategy_priority[method],
        ),
    )

    print("Computing business-analysis evidence and user examples...", flush=True)
    business_rows: list[dict[str, object]] = []
    for method, rankings in validation_rankings.items():
        row: dict[str, object] = {"method": method}
        row.update(validation_metrics[method])
        row.update(
            business_metrics(
                rankings,
                users,
                validation_info.candidates,
                train_interactions,
            )
        )
        business_rows.append(row)
    for left, right in (("content", "association"), ("content", "hybrid"), ("association", "hybrid")):
        business_rows.append(
            {
                "method": f"overlap_{left}_vs_{right}",
                "mean_top5_jaccard": mean_top5_jaccard(
                    validation_rankings[left], validation_rankings[right], users
                ),
            }
        )

    eligible_warm = [
        user for user in users if validation_targets.get(user) and sum(train_interactions.get(user, {}).values()) > 0
    ]
    heavy_user = min(
        eligible_warm,
        key=lambda user: (-sum(train_interactions[user].values()), user),
    )
    light_user = min(
        eligible_warm,
        key=lambda user: (sum(train_interactions[user].values()), user),
    )
    selected_users = [heavy_user, light_user]
    titles = merge_counter_maps(train_info.titles, validation_info.titles)
    categories = merge_counter_maps(train_info.categories, validation_info.categories)
    example_rows: list[dict[str, object]] = []
    for role, user in zip(("heavy", "light"), selected_users):
        for method, rankings in validation_rankings.items():
            for rank, item in enumerate(rankings[user][:K], start=1):
                example_rows.append(
                    {
                        "selection_role": role,
                        "user_id": user,
                        "training_watched_airings": sum(train_interactions[user].values()),
                        "training_unique_programs": len(train_interactions[user]),
                        "method": method,
                        "rank": rank,
                        "tv_show_id": item,
                        "title": canonical_label(titles, item, "Unknown title"),
                        "category": canonical_label(categories, item, "Unknown category"),
                        "is_validation_relevant": item in validation_targets[user],
                    }
                )
    actual_rows: list[dict[str, object]] = []
    for role, user in zip(("heavy", "light"), selected_users):
        counts = validation_interactions[user]
        for item in sorted(validation_targets[user], key=lambda item: (-counts[item], item)):
            actual_rows.append(
                {
                    "selection_role": role,
                    "user_id": user,
                    "tv_show_id": item,
                    "watched_airing_count": counts[item],
                    "title": canonical_label(titles, item, "Unknown title"),
                    "category": canonical_label(categories, item, "Unknown category"),
                }
            )

    selected_validation_ranking = validation_rankings[selected_strategy]
    positive_manual_users = [
        user
        for user in users
        if validation_targets.get(user)
        and ap_at_k(selected_validation_ranking[user], validation_targets[user]) > 0
    ]
    manual_user = min(positive_manual_users) if positive_manual_users else min(validation_targets)
    manual_prediction = selected_validation_ranking[manual_user][:K]
    manual_relevant = validation_targets[manual_user]
    manual_steps: list[dict[str, object]] = []
    hits = 0
    numerator = 0.0
    for rank, item in enumerate(manual_prediction, start=1):
        is_hit = item in manual_relevant
        if is_hit:
            hits += 1
        precision = hits / rank
        contribution = precision if is_hit else 0.0
        numerator += contribution
        manual_steps.append(
            {
                "rank": rank,
                "tv_show_id": item,
                "is_relevant": is_hit,
                "cumulative_hits": hits,
                "precision_at_rank": precision,
                "ap_contribution": contribution,
            }
        )
    denominator = min(K, len(manual_relevant))
    manual_ap = numerator / denominator

    print("Retraining selected configurations on weeks 11-30...", flush=True)
    combined_interactions = merge_interactions(train_interactions, validation_interactions)
    combined_events = merge_events(train_events, validation_events)
    fallback_test = popularity_order(
        combined_interactions,
        test_info.candidates,
        test_info.schedule_frequency,
    )
    final_popularity = popularity_rankings(users, fallback_test, topn=100)

    content_params = best_content[1]
    combined_train_features = merge_feature_maps(
        train_info.features_basic if content_params["feature_set"] == "basic" else train_info.features_full,
        validation_info.features_basic
        if content_params["feature_set"] == "basic"
        else validation_info.features_full,
    )
    test_features = (
        test_info.features_basic
        if content_params["feature_set"] == "basic"
        else test_info.features_full
    )
    started = time.perf_counter()
    final_content = content_rankings(
        users,
        combined_interactions,
        combined_train_features,
        test_features,
        test_info.candidates,
        fallback_test,
        history_weighting=str(content_params["history_weighting"]),
        topn=100,
    )
    timings["final_content_fit_and_rank"] = time.perf_counter() - started

    association_params = best_association[1]
    started = time.perf_counter()
    final_rule_counts = build_rule_counts(combined_events, int(association_params["window_days"]))
    final_association = association_rankings(
        users,
        combined_events,
        final_rule_counts,
        test_info.candidates,
        fallback_test,
        min_pair_support=int(association_params["min_pair_support"]),
        metric=str(association_params["metric"]),
        history_limit=int(association_params["history_limit"]),
        recency_decay=float(association_params["recency_decay"]),
        topn=100,
    )
    timings["final_association_fit_and_rank"] = time.perf_counter() - started

    hybrid_params = best_hybrid[1]
    started = time.perf_counter()
    final_hybrid = reciprocal_rank_fusion(
        users,
        final_content,
        final_association,
        test_info.candidates,
        fallback_test,
        left_weight=float(hybrid_params["content_weight"]),
        rank_constant=int(hybrid_params["rank_constant"]),
        topn=100,
    )
    timings["final_hybrid_rank"] = time.perf_counter() - started
    final_rankings = {
        "popularity": final_popularity,
        "content": final_content,
        "association": final_association,
        "hybrid": final_hybrid,
    }

    print("Writing and validating Kaggle submission files...", flush=True)
    submission_checks: dict[str, object] = {}
    for method, rankings in final_rankings.items():
        path = SUBMISSIONS / f"submission_{method}.csv"
        write_submission(path, users, rankings)
        check = validate_submission(path, users, test_info.candidates)
        check["sha256"] = sha256_file(path)
        check["bytes"] = path.stat().st_size
        submission_checks[method] = check
        if not check["valid"]:
            raise ValueError(f"Invalid {method} submission: {check}")

    validation_rows = []
    for method in ("popularity", "content", "association", "hybrid"):
        row = {
            "method": method,
            "selected_strategy": method == selected_strategy,
            "parameters": json.dumps(validation_params[method], sort_keys=True),
        }
        row.update(validation_metrics[method])
        validation_rows.append(row)
    write_csv(
        EVIDENCE / "validation_results.csv",
        [
            "method",
            "selected_strategy",
            "map_at_5",
            "hit_rate_at_5",
            "mean_recall_at_5",
            "eligible_users",
            "all_requested_users",
            "cold_start_users",
            "warm_users",
            "cold_map_at_5",
            "warm_map_at_5",
            "zero_filled_all_user_map_at_5",
            "parameters",
        ],
        validation_rows,
    )
    write_csv(
        EVIDENCE / "tuning_results.csv",
        [
            "method",
            "variant",
            "map_at_5",
            "hit_rate_at_5",
            "mean_recall_at_5",
            "runtime_seconds",
            "parameters",
        ],
        tuning_rows,
    )
    write_csv(
        EVIDENCE / "business_metrics.csv",
        sorted({key for row in business_rows for key in row}),
        business_rows,
    )
    write_csv(
        EVIDENCE / "selected_users_top5.csv",
        list(example_rows[0]),
        example_rows,
    )
    write_csv(
        EVIDENCE / "selected_users_actual.csv",
        list(actual_rows[0]),
        actual_rows,
    )

    data_summary = {
        "join_rule": "same channel and positive interval overlap; union overlap per user-airing",
        "watch_threshold": WATCH_THRESHOLD,
        "watch_rule": (
            "Our operational decision: a program is classified as watched when the union "
            "of the user's viewing intervals covers at least 80% of its scheduled duration "
            "on the same channel."
        ),
        "threshold_basis": (
            "Our 0.80 threshold was fixed before validation and was not tuned. It aligns "
            "with the Kaggle Data page's description of target construction; this is not "
            "presented as a universal viewing rule defined by Kaggle."
        ),
        "threshold_kaggle_quote": (
            "Only TV shows with high screen time (>= 0.8) were taken into account."
        ),
        "threshold_source": "https://www.kaggle.com/competitions/sweettv-tv-program-recommender/data",
        "tv_show_id_zero_policy": "Dropped before the join because zero denotes unlabelled airtime, not an item.",
        "repeats_policy": "Previously watched programs remain eligible for every method.",
        "cold_start_policy": "Training interaction popularity restricted to the target candidate pool; target schedule frequency then item id break ties.",
        "candidate_pool_policy": "All nonzero tv_show_id values airing in the target schedule period.",
        "view_rows": join_result.view_rows,
        "skipped_view_rows": join_result.skipped_view_rows,
        "view_sort_violations": join_result.sorted_violations,
        "matched_user_airings": join_result.matched_user_airings,
        "watched_user_airings_at_0_8": join_result.watched_user_airings,
        "fraction_at_or_above_0_8": sum(join_result.overlap_histogram[800:])
        / max(1, join_result.matched_user_airings),
        "percentage_at_or_above_0_8": round(
            100
            * sum(join_result.overlap_histogram[800:])
            / max(1, join_result.matched_user_airings),
            4,
        ),
        "overlap_fraction_quantiles": {
            str(q): histogram_quantile(join_result.overlap_histogram, q)
            for q in (0.25, 0.50, 0.75, 0.80, 0.90, 0.95)
        },
        "schedule_duration_seconds_quantiles": {
            period: {
                str(q): exact_quantile(info.durations, q)
                for q in (0.25, 0.50, 0.75, 0.90, 0.99)
            }
            for period, info in (
                ("train", train_info),
                ("validation", validation_info),
                ("test", test_info),
            )
        },
        "periods": {
            period: {
                "labelled_schedule_rows": info.labelled_rows,
                "zero_id_schedule_rows": info.zero_id_rows,
                "invalid_duration_rows": info.invalid_duration_rows,
                "candidate_items": len(info.candidates),
            }
            for period, info in (
                ("train", train_info),
                ("validation", validation_info),
                ("test", test_info),
            )
        },
        "train_users_with_interactions": len(train_interactions),
        "validation_users_with_targets": len(validation_targets),
        "template_users": len(users),
        "input_sha256": {
            path.name: sha256_file(path)
            for path in (VIEWS, TRAIN_SCHEDULE, VALIDATION_SCHEDULE, TEST_SCHEDULE, TEMPLATE)
        },
    }
    write_json(EVIDENCE / "data_summary.json", data_summary)
    write_json(EVIDENCE / "association_rule_counts.json", rule_count_evidence)
    write_json(
        EVIDENCE / "selected_strategy.json",
        {
            "strategy": selected_strategy,
            "selection_rule": "Highest validation MAP@5; exact ties prefer popularity, content, association, then hybrid for simplicity.",
            "validation_metrics": validation_metrics[selected_strategy],
            "parameters": validation_params[selected_strategy],
            "two_user_selection_rule": (
                "Among validation-eligible users with positive training history, choose maximum and minimum "
                "training watched-airing counts; ties use the smallest numeric user_id."
            ),
            "heavy_user": heavy_user,
            "light_user": light_user,
        },
    )
    write_json(EVIDENCE / "submission_checks.json", submission_checks)
    write_json(
        EVIDENCE / "manual_ap5_check.json",
        {
            "method": selected_strategy,
            "user_id": manual_user,
            "predictions": manual_prediction,
            "relevant_items": sorted(manual_relevant),
            "steps": manual_steps,
            "numerator": numerator,
            "denominator": denominator,
            "ap_at_5": manual_ap,
            "code_cross_check": ap_at_k(manual_prediction, manual_relevant),
        },
    )
    manual_lines = [
        "# Manual AP@5 verification",
        "",
        f"- Method: `{selected_strategy}`",
        f"- User: `{manual_user}`",
        f"- Predictions: `{' '.join(map(str, manual_prediction))}`",
        f"- Relevant set: `{' '.join(map(str, sorted(manual_relevant)))}`",
        "",
        "| Rank | Program | Relevant? | Hits so far | Precision | Contribution |",
        "| ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    for step in manual_steps:
        manual_lines.append(
            f"| {step['rank']} | {step['tv_show_id']} | {'yes' if step['is_relevant'] else 'no'} | "
            f"{step['cumulative_hits']} | {step['precision_at_rank']:.6f} | "
            f"{step['ap_contribution']:.6f} |"
        )
    manual_lines.extend(
        [
            "",
            f"Numerator = `{numerator:.12f}`.",
            f"Denominator = `min(5, {len(manual_relevant)}) = {denominator}`.",
            f"AP@5 = `{numerator:.12f} / {denominator} = {manual_ap:.12f}`.",
            "",
            "The student must recompute this value independently before it is cited as personally verified.",
        ]
    )
    (EVIDENCE / "manual_ap5_check.md").write_text("\n".join(manual_lines) + "\n", encoding="utf-8")

    timings["total_pipeline"] = time.perf_counter() - total_started
    write_json(
        EVIDENCE / "run_manifest.json",
        {
            "environment": environment_info(DEFAULT_SEED),
            "k": K,
            "seed": DEFAULT_SEED,
            "watch_threshold": WATCH_THRESHOLD,
            "timings_seconds": timings,
            "peak_memory_mib": peak_memory_mib(),
            "selected_strategy": selected_strategy,
            "selected_parameters": validation_params[selected_strategy],
            "method_parameters": validation_params,
            "test_period_viewing_used": False,
            "test_period_schedule_use": "candidate pool and static item metadata only",
        },
    )
    print(
        f"Done. Selected strategy: {selected_strategy}; "
        f"validation MAP@5={validation_metrics[selected_strategy]['map_at_5']:.8f}; "
        f"peak memory={peak_memory_mib():.1f} MiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
