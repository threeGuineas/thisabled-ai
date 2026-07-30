"""P4 — MATCH v2 랭커의 공정성·활동량 편향 평가.

측정(모두 학습과 동일한 결정적 test split에서):
1. ui_mode Demographic Parity: 각 쿼리의 후보를 모델 점수로 정렬해 top-K 추천을 만들고,
   ui_mode g별 selection_rate[g] = (top-K에 든 g 후보 수)/(슬레이트의 g 후보 총수).
   DP diff = max_g − min_g. project_facts 게이트 승계: ≤ 0.10.
2. 활동량 편향: 후보를 콘텐츠 수(authored+liked) 중앙값 기준 상/하로 나눠 top-K 선택률 비교.
   활동이 많은 사용자가 점수를 지배하는지(구현계획 위험 항목) 정량 확인.

라벨은 ui_mode·장애에 의존하지 않으므로(build_pairs_v2), DP가 크면 모델이 만든 편향이다.
합성 데이터라 결과는 '합성 타당성' 수준으로 해석한다.

SBERT가 필요하므로 Colab 등 GPU 환경에서 학습 직후 같은 encoder로 실행하거나,
run_fairness(encoder=...)를 노트북에서 호출한다.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.build_pairs_v2 import LabelWeights, build_pairs  # noqa: E402
from src.data.build_profiles_v2 import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    GenerationConfig,
    generate_population,
    load_allowed_tags,
)
from src.data.match_eval_pack import (  # noqa: E402
    load_current_match_run,
    load_trusted_model_bundle,
)
from src.data.matching_input import (  # noqa: E402
    MatchingInputPolicy,
    TextEncoder,
    build_pair_features,
    candidate_exclusion_reason,
)
from src.data.matching_trainset import embed_users, split_users  # noqa: E402

DP_GATE = 0.10


def _scored_slates(
    test_users: list,
    test_pairs: list,
    *,
    model: Any,
    columns: list[str],
    encoder: TextEncoder,
    policy: MatchingInputPolicy,
    as_of,
) -> list[dict[str, Any]]:
    """쿼리별로 후보를 모델 점수로 정렬한 슬레이트를 만든다(제외 규칙 반영)."""

    feat_by_id = embed_users(test_users, encoder=encoder, policy=policy, as_of=as_of)
    snap_by_id = {u.snapshot.user_id: u for u in test_users}

    grouped: dict[str, list] = defaultdict(list)
    for pair in test_pairs:
        grouped[pair.query_id].append(pair)

    slates: list[dict[str, Any]] = []
    for query_id, recs in grouped.items():
        query_feat = feat_by_id.get(query_id)
        if query_feat is None:
            continue
        me = snap_by_id[query_id].snapshot
        rows: list[list[float]] = []
        ui_modes: list[str] = []
        activity: list[int] = []
        for pair in recs:
            cand_feat = feat_by_id.get(pair.cand_id)
            if cand_feat is None:
                continue
            relationship = pair.candidate_input.relationship
            if (
                candidate_exclusion_reason(
                    me,
                    pair.candidate_input.profile,
                    relationship,
                    as_of=as_of,
                    rejection_cooldown_days=policy.rejection_cooldown_days,
                )
                is not None
            ):
                continue
            feats = build_pair_features(query_feat, cand_feat, relationship)
            rows.append([float(feats[c]) for c in columns])
            ui_modes.append(pair.candidate_input.profile.ui_mode or "")
            activity.append(cand_feat.authored_count + cand_feat.liked_count)
        if len(rows) < 2:
            continue
        scores = np.asarray(model.predict(np.asarray(rows, dtype=np.float32)), dtype=float)
        slates.append({"scores": scores, "ui_mode": ui_modes, "activity": activity})
    return slates


def _selection_rates(slates: list[dict[str, Any]], group_of, k: int) -> dict[Any, tuple[int, int]]:
    """group별 (top-K 선택 수, 슬레이트 총수)."""

    hit: dict[Any, int] = defaultdict(int)
    total: dict[Any, int] = defaultdict(int)
    for slate in slates:
        groups = group_of(slate)
        top_idx = set(np.argsort(slate["scores"])[::-1][:k].tolist())
        for i, g in enumerate(groups):
            if g is None:
                continue
            total[g] += 1
            if i in top_idx:
                hit[g] += 1
    return {g: (hit[g], total[g]) for g in total}


def run_fairness(
    *,
    encoder: TextEncoder,
    model_path: Path,
    expected_model_sha256: str,
    n_users: int = 10_000,
    n_test_queries: int = 1_000,
    n_candidates: int = 20,
    seed: int = 42,
    k: int = 10,
    config_path: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    """SHA가 확인된 모델로 결정적 test split의 DP·활동량 편향을 측정한다."""

    model_path = Path(model_path)
    if out_path is not None and out_path.resolve(strict=False) == model_path.resolve(strict=False):
        raise ValueError("fairness report must not overwrite the model")
    model_bundle = load_trusted_model_bundle(
        model_path,
        expected_model_sha256=expected_model_sha256,
    )
    return _run_fairness_with_bundle(
        encoder=encoder,
        model_path=model_path,
        expected_model_sha256=expected_model_sha256,
        model_bundle=model_bundle,
        n_users=n_users,
        n_test_queries=n_test_queries,
        n_candidates=n_candidates,
        seed=seed,
        k=k,
        config_path=config_path,
        out_path=out_path,
    )


def _run_fairness_with_bundle(
    *,
    encoder: TextEncoder,
    model_path: Path,
    expected_model_sha256: str,
    model_bundle: dict[str, Any],
    n_users: int,
    n_test_queries: int,
    n_candidates: int,
    seed: int,
    k: int,
    config_path: Path | None,
    out_path: Path | None,
) -> dict[str, Any]:
    config_path = config_path or DEFAULT_CONFIG_PATH
    model, columns = model_bundle["model"], model_bundle["columns"]

    tags = load_allowed_tags(config_path)
    policy = MatchingInputPolicy(allowed_tag_ids=frozenset(tags))
    as_of = GenerationConfig().as_of

    # 학습(train_match_v2.run)과 동일한 생성·split·test 페어 구성.
    users = generate_population(GenerationConfig(n_users=n_users, seed=seed), policy=policy)
    _train_users, test_users = split_users(users, test_ratio=0.2, seed=seed)
    test_pairs = build_pairs(
        test_users,
        n_queries=n_test_queries,
        n_candidates=n_candidates,
        weights=LabelWeights(),
        seed=seed + 1,
    )

    slates = _scored_slates(
        test_users,
        test_pairs,
        model=model,
        columns=columns,
        encoder=encoder,
        policy=policy,
        as_of=as_of,
    )

    # 1. ui_mode DP (빈 ui_mode는 제외)
    ui_rates = _selection_rates(slates, lambda s: [m if m else None for m in s["ui_mode"]], k)
    ui_sel = {g: hit / total for g, (hit, total) in ui_rates.items() if total > 0}
    dp_diff = (max(ui_sel.values()) - min(ui_sel.values())) if len(ui_sel) >= 2 else 0.0

    # 2. 활동량 편향 (전체 후보 콘텐츠 수 중앙값 기준 상/하)
    all_activity = [a for s in slates for a in s["activity"]]
    median = float(np.median(all_activity)) if all_activity else 0.0
    act_rates = _selection_rates(
        slates, lambda s: ["high" if a > median else "low" for a in s["activity"]], k
    )
    act_sel = {g: hit / total for g, (hit, total) in act_rates.items() if total > 0}
    act_gap = abs(act_sel.get("high", 0.0) - act_sel.get("low", 0.0))

    result = {
        "model": {
            "path": str(model_path),
            "sha256": expected_model_sha256,
        },
        "k": k,
        "n_test_slates": len(slates),
        "ui_mode_selection_rate": ui_sel,
        "ui_mode_dp_diff": dp_diff,
        "ui_mode_dp_gate": DP_GATE,
        "ui_mode_dp_pass": dp_diff <= DP_GATE,
        "activity_median": median,
        "activity_selection_rate": act_sel,
        "activity_gap": act_gap,
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _report_overlaps(path: Path, protected: list[Path]) -> bool:
    report = path.resolve(strict=False)
    for item in protected:
        resolved = item.resolve(strict=False)
        if report == resolved or (item.is_dir() and report.is_relative_to(resolved)):
            return True
    return False


def _print(result: dict[str, Any]) -> None:
    print(f"test slates = {result['n_test_slates']}  (top-K={result['k']})")
    print("[ui_mode 선택률]")
    for g, r in sorted(result["ui_mode_selection_rate"].items()):
        print(f"  {g:14s} {r:.4f}")
    print(
        f"  DP diff = {result['ui_mode_dp_diff']:.4f}  (게이트 ≤ {result['ui_mode_dp_gate']}) → "
        f"{'PASS' if result['ui_mode_dp_pass'] else 'REVIEW'}"
    )
    print("[활동량 편향]")
    for g, r in sorted(result["activity_selection_rate"].items()):
        print(f"  {g:14s} {r:.4f}")
    print(f"  high-low gap = {result['activity_gap']:.4f}  (중앙값 {result['activity_median']})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument(
        "--expected-model-sha256",
        help="--model 직접 지정 시 필수인 외부 SHA-256; current pointer 사용 시 선택 검증값",
    )
    parser.add_argument(
        "--current-pointer",
        type=Path,
        default=ROOT / "artifacts" / "match_v2_current.json",
        help="--model 미지정 시 사용할 완료 run 포인터",
    )
    parser.add_argument("--n-users", type=int, default=10_000)
    parser.add_argument("--test-queries", type=int, default=1_000)
    parser.add_argument("--candidates", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "reports" / "validation_reports" / "module2" / "fairness_v2.json",
    )
    args = parser.parse_args(argv)

    protected: list[Path]
    if args.model is not None:
        if args.expected_model_sha256 is None:
            parser.error("--model requires --expected-model-sha256")
        model_path = args.model
        model_sha256 = args.expected_model_sha256
        protected = [model_path]
    else:
        try:
            current = load_current_match_run(args.current_pointer)
        except ValueError as exc:
            parser.error(f"invalid current MATCH run: {exc}")
        model_path = current.model_path
        model_sha256 = current.model_sha256
        if args.expected_model_sha256 is not None and args.expected_model_sha256 != model_sha256:
            parser.error("--expected-model-sha256 differs from current pointer")
        protected = [args.current_pointer, current.run_dir]

    if _report_overlaps(args.out, protected):
        parser.error("--out must not overwrite or reside inside current model artifacts")

    try:
        model_bundle = load_trusted_model_bundle(
            model_path,
            expected_model_sha256=model_sha256,
        )
    except ValueError as exc:
        parser.error(f"invalid MATCH model: {exc}")

    from scripts.train_match_v2 import _build_sbert

    encoder = _build_sbert(DEFAULT_CONFIG_PATH)
    result = _run_fairness_with_bundle(
        encoder=encoder,
        model_path=model_path,
        expected_model_sha256=model_sha256,
        model_bundle=model_bundle,
        n_users=args.n_users,
        n_test_queries=args.test_queries,
        n_candidates=args.candidates,
        seed=args.seed,
        k=args.k,
        config_path=None,
        out_path=args.out,
    )
    _print(result)
    print(f"saved: {args.out}")
    return 0 if result["ui_mode_dp_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
