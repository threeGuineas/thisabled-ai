# SAFE 이진 모델 — 재학습·평가 반복 파이프라인

SAFE 이진(KcELECTRA, `num_labels=2`, 라벨 `["정상","주의"]`) 모델을 **평가 → 약점 파악 → hard-case 증강 → 재학습 → blind 평가** 루프로 개선하는 구조. 각 라운드는 노트북 한 개로 실행되며, GPU 단계는 Colab A100에서 사용자가 커널을 켜고 돌린다.

- 템플릿 노트북: [notebooks/08_iterative_hardcase_retrain.ipynb](../notebooks/08_iterative_hardcase_retrain.ipynb) (현행 프로덕션 모델을 만든 라운드)
- 최신 라운드: [notebooks/09_iterative_hardcase_retrain_v3.ipynb](../notebooks/09_iterative_hardcase_retrain_v3.ipynb)

---

## 1. 데이터 흐름

```
시드(AIHub/BEEP/PAN12) ──► build_processed_dataset.py ──► data/processed/train.parquet
                                                              │
합성 긴급(emergency/*) ──► build_final_dataset.py ────────────┤  (MinHash 0.8 중복 제거)
   --synth-repeat N --include-aihub-train                      │
                                                              ▼
hard-case base 문장 ──► build_safe_hardcase_dataset.py ──► data/synthetic/safe_hardcases_v*/train.jsonl
   (--include-v2/--include-v3, --forbidden blind)              │  (extra_train_repeat 배로 학습 시 병합)
                                                              ▼
                                              train_module1.py --config configs/module1_binary_hardcases_v*.yaml
                                                              ▼
                                              models/checkpoints/module1_binary_hardcases_v*_r{repeat}
```

- 실데이터 holdout: [data/eval/aihub_real_holdout.jsonl](../data/eval), `beep_real_holdout.jsonl` — 학습에 절대 병합 안 함. 실데이터 회귀(일반화) 확인용.
- `data/synthetic/safe_hardcases_v*/train.jsonl` 은 **.gitignore 대상** — 노트북이 Colab에서 스크립트로 재생성한다(커밋하지 않음).

## 2. 핵심 스크립트

| 스크립트 | 역할 |
| --- | --- |
| [scripts/build_processed_dataset.py](../scripts/build_processed_dataset.py) | 시드 train/val/test (StratifiedGroupKFold 80/10/10, seed=42) |
| [scripts/build_final_dataset.py](../scripts/build_final_dataset.py) | 합성 데이터 oversample 병합 + MinHash(0.8) 중복 제거 |
| [scripts/build_safe_hardcase_dataset.py](../scripts/build_safe_hardcase_dataset.py) | hard-case 결정론적 생성 + blind 누수 가드 |
| [scripts/train_module1.py](../scripts/train_module1.py) | KcELECTRA fine-tune (config 구동) |
| [scripts/evaluate_safe_blind.py](../scripts/evaluate_safe_blind.py) | 고정 blind 셋 평가 (`/analyze` 계약, 규칙 OFF 가능) |
| [src/data/dedup.py](../src/data/dedup.py) | MinHash LSH 근사 중복 판정 (`find_duplicate_indices`) |

### hard-case 생성 규칙 (build_safe_hardcase_dataset.py)
- 손으로 작성한 **base 문장**(NORMAL_BASES/RISK_BASES + v2/v3)을 `PREFIXES × SUFFIXES` 로 조합 확장 → 정규화 dedup.
- 버전은 **누적**: `--include-v3` ⇒ v1+v2+v3 전부 포함.
- `--forbidden <blind.jsonl>` 로 지정한 blind 문장과 **exact + near-dup(0.8)** 겹치면 `RuntimeError`. 라운드마다 소비된 blind 전부를 forbidden 으로 넘긴다.
- v3 타깃: blind v4 회귀 약점(`fraud_credentials`·`coercive_control` recall 0.75, 정당한 삭제/환급 안내 FP).

## 3. blind 방법론 (정직 평가의 핵심)

1. 라운드마다 **새 fresh blind** 를 손으로 작성한다(40행, 스키마 `{id,label,slice,receiver_is_minor,text}`, 균형 20/20, 슬라이스 10×4).
2. 후보 모델을 **고정한 뒤 정확히 1회만** blind 를 평가하고, 평가 전후 **SHA-256 해시**로 원문 불변을 검증한다.
3. 소비된 blind 는 다음 라운드부터 **dev 회귀셋**으로 격하 — 최종 blind 로 재사용 금지.
4. blind 원문은 **train 에 절대 병합하지 않는다**. hard-case·train 과 exact+near(0.8) 중복 0 을 노트북에서 단언한다.
5. 한계: blind 도 저자 생성 합성셋이라 *패턴 일반화*를 측정하며 완전 독립 실데이터가 아니다. 실데이터 회귀는 aihub/beep holdout 이 담당.

blind 이력: v1~v5 소비 완료(dev 회귀셋). 현재 fresh = **[tests/fixtures/safe_blind_v6.jsonl](../tests/fixtures/safe_blind_v6.jsonl)** ([notebooks/10_iterative_hardcase_retrain_v4.ipynb](../notebooks/10_iterative_hardcase_retrain_v4.ipynb)에서 소비).

## 4. 재학습·게이트 (노트북 셀 순서)

1. A100·저장소 확인 + `requirements-colab.txt` 설치
2. `data_bundle.zip` 업로드(실데이터 holdout — git 미포함)
3. 데이터 빌드 + 누수 가드 (hard-case↔blind, fresh blind↔train/hard-case)
4. **dev 평가 함수** — dev 회귀셋 = 실데이터 holdout + 소비된 blind 전부. threshold 스윕(성인 0.40~0.85, 미성년=성인−0.16, 최소 0.35). **규칙 보조 OFF**.
5. **repeat 1→2→3 재학습**, dev 게이트 통과 시 즉시 중단. 게이트: real_recall≥.80, real_spec≥.80, dev_recall≥.85, dev_spec≥.90, 타깃 슬라이스 recall(fraud·coercive·grooming·extortion)≥.80.
6. 후보 고정 후 **fresh blind 1회** 평가(해시 검증). 게이트: risk_recall≥.80, specificity≥.90, 슬라이스 recall 최소≥.75.
7. **HF 업로드(기본 OFF)** — 켜면 추론 파일만 올림(`ignore_patterns` 로 학습상태 자동 제외).

- 규칙 보조 OFF 이유: 운영이 `SAFE_RULE_ASSIST=0`. 규칙 레이어가 정상 부정문을 FP 처리해 비활성화됨. 코드는 [serving/safety_server/app.py](../serving/safety_server/app.py).
- 슬라이스 이름은 버전별로 다름(grooming: `grooming_secrecy/isolation`, `grooming`, `grooming_threat`) → 접두 매칭으로 흡수.

## 5. 서빙 반영

업로드 성공 시 새 HF 커밋 SHA 를 서빙 env 에 고정한다.

```
SAFE_MODEL_DIR=soyuncj/thisabled-safety-kcelectra
SAFE_MODEL_REVISION=<새 HF 커밋 SHA>
SAFE_FLAG_THRESHOLD=<성인 임계값>
SAFE_FLAG_THRESHOLD_MINOR=<미성년 임계값>
SAFE_RULE_ASSIST=0
```

- `/health` 가 실제 로드된 커밋을 `revision` 으로 보고(코드가 `SAFE_MODEL_REVISION` 소비). 배포 후 이 값이 새 SHA 와 일치하는지 확인.
- 현행 프로덕션 revision: `79bbd16e2ea9a5c9133fb01c6f8c1c09671283aa`, 임계값 0.85/0.69.

## 6. 새 라운드 추가 체크리스트

1. 직전 blind 결과에서 약한 슬라이스/오판 패턴 식별.
2. `build_safe_hardcase_dataset.py` 에 `NORMAL_V{n}_BASES`/`RISK_V{n}_BASES` + `--include-v{n}` 추가(약점 정조준, blind 와 어휘 구분).
3. `--include-v{n}` 로 로컬 생성해 소비된 blind 전부 대비 exact+near(0.8) **0** 확인.
4. **새 fresh blind v{m}** 작성 → train·hard-case·이전 blind 전부와 중복 **0** 검증(스키마는 `evaluate_safe_blind.load_cases` 규격).
5. `configs/module1_binary_hardcases_v{n}.yaml` (extra_train_jsonl 을 새 hard-case 로).
6. **기존 노트북 수정 금지** — 08/09 를 템플릿으로 새 노트북 작성. dev 회귀셋에 새로 소비된 blind 추가, fresh blind 1회 소비.
7. Colab 실행 → 게이트 통과 시 HF 업로드 → 서빙 revision 갱신.
