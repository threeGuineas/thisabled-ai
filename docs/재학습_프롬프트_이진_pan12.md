# 재학습 실행 프롬프트 — SAFE 이진 전환 + PAN12 반영 (Colab)

아래 블록을 Colab(또는 GPU 환경)의 AI 세션에 그대로 붙여 사용한다. 저장소는 thisabled-ai,
브랜치는 `feature/grooming-augmentation`(PAN12·이진 문서·스크립트가 여기 있음).

---

```
# 임무: ThisAbled SAFE 모듈①을 "이진(정상/주의)"으로 재학습하고 HF에 재업로드

## 배경 (반드시 먼저 읽을 문서)
- docs/safe_라벨_및_판정_기준.md  ← 4-class→이진(정상/주의) 전환 결정과 근거 (정본)
- docs/grooming_번역_증강_계획.md ← PAN12 predator 번역 증강, normal 미사용 결정
- docs/label_mapping.md          ← 시드→4-class 매핑 상세 (붕괴 규칙의 출발점)
현 상태: 데이터·config는 아직 4-class(0/1/2/3). config num_labels=4.
직전 업로드된 HF 모델(soyuncj/thisabled-safety-kcelectra)은 이전 재학습본 —
이번 작업으로 이진 모델로 교체한다.

## 목표
- KcELECTRA-base-v2022 파인튜닝, **num_labels=2** (0=정상, 1=주의).
- 재현: seed=42. Stratified split. MLflow 로깅. Colab이면 완료 즉시 Drive 백업.

## 라벨 재정식화 (핵심) — 4-class를 이진으로 붕괴
모든 소스의 label을 다음으로 매핑한 뒤 학습:
    0(정상) → 0
    1(주의)·2(경고)·3(긴급) → 1(주의)
적용 대상: 시드(Unsmile·KOLD), 합성, AI-Hub(data/eval/aihub_train.jsonl), PAN12.
검증: 붕괴 후 클래스 분포를 출력(정상:주의 대략 4:6 예상). 극단 불균형이면 중단·보고.

## 데이터 구성 (ablation 3종)
공통: 홀드아웃은 실데이터만(AI-Hub real, beep_real_holdout) — 합성·PAN12·번역분은
train 전용, 홀드아웃에 절대 포함 금지. MinHash 근사중복 제거(src/data/dedup.py)로
train↔holdout 및 소스 간 누수 차단(코드로 확인 후 학습).
- (a) 시드 + 합성
- (b) (a) + AI-Hub 실데이터(aihub_train.jsonl)
- (c) (b) + PAN12 predator 번역분
    · 입력: data/synthetic/pan12_translated.jsonl 에서 split_role=="predator" 만 → 라벨 1(주의)
      (normal은 사용 금지 — 계획서 ③ 발견)
    · 현재 확보 190건(무료 티어 쿼터로 250 중 190). 쿼터 리셋 후
      `python scripts/translate_pan12.py` 재실행으로 나머지 보충 가능(재개 로직 있음).
    · '정상' 보강은 PAN12가 아니라 시드 clean(Unsmile clean, KOLD OFF=False)에서.

## 실행 흐름
1. python scripts/build_processed_dataset.py          # 시드 split 재생성
2. python scripts/build_final_dataset.py --synth-repeat 8   # (긴급 oversample은 이진에선
   불필요할 수 있음 — 붕괴 후 분포 보고 --synth-repeat 1~2로 낮추는 것 검토)
3. **이진 학습 — 준비 완료됨**: `configs/module1_binary.yaml`과 trainer 이진 경로가 이미 있음.
   그대로 실행: `python scripts/train_module1.py --config configs/module1_binary.yaml`
   config의 `data.binary: true`가 train/val 라벨 {1,2,3}→1 붕괴를 수행하고,
   `data.extra_caution_jsonl`(pan12_translated.jsonl, split_role==predator만)를 label=1로 병합.
   metrics도 이진(caution_recall·normal_precision·macro_f1)으로 자동 분기.
   ※ PAN12 predator 현재 190건 — 쿼터 리셋 후 translate 재실행하면 자동으로 더 병합됨(경로 동일).
   ※ 붕괴 후 분포가 출력됨(`[binary] train label 분포`) — 확인만.
4. 평가 (두 관점 모두 보고):
   a) argmax 기준: 홀드아웃 주의(1) Recall, 정상(0) Precision, F1, 혼동행렬
   b) 서빙 임계값 기준: P(주의) ≥ 0.50(성인)/0.35(미성년) → flagged, 홀드아웃 flagged율
   c) **그루밍 하위셋 Recall**: 홀드아웃 중 성적 유인·그루밍 상당 샘플만 별도 집계
   d) **번역투 편향 점검**: 번역 predator(주의) vs 한국어 원어 정상만의 미니 대조셋에서
      모델이 문체가 아닌 내용으로 구분하는지 (정상 오탐률 확인)

## 규약
- seed 42. synthetic_holdout·real_holdout은 학습 절대 미포함(코드로 재확인 후 시작).
- 수치를 지어내지 말 것. 각 단계 실행 로그와 실측치만 보고.
- Colab: 학습 종료 즉시 models/checkpoints/module1_binary를 Drive에 복사(런타임 유실 방지).
- 저장 확인: config.json, model.safetensors, tokenizer 일체 존재 + num_labels==2.

## 성공 기준 / 폴백
- 홀드아웃 주의 Recall ≥ 0.80 이고 정상 Precision이 과블러 수준 아님 → 성공.
- (c)가 (b)보다 그루밍 하위셋 Recall을 유의하게 올리는지 확인(PAN12 효과 입증 포인트).
- 편향 점검에서 정상 오탐이 급증하면 → 정상 측 소량 번역 투입 검토(후속), 또는 PAN12 혼합비
  하향(5%→2%). 각 시도 결과 기록.

## 완료 후 HF 재업로드
hf auth login   # write 토큰
hf upload soyuncj/thisabled-safety-kcelectra models/checkpoints/module1_binary . \
  --repo-type model --private \
  --exclude "optimizer.pt" --exclude "rng_state.pth" --exclude "scheduler.pt" --exclude "training_args.bin"
→ 기존 repo에 새 커밋으로 덮어씀(이진 모델로 교체).

## 서빙 반영 (재업로드 후, 별도)
serving/safety_server/app.py는 4-class 가정(P(주의+경고+긴급)). 이진 모델에선:
  - risk_prob = P(주의)  (인덱스 1) 한 줄로 단순화
  - LABEL_NAMES = ["정상","주의"]
docs/safe_라벨_및_판정_기준.md §3 참조. 규칙 보조 레이어(금전 사기)는 유지.
```

---

## 참고: 이번 재학습이 이전과 다른 점

| 항목 | 이전(누수 수정본) | 이번 |
|---|---|---|
| 출력 | 4-class (0/1/2/3) | **이진 (정상/주의)** |
| 긴급 데이터 | 합성 only (실전이 실패 recall 0.005) | +AI-Hub 실데이터 +PAN12 predator 번역 |
| 불균형 처리 | 긴급 oversample×8, α=5 | 붕괴로 불균형 대폭 완화 — 가중 최소화 |
| 서빙 매핑 | P(1+2+3)≥τ | P(주의)≥τ |
