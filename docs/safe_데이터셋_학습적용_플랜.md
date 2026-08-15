# SAFE 데이터셋 학습 적용 플랜

> 대상: 모듈① 이진(정상/주의) 분류기 재학습
> 후보 근거: [safe_데이터셋_확장_후보.md](safe_데이터셋_확장_후보.md) · 파이프라인 규약: [safe_이진_재학습_평가_파이프라인.md](safe_이진_재학습_평가_파이프라인.md)
> 작성 2026-07-09 · 성격: **실행 플랜** (다운로드·어댑터·주입·재학습 절차)

---

## 0. 공통 주입 메커니즘 (모든 데이터셋 동일)

새 데이터셋은 코드 수정 없이 **jsonl 어댑터 + config 키**만으로 학습에 붙는다. 트레이너([src/training/trainer.py](../src/training/trainer.py))에 이미 두 경로가 있다.

| 주입 경로 | config 키 | 동작 | 적합 소스 |
|---|---|---|---|
| **A. 위험 전량 주입** | `data.extra_caution_jsonl` + `extra_caution_filter` | 필터 통과 행을 **전량 label=1(주의)** 로 병합 ([trainer.py:100](../src/training/trainer.py#L100)) | 위험만 있는 소스: PAN12/그루밍, DKTC 위협4종, 보이스피싱, PsyScam |
| **B. 라벨 보존 주입** | `data.extra_train_jsonl` + `extra_train_repeat` | `{text, label:0/1, source}` 를 **라벨 그대로** 병합·반복 ([trainer.py:138](../src/training/trainer.py#L138)) | 정상/위험 혼재 소스: K-MHaS, APEACH, 번역 대조셋 |

**어댑터 산출 스키마** (둘 중 하나로 변환):
- A용: `{"text": <한국어>, "source": <이름>, "split_role": <필터키>, ...}`
- B용: `{"text": <한국어>, "label": 0|1, "source": <이름>}`

**공통 파이프라인 흐름** (노트북 09~11 템플릿 그대로):
```
어댑터(raw→jsonl) → (외국어면) translate_pan12.py식 현지화 번역 → data/synthetic/<name>/
  → build_final_dataset.py (MinHash 0.8 dedup, 홀드아웃 분리)
  → config에 extra_* 키 추가 → train_module1.py --config ...
  → dev 게이트 → fresh blind 1회 소비
```
재사용 자산: [scripts/download_seed_datasets.py](../scripts/download_seed_datasets.py)(자동 다운로드 패턴), [scripts/extract_pan12.py](../scripts/extract_pan12.py)(대화 window 추출), [scripts/translate_pan12.py](../scripts/translate_pan12.py)(Gemini 현지화 번역·재개), [src/data/dedup.py](../src/data/dedup.py)(MinHash), [scripts/build_safe_hardcase_dataset.py](../scripts/build_safe_hardcase_dataset.py).

---

## 1. 다운로드 분류 (핵심)

### 🟢 A. 자동 — 스크립트/CLI로 즉시, **사용자 개입 불필요**

| 데이터셋 | 획득 방법 | 상태 |
|---|---|---|
| UnSmile, KOLD | `download_seed_datasets.py` (GitHub raw) | 이미 자동화됨 |
| **DKTC** | `git clone https://github.com/tunib-ai/DKTC` 또는 raw CSV | 공개, 즉시 |
| **K-MHaS** | `load_dataset("jeanlee/kmhas_korean_hate_speech")` | HF 공개, 즉시 |
| **APEACH** | `load_dataset("jason9693/APEACH")` | HF 공개, 즉시 |
| PAN12 | Zenodo record 3713280 zip | 이미 확보(190건) |

### 🟡 B. 반자동 — **사용자가 인증/토큰 1회 제공** 후 자동

| 데이터셋 | 필요한 사용자 조치 | 이후 |
|---|---|---|
| **Kaggle SuicideWatch** (232k) | Kaggle 계정의 `kaggle.json` API 토큰을 `~/.kaggle/`에 배치 | `kaggle datasets download`로 자동 |
| **PsyScam** | 익명 repo URL(`anonymous.4open.science/r/PsyScam-66E4`)이 유효한지 확인 후 zip 링크 전달 | 스크립트로 다운로드 |
| Voice-Phishing GitHub | repo 라벨·구조 적합성 1회 확인 | git clone 자동 |

### 🔴 C. 사용자 직접 — **로그인·승인·연구자 신청 필수 (자동화 불가)**

| 데이터셋 | 이유 | 사용자가 할 일 |
|---|---|---|
| **AI-Hub 텍스트 윤리검증** (45만) | 회원가입 + 데이터 사용신청 **승인**, 한국 실명계정, 공개 API 없음 | aihub.or.kr 로그인 → 신청·승인 → 수동 다운로드 → `data/raw/`에 배치 |
| **보이스피싱 벤치마크** (HLT 2025) | 논문 부속·FSS 자료, 공개 배포처 불명 | 저자/FSS 문의로 확보 → `data/raw/`에 배치 |
| **ChatCoder2 / PANC / eSPD** (그루밍) | "연구 커뮤니티 회원에게만 제공", April Edwards 연락 필요 | chatcoder.com/data.html·GitLab 통해 **연구자 신청** → 승인분 배치 |
| **UMD Reddit Suicidality** | 공식 데이터 사용 신청(IRB형 절차) | 신청·승인 후 배치 (대안: Kaggle SuicideWatch가 공개라 우선) |

> 요약: **한국어 네이티브 3종(DKTC·K-MHaS·APEACH)은 100% 자동**이라 즉시 착수 가능. 그루밍 확장·AI-Hub·보이스피싱만 사용자 직접 다운로드가 필요하다.

---

## 2. 데이터셋별 학습 적용 플랜

각 항목: **다운로드 → 어댑터/라벨 매핑 → 주입 경로 → 번역 → 실행**.

### 2-1. DKTC (협박·갈취·괴롭힘) 🟢 최우선
- **다운로드**: `git clone tunib-ai/DKTC` → `data/raw/dktc/`. (신규 `scripts/fetch_dktc.py` 또는 `download_seed_datasets.py`에 URL 추가)
- **어댑터**: 대화(train.csv 4클래스)를 발화/윈도우로 분해 → `{text, source:"dktc", split_role:"threat"}`. test의 `일반`은 사용 안 함(정상은 시드 clean 유지).
- **주입**: 경로 A — `extra_caution_jsonl: [data/synthetic/dktc.jsonl]`, `extra_caution_filter: {split_role: threat}` → 전량 주의.
- **번역**: 불필요(한국어).
- **주의**: 대화 단위 → 발화 분해 시 무해 턴 라벨 노이즈 완화 위해 window 방식(`extract_pan12.py` 로직 참고). CC-BY-NC-SA 준수.

### 2-2. K-MHaS / APEACH (혐오 보강) 🟢
- **다운로드**: HF `datasets.load_dataset` (자동).
- **어댑터**: K-MHaS 멀티라벨 → `Not Hate=label 0`, 그 외 = `label 1`. APEACH도 동일 이진 붕괴. → `{text, label, source}`.
- **주입**: 경로 B — `extra_train_jsonl: [data/synthetic/kmhas.jsonl, apeach.jsonl]`, `extra_train_repeat: 1`.
- **번역**: 불필요. **정상/위험 양쪽 라벨을 모두 제공**해 번역투 편향 없이 밸런스 보강.

### 2-3. 그루밍 확장 (ChatCoder2/PANC) 🔴→번역
- **다운로드**: 연구자 신청(사용자 직접) → `data/raw/chatcoder2/`.
- **어댑터**: `extract_pan12.py --predator-only` 확장 적용(같은 window 로직) → predator window.
- **주입**: 경로 A — 기존 `pan12_translated.jsonl`과 동일 스키마로 합류(`split_role:predator`).
- **번역**: 필요 — `translate_pan12.py`(현지화·재개) 그대로 재사용.
- **윤리**: 아동 범죄 원데이터 → git 커밋 금지, `data/` gitignore 하위, 재배포 금지.

### 2-4. 금전 사기 🟡/🔴→일부 번역
- **보이스피싱 벤치마크(🔴)**: 사용자 확보 → 통화체를 메신저체로 정규화 어댑터 → 경로 A(전량 주의).
- **PsyScam(🟡, 번역)**: 730건 사기범 메시지 → `translate_pan12.py`식 현지화(달러→원 등) → 경로 A.
- **주의**: 정상 부정문 오탐(규칙 레이어 OFF 사유) 재발 방지 위해, 사기 주입 후 warm_normal·삭제/환급 정상 회귀셋으로 FP 감시.

### 2-5. 자해/자살 (Kaggle SuicideWatch) 🟡→번역
- **다운로드**: Kaggle API(사용자 토큰) → 자동.
- **어댑터**: SuicideWatch=위험 후보. 게시글체 → 대화체 변환 + 번역.
- **주입**: 경로 A(주의) 또는 별도 정책.
- **⚠ 정책 선결**: 자해 신호는 "차단"이 아니라 "도움 연계"가 서비스상 맞을 수 있음 → **라벨링 방향(주의 포함 여부)을 제품 정책 확정 후** 투입. 임상 민감·라벨 노이즈 검수 필수.

### 2-6. 강압적 통제 🔴→번역 + 합성 병행
- **다운로드**: Reddit 심리학대 6종(논문 부속) — 확보 난이도 높음.
- **현실안**: 공개셋 희소 → **`build_safe_hardcase_dataset.py` 합성 하드케이스(이미 v1~v5 존재) 확장**을 주력으로, 번역분은 소량 보조.
- **주입**: 합성=경로 B(라벨 보존), 번역분=경로 A.

---

## 3. 권장 실행 순서 (라운드 단위, 노트북 09~11 방식)

방법론 규약: **라운드당 fresh blind 1회 소비**, 소비분은 dev 격하, blind 원문 학습 미병합.

1. **R1 — 한국어 즉시 트랙 (DKTC + K-MHaS + APEACH)**: 다운로드 자동, 번역 0. 협박·갈취·혐오 공백을 한 번에. 신규 config `configs/module1_binary_hardcases_v6.yaml`(v5 ⊇ + extra_caution(dktc) + extra_train(kmhas,apeach)). fresh blind v8로 검증.
2. **R2 — 사기 트랙**: (사용자) 보이스피싱 확보 + (자동) PsyScam 번역. warm_normal/삭제·환급 FP 집중 감시. blind v9.
3. **R3 — 그루밍 확장**: (사용자) ChatCoder2 신청분 도착 시 번역 병합. blind v10.
4. **R4 — 자해 트랙**: 제품 정책 확정 후. blind v11.
5. 강압통제는 합성 하드케이스로 상시 병행.

각 라운드는 dev 게이트(실데이터 recall≥0.80·specificity≥0.80, dev recall≥0.85·specificity≥0.90, 타깃 슬라이스≥0.80, 규칙보조 OFF) 통과분만 채택. 회귀 시 미채택(v5 사례처럼).

---

## 4. 리스크·주의 (전 라운드 공통)

- **홀드아웃 오염 금지**: 신규 소스 전량 `--forbidden`으로 소비 blind + 실 홀드아웃(AI-Hub/BEEP)과 MinHash(0.8) 교차 dedup 후 학습. 번역분은 train 전용.
- **번역투 편향**: 위험만 번역 주입하면 "번역체=주의" 지름길 학습 위험. 정상 대화도 동일 파이프라인 소량 번역 + 문체 대조 미니셋 모니터링([grooming_번역_증강_계획.md](grooming_번역_증강_계획.md) 규약). → K-MHaS/APEACH(한국어 정상 포함)가 이 편향의 균형추.
- **라이선스**: DKTC(NC)·K-MHaS/APEACH(BY-SA)·그루밍 원데이터(재배포 금지)·AI-Hub(약관) 각각 준수. 상업화 전환 시 NC/SA 전면 재검토.
- **자해 정책 선결**: 2-5 참고 — 라벨링 방향 미확정 상태로 투입 금지.

## 5. 검증 방법 (실행 시)

1. 어댑터 산출 jsonl 스키마 검사: 경로 A는 `text`+필터키, 경로 B는 `text`+`label∈{0,1}`. (`load_extra_binary_train`이 label 오류 시 raise → 자동 방어)
2. `build_final_dataset.py` 로그에서 `시드중복 N제거`·붕괴 후 분포·홀드아웃 무누수 확인.
3. 학습 후 dev 게이트 통과 여부 + fresh blind 1회(SHA256 전후 검증) 결과로 채택 판정.

---

## 6. 실행 기록 (2026-07-09, R1 한국어 즉시 트랙)

R1(DKTC + K-MHaS + APEACH)을 **로컬에서 실제 실행**했다. 셋 다 한국어라 번역 불필요.

### 6.1 실제로 수행·검증된 것 (로컬 macOS)

| 단계 | 결과 | 산출물 |
|---|---|---|
| 다운로드 | DKTC(3,950 대화, GitHub), K-MHaS(78,977, HF), APEACH(11,666, HF) | `data/raw/dktc/`, HF 캐시 |
| 어댑터 + dedup | [scripts/adapt_external_datasets.py](../scripts/adapt_external_datasets.py) | `data/synthetic/{dktc,kmhas,apeach}.jsonl` |
| **누수 차단 실측** | **APEACH에서 홀드아웃/blind와 exact 중복 1,409건 제거** (APEACH↔BEEP 겹침 확인), K-MHaS·DKTC near-dup 각 1건 | — |
| config | [configs/module1_binary_hardcases_v6.yaml](../configs/module1_binary_hardcases_v6.yaml) | v5 ⊇ + 신규 3소스 |
| **주입 검증** | 트레이너 실제 로더(`load_extra_caution`/`load_extra_binary_train`/`collapse_binary`)로 확인, 라벨 도메인 {0,1} 무결 | [scripts/verify_v6_ingestion.py](../scripts/verify_v6_ingestion.py) |

**어댑터 산출 규모** (홀드아웃/blind dedup 후):
- DKTC 15,000 (윈도우 3턴, 전량 주의) · K-MHaS 14,999 (정상 8,184 / 주의 6,815, 15k 층화표본) · APEACH 10,256 (정상 4,729 / 주의 5,527)

**v6 신규 주입 합계**: 42,605 (정상 13,889 / 주의 28,716). base(README 실측 정상 19,941 / 주의 28,386)와 합산 시 **정상 33,830 / 주의 57,102 / 총 90,932** (기존 ~48k → 약 1.9배).

> ⚠ **분포 리스크**: DKTC가 전량 주의(15k)라 신규 주입이 67% 주의로 치우침 → 합산 주의 비율 63%(기존 59%)로 상승. 과플래그(specificity 하락) 가능 → dev 게이트의 specificity·warm_normal 회귀로 반드시 감시. 필요 시 DKTC 표본 축소 또는 K-MHaS 정상 비중 확대.

### 6.2 로컬에서 못 한 것 (정직 보고)

**모델 재학습은 이 macOS 환경에서 실행 불가** — 두 하드 블로커:
1. **transformers 임포트 불가**: venv의 `huggingface_hub==1.22.0`이 `transformers 4.46.3`(요구 `<1.0`)과 충돌. 로컬 pip이 환경 버그로 `huggingface_hub` 0.17+ 를 전부 "다른 파이썬 버전 필요"로 오필터 → 호환 버전 설치 실패.
2. **GPU 없음**: CUDA 미지원(MPS만). 학습은 원래 Colab A100 전제(README·노트북 09~11).

→ 데이터 준비·주입 검증까지 로컬 완료, **학습은 아래 Colab 절차로 실행**한다.

### 6.3 Colab 재학습 절차 (프로덕션, A100)

노트북 09~11과 동일 패턴. config·어댑터·검증 스크립트는 커밋되어 repo에 있음.

```bash
# 0) repo 클론·브랜치·의존성(requirements-colab.txt), data_bundle.zip 업로드·해제
# 1) 신규 3소스 어댑터 (인터넷 필요 — DKTC/HF 자동 다운로드 + dedup)
python scripts/adapt_external_datasets.py
# 2) base 데이터셋 빌드 (신규 top-level jsonl은 build_final이 무시 — 검증됨)
python scripts/download_seed_datasets.py
python scripts/build_processed_dataset.py
python scripts/build_final_dataset.py --synth-repeat 1 --include-aihub-train
# 3) v6 재학습 (extra_caution=pan12 predator, extra_train=하드케이스+dktc+kmhas+apeach)
python scripts/train_module1.py --config configs/module1_binary_hardcases_v6.yaml
# 4) fresh blind v8 1회 소비(SHA256 전후) — v1~v7은 소비되어 dev 회귀셋
python scripts/evaluate_safe_blind.py --model models/checkpoints/module1_binary_hardcases_v6 \
  --data tests/fixtures/safe_blind_v8.jsonl   # v8은 신규 작성 필요
```
채택 기준: dev 게이트(실 recall≥0.80·specificity≥0.80, dev recall≥0.85·specificity≥0.90, 타깃 슬라이스≥0.80, 규칙보조 OFF) 통과 + blind v8 결과. 회귀 시 미채택.

> **다음 라운드(R2 사기, R3 그루밍, R4 자해)**는 §1-C의 사용자 직접 다운로드분이 도착한 뒤 같은 어댑터 패턴(+`translate_pan12.py` 현지화 번역)으로 진행.

---

## 7. 사용자 직접 다운로드 링크 (R2~R4)

DKTC·K-MHaS·APEACH(R1)와 AI-Hub 텍스트 윤리검증은 **이미 확보/자동**이라 대상 아님.
(AI-Hub 윤리검증은 `data/raw/aihub_558/147.텍스트 윤리검증 데이터`로 로컬 존재.)

아래는 **직접 받으셔야** 하는 것들. 받은 원본은 `data/raw/<name>/`에 두면 어댑터가 처리.

### R2 — 금전 사기
| 데이터 | 링크 | 접근 | 번역 |
|---|---|---|---|
| 보이스피싱 벤치마크 (HLT 2025) | https://www.koreascience.kr/article/CFKO202533636030737.view | 논문 → **저자/FSS 문의**(금감원 보이스피싱 체험관 자료) | 불필요(한국어) |
| Voice-Phishing-Detection-App/ML | https://github.com/Voice-Phishing-Detection-App/ML | GitHub 공개 (git clone) | 불필요 |
| PsyScam | https://anonymous.4open.science/r/PsyScam-66E4 · 논문 https://arxiv.org/abs/2505.15017 | 공개 repo zip | **필요**(영어) |

### R3 — 그루밍 (아동 성범죄 원데이터 — git 커밋·재배포 금지)
| 데이터 | 링크 | 접근 | 번역 |
|---|---|---|---|
| eSPD-datasets (PANC/VTPAN 전처리) | https://gitlab.com/early-sexual-predator-detection/eSPD-datasets | 연구용, 안내 따름 | **필요**(영어) |
| ChatCoder2 | https://www.chatcoder.com/data.html | **연구자 신청**(April Edwards) | **필요**(영어) |

### R4 — 자해/자살 (⚠ 제품 정책: 차단 vs 도움 연계 확정 후 투입)
| 데이터 | 링크 | 접근 | 번역 |
|---|---|---|---|
| Kaggle Suicide & Depression Detection (232k) | https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch | Kaggle 계정(kaggle.json) | **필요**(영어) |
| UMD Reddit Suicidality v2 | https://psresnik.github.io/umd_reddit_suicidality_dataset.html | **신청**(resnik@umd.edu, AAS 심사) | **필요**(영어) |

### R5(보조) — 강압적 통제
공개 라벨셋 희소 → 별도 다운로드보다 `scripts/build_safe_hardcase_dataset.py` 합성 하드케이스 병행 권장. 참고 논문: 관계학대 탐지 AAAI(https://ojs.aaai.org/index.php/AAAI/article/download/35294/37449), IPV 심리학대 Reddit(https://doi.org/10.1177/08944393261435087).
