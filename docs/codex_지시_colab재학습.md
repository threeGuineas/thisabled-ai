# Codex 지시 프롬프트 — Colab 이진 재학습 실행·감독

로컬 Codex(또는 코딩 에이전트)에게 아래를 그대로 지시한다. 목적은 **노트북을 대신 짜는 게
아니라**, 이미 준비된 `notebooks/06_retrain_binary_pan12.ipynb`와 `configs/module1_binary.yaml`을
Colab에서 **실행·감독하고 결과를 검증**하는 것.

---

```
너는 ThisAbled SAFE 모듈①의 이진(정상/주의) 재학습을 Colab에서 실행·감독한다.
코드/설정은 이미 브랜치 feature/grooming-augmentation에 준비돼 있다. 새로 짜지 말고 실행·검증하라.

## 준비물 확인 (없으면 사용자에게 요청, 임의 생성·추측 금지)
- 저장소 접근: GITHUB_TOKEN (private repo)
- 데이터(git 밖, Drive에서 복사): data/synthetic/ (합성 + pan12_translated.jsonl),
  data/eval/ (aihub_train.jsonl, aihub_real_holdout.jsonl, beep_real_holdout.jsonl)
- HF 업로드: HF_TOKEN (write)
합성 데이터가 없으면 재생성하지 말고 중단·보고 (OpenAI 비용). Drive 경로를 사용자에게 물어라.

## 실행 (notebooks/06_retrain_binary_pan12.ipynb 순서대로)
1. GPU 런타임 확인. 저장소 clone + 브랜치 checkout + pull.
2. 의존성 설치, CUDA 확인.
3. 데이터 배치 확인: data/synthetic·data/eval 비어있으면 중단·보고.
   pan12_translated.jsonl 존재·건수 확인(현재 predator 190).
4. build_processed_dataset.py → build_final_dataset.py --synth-repeat 1
5. train_module1.py --config configs/module1_binary.yaml
   → 로그에서 반드시 확인·보고:
     · "[binary] train label 분포" (정상:주의 비율) — 극단 불균형이면 중단·보고
     · "[extra_caution] 병합 N건" (PAN12 predator가 실제 병합됐는지)
6. 홀드아웃 평가 셀 실행. 다음을 표로 보고:
     · argmax(τ=0.5): 주의 Recall, 정상 Precision, macro-F1, 혼동행렬
     · 서빙 임계값 τ=0.5(성인)/0.35(미성년) 각각의 주의 Recall·정상 Precision
7. Drive 백업 후 HF 재업로드(hf upload, 추론 파일만 --exclude로 학습상태 제외).

## 판정 기준 / 폴백
- 성공: 홀드아웃 주의 Recall ≥ 0.80 AND 정상 Precision 과블러 아님(예: ≥ 0.6).
- PAN12 효과: 시간이 되면 config에서 extra_caution_jsonl을 비운 ablation과 비교해
  주의(특히 그루밍성 문장) Recall이 오르는지 확인. 안 되면 결과만 기록하고 진행.
- 미달 시 임의 튜닝 금지. 다음만 순서대로 1~2회 시도하고 각 결과 기록:
  (a) loss.alpha를 [1.0, 1.2~1.5]로 주의 가중, (b) build_final_dataset --synth-repeat 2.
- 여전히 미달이면 표와 함께 중단·보고.

## 절대 규칙
- seed=42 고정. 홀드아웃(실데이터)은 학습에 절대 미포함 — 코드로 재확인 후 학습.
- 수치를 지어내지 말 것. 각 셀의 실제 출력만 인용해 보고.
- Colab 런타임 유실 대비: 학습 종료 즉시 체크포인트를 Drive에 복사한 뒤 평가·업로드.
- HF 업로드는 num_labels==2 확인 후에만. 업로드 후 사용자에게 "서빙 컨테이너 재시작 시
  자동으로 이진 모델 로드(/health의 num_labels로 확인)"라고 안내.

## 완료 보고 형식
- train label 분포 / 병합 건수 / 홀드아웃 지표표(argmax·τ별) / 혼동행렬 /
  HF 업로드 커밋 URL / 다음 액션(서빙 반영).
```

---

## 참고 — 이 트랙에서 이미 끝난 것 (Codex가 다시 안 해도 됨)

| 항목 | 상태 |
|---|---|
| PAN12 추출·번역 파이프라인 | 완료 (`scripts/extract_pan12.py`, `translate_pan12.py`) |
| 이진 config·trainer 경로 | 완료 (`configs/module1_binary.yaml`, `src/training/trainer.py`) |
| 서빙 이진·4-class 자동 지원 | 완료 (`serving/safety_server/app.py`) |
| Colab 노트북 | 완료 (`notebooks/06_retrain_binary_pan12.ipynb`) |
| **실제 재학습 실행·HF 교체** | **← Codex가 할 일** |
