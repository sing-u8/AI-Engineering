---
category: fine-tuning
title: "04. 모델 병합(Model Merging)과 가중치 산술 연산 (pp. 342-358)"
source: "AI Engineering · Chapter 7 (p.342-358)"
tags: [model-merging, slerp, ties-merging, dare, task-arithmetic, frankenmoe, solar-10-7b, depthwise-scaling, lora-merging]
---

# 04. 모델 병합(Model Merging)과 가중치 산술 연산

## 📌 핵심 요약 & 전체 맥락
> **"단 1원의 추가 GPU 학습 비용도 들이지 않고, 여러 전문 모델의 지능을 단 하나의 모델로 결합할 수 있습니다."**  
> 전통적인 앙상블(Ensemble) 기법은 $N$개의 모델을 동시에 띄워야 하므로 서빙 메모리와 연산 비용이 $N$배로 폭증합니다.  
> 반면 **모델 병합 (Model Merging)**은 동일한 베이스 모델에서 파생된 여러 파인튜닝 모델의 **가중치 행렬 자체를 산술 연산(Weight Arithmetic)하여 단 하나의 모델로 합성**하는 혁신적인 제로 GPU 학습(Zero-GPU Training) 패러다임입니다.  
> 본 섹션에서는 고차원 초구면 보간법인 **SLERP (Spherical Linear Interpolation)**, 가중치 충돌을 제거하는 **TIES-Merging**과 **DARE**, 여러 특화 모델을 전문가로 묶는 **FrankenMoE (Mixture of Experts)**, 그리고 32개 레이어를 48개로 늘려 10.7B 모델을 탄생시킨 Upstage의 **Depthwise Scaling (Solar-10.7B)**까지 모델 병합의 최신 기법을 총망라합니다.

---

## 🗺️ 도표(Figure & Table) 색인 및 매핑 가이드

본문의 핵심 원리를 시각적으로 확인할 수 있는 책 내 도표의 위치와 해설 매핑입니다:

| 도표 번호 | 도표 제목 및 핵심 내용 | 책 페이지 | 본문 해당 주제 |
| :---: | :--- | :---: | :--- |
| **Figure 7-13** | 인퍼런스 비용이 배가되는 앙상블(Ensemble) vs 단일 모델로 합치는 모델 병합(Model Merging) 비교 | **p. 343** | 1. 앙상블 vs 모델 병합 |
| **Figure 7-14** | 모델 병합 3대 접근법 (가중치 산술 연산, 깊이 스태킹, 라우터 기반 MoE 결합) | **p. 345** | 2. 모델 병합 3대 기법 분류 |
| **Figure 7-15** | 동일한 구조의 두 모델 레이어 가중치를 단순 산술 평균하는 선형 결합 구조 | **p. 346** | 3. 선형 평균과 한계 |
| **Figure 7-16** | 고차원 초구면 상에서 두 벡터의 최단 호를 따라 각도를 보간하는 SLERP 기하학적 원리 | **p. 347** | 4. 구면 선형 보간 (SLERP) |
| **Figure 7-17** | 태스크 벡터의 하위 80%를 제거(가지치기)해도 성능이 유지되는 TIES-Merging/DARE 곡선 | **p. 349** | 5. TIES-Merging 및 DARE |
| **Figure 7-18** | 복수의 파인튜닝 모델을 전문가(Experts)로 묶고 라우터를 붙여 MoE로 변환하는 FrankenMoE | **p. 351** | 6. FrankenMoE 전문가 결합 |
| **Figure 7-19** | 32개 레이어를 중첩 연결하여 48개 레이어로 확장하는 Depthwise Scaling (Solar-10.7B) | **p. 353** | 7. Depthwise Scaling (Solar-10.7B) |
| **Figure 7-20** | 서로 다른 LoRA 어댑터 행렬을 결합하여 랭크 $r_1 + r_2$의 통합 어댑터를 만드는 LoRA 병합 | **p. 355** | 8. LoRA 어댑터 가중치 병합 |

---

## 1. 앙상블 vs 모델 병합 (Figure 7-13, pp. 342 ~ 344)

```
[ 앙상블 (Ensemble) vs 모델 병합 (Model Merging) 비교 ]

• 앙상블 (Ensemble)     : 3개 모델 띄움 ➔ 3번 추론 실행 ➔ 답변 3개 투표
                          (서빙 GPU 메모리 3배 소모 💸, 지연시간 3배 증가)
• 모델 병합 (Model Merging): 가중치 파일 3개를 수학적으로 합쳐 1개 모델 생성 ➔ 1번만 실행
                          (서빙 비용 1배 그대로 유지 🚀, 추가 GPU 학습 불필요)
```

---

## 2. 가중치 산술 연산과 SLERP (Figure 7-15, 7-16, pp. 345 ~ 348)

### ① 선형 평균(Linear Average)과 태스크 산술 (Task Arithmetic)
* **태스크 벡터 (Task Vector, $\tau$):** 파인튜닝 모델 가중치 $W_{\text{FT}}$에서 베이스 모델 가중치 $W_{\text{Base}}$를 뺀 변화량 ($\tau = W_{\text{FT}} - W_{\text{Base}}$).
* **능력 더하기/빼기:**  
  $$W_{\text{Merged}} = W_{\text{Base}} + \lambda_1 \tau_{\text{Math}} + \lambda_2 \tau_{\text{Code}} - \lambda_3 \tau_{\text{Toxicity}}$$  
  수학 능력과 코딩 능력을 더하고, 독성(유해성)을 빼는 산술 연산이 실제로 작동합니다!

### ② 구면 선형 보간 (SLERP, Spherical Linear Interpolation, Figure 7-16) ⭐
고차원 가중치 공간에서 단순 선형 평균($\frac{W_1 + W_2}{2}$)을 내면 **가중치 벡터의 크기(Norm)가 줄어들어 모델이 멍청해지는 축소 현상**이 발생합니다.
* **기하학적 원리:** 두 가중치 벡터 사이를 고차원 초구면(Hypersphere) 상의 최단 호(Arc)를 따라 일정한 각속도로 회전 보간하여 **벡터의 크기와 방향성을 100% 보존**합니다:

$$\text{SLERP}(W_1, W_2; t) = \frac{\sin((1-t)\theta)}{\sin\theta} W_1 + \frac{\sin(t\theta)}{\sin\theta} W_2$$

---

## 3. 간섭 제거 병합: TIES-Merging & DARE (Figure 7-17, pp. 348 ~ 351)

여러 모델을 합칠 때 서로 다른 모델이 동일한 가중치를 정반대 방향(+$0.5$ vs -$0.5$)으로 수정하여 0으로 상쇄되는 **부호 충돌(Sign Interference)**이 발생합니다:

1. **TIES-Merging (TrIm, Elect Sign, and Merge):**  
   * **Trim (가지치기):** 변화량이 미미한 하위 80% 가중치를 0으로 리셋.
   * **Elect Sign (부호 투표):** 여러 모델 간 부호 충돌 발생 시 다수결로 전체 부호 통일.
   * **Merge (병합):** 살아남은 가중치들만 평균 합산.
2. **DARE (Drop And REscale):**  
   태스크 가중치의 90% 이상을 무작위(Random Dropout)로 0으로 날려버리고 남은 가중치를 $\frac{1}{1-p}$로 스케일업하여 파괴적 간섭을 원천 차단.

---

## 4. 구조 확장: FrankenMoE & Solar-10.7B Depthwise Scaling (Figures 7-18, 7-19)

### ① FrankenMoE (Figure 7-18)
코딩 특화 Llama, 의료 특화 Llama, 수학 특화 Llama를 각각 전문가(**Experts**)로 묶고, 입력 프롬프트를 보고 어떤 전문가에게 보낼지 결정하는 게이팅 라우터(Router)를 부착하여 **비공식 MoE (Mixture of Experts) 모델**을 즉석 합성.

### ② Upstage Solar-10.7B의 Depthwise Scaling (Figure 7-19)
7B 모델의 32개 레이어를 단순히 복사하여 중간 16개 레이어를 중첩 연결(32 ➔ 48 레이어)함으로써, **10.7B 크기로 확장한 뒤 가벼운 후속 파인튜닝만으로 당시 13B~30B 모델들을 벤치마크에서 압도**한 한국의 대표적 성공 사례.

---

## 🔗 연관 문서
* [[00-ch07-overview|00. Chapter 7 전체 개요 및 목차]]
* [[01-finetuning-foundations-and-decision-framework|01. 파인튜닝 기초와 엔지니어링 의사결정 프레임워크]]
* [[03-peft-lora-and-qlora|03. 매개변수 효율적 파인튜닝(PEFT)과 LoRA / QLoRA]]
* [[chapter-qa/ch09-inference-optimization-qa/03-speculative-decoding-caching-and-parallelism|Ch09-03. 추측 디코딩, 프롬프트 캐싱 및 분산 병렬화]]
