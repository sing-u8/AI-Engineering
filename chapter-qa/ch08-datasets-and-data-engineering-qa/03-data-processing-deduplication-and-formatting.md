---
category: datasets-and-data-engineering
title: "03. 데이터 탐색, 중복 제거 및 포맷팅 엔지니어링 (pp. 391-404)"
source: "AI Engineering · Chapter 8 (p.391-404)"
tags: [data-processing, eda, deduplication, minhash, lsh, perplexity, data-filtering, chat-templates, prompt-loss-masking]
---

# 03. 데이터 탐색, 중복 제거 및 포맷팅 엔지니어링

## 📌 핵심 요약 & 전체 맥락
> **"데이터를 모으는 것보다 중요한 것은, 쓰레기 데이터와 중복을 가려내고 모델이 소화할 수 있는 완벽한 형태로 다듬는 엔지니어링 파이프라인입니다."**  
> 데이터 전처리 단계에서는 먼저 **탐색적 데이터 분석(EDA, Exploratory Data Analysis)**을 통해 동사-명사 분포(Figure 8-6)와 응답 길이 편향(Figure 8-7)을 진단합니다.  
> 이어 학습 데이터셋에 중복(Duplicates)이 존재할 때 모델이 특정 출력으로 심각하게 편향되는 문제를 해결하기 위해, 대규모 텍스트 유사도를 고속 계산하는 **MinHash 및 LSH (Locality-Sensitive Hashing)** 알고리즘을 적용합니다.  
> 마지막으로 모델이 질문(프롬프트)을 외우느라 불필요한 과적합을 일으키지 않도록 지시 영역의 역전파 손실을 0으로 깎아내는 **프롬프트 손실 마스킹(Prompt Loss Masking)**과 토크나이저 일치 검증을 완벽히 정리합니다.

---

## 🗺️ 도표(Figure & Table) 색인 및 매핑 가이드

본문의 핵심 원리를 시각적으로 확인할 수 있는 책 내 도표의 위치와 해설 매핑입니다:

| 도표 번호 | 도표 제목 및 핵심 내용 | 책 페이지 | 본문 해당 주제 |
| :---: | :--- | :---: | :--- |
| **Figure 8-6** | Self-Instruct 지시 데이터셋의 핵심 동사-명사 조합을 시각화한 선버스트(Sunburst) 차트 | **p. 392** | 1. 탐색적 데이터 분석 (EDA) |
| **Figure 8-7** | GPT-4(더 상세하고 긴 응답)와 GPT-3(짧은 응답)의 토큰 길이 분포 비교 그래프 | **p. 394** | 1. 응답 길이 편향 분석 |
| **Table 8-3** | 중복 데이터가 포함된 예시(예: 음식 분류에서 피자만 10번 반복)로 인해 모델이 모든 입력에 피자라고 답하는 편향 실증 | **p. 396** | 2. 중복 데이터의 위험성과 MinHash |
| **Table 8-4** | 감성 분석 및 음식 분류 태스크의 입력-출력 포맷팅 및 프롬프트 손실 마스킹 예시표 | **p. 402** | 4. 포맷팅 및 프롬프트 손실 마스킹 |

---

## 1. 탐색적 데이터 분석 (EDA, Exploratory Data Analysis, pp. 391 ~ 395)

1. **지시 동사-명사 선버스트 차트 (Figure 8-6):**  
   지시문의 핵심 루트 동사(`Write`, `Explain`, `Extract`, `Calculate`)와 목적어 명사(`Code`, `Email`, `Summary`)의 조합 빈도를 분석하여, 특정 행동(예: 요약에만 80% 치우침)에 데이터셋이 편중되지 않았는지 균형을 점검.
2. **응답 길이 편향 (Length Bias, Figure 8-7):**  
   언어 모델은 **"긴 응답이 더 유익하다"**는 강한 휴리스틱 편향을 학습하기 쉽습니다. GPT-4는 GPT-3보다 평균 응답 길이가 1.5~2배 길며, 짧고 명확한 단답형을 요구하는 태스크 데이터도 함께 배합하여 길이 편향을 교정해야 합니다.

---

## 2. 중복 데이터의 위험성과 MinHash LSH 중복 제거 (pp. 395 ~ 398)

* ⚠️ **중복 데이터의 치명적 부작용 (Table 8-3):**  
  학습 데이터 1,000개 중 `"오늘 저녁 뭐 먹지? ➔ 피자"`라는 데이터가 10번 중복되어 들어가면, 모델의 내부 확률 분포가 왜곡되어 사용자가 *"점심 추천해줘"*, *"야식 추천해줘"*라고 물어도 **모든 질문에 오직 "피자"만 앵무새처럼 답하는 과적합 편향**이 발생합니다.

```
[ MinHash + LSH (Locality-Sensitive Hashing) 고속 중복 제거 원리 ]

1. Shingling  : 텍스트를 N-gram 단어 조각(예: 5-gram)의 집합으로 쪼갬
2. MinHash    : 수백 개의 독립 해시 함수를 적용하여 각 문서의 최소 해시값들을 뽑아 
                작은 서명(Signature Vector, 예: 128차원)으로 압축
                ➔ 자카드 유사도 Jaccard(A, B) ≈ MinHash 서명 일치 확률
3. LSH Bucket : 유사한 서명을 가진 문서들을 같은 해시 버킷(Bucket)에 매핑하여 
                수억 개 문서 간 O(N²) 전수비교를 O(N) 선형 시간으로 고속 중복 제거
```

---

## 3. 텍스트 품질 필터링 기법 (pp. 398 ~ 401)

1. **휴리스틱 필터 (Heuristics):**  
   * 글자 수가 10자 미만이거나 10,000자 초과인 이상치 제거.
   * 특수문자나 숫자 비율이 50% 이상인 깨진 데이터 필터링.
   * 개인 식별 정보(**PII, Personally Identifiable Information**) 정규식 마스킹.
2. **퍼플렉서티 기반 필터 (Perplexity Filtering, PPL):**  
   참조 언어 모델(KenLM 등)을 통과시켜 텍스트의 당혹도(Perplexity)를 측정.  
   * **PPL이 너무 높은 데이터:** 스팸, 오타 도배, 난독화 텍스트 ➔ **제거**
   * **PPL이 너무 낮은 데이터:** `"1 1 1 1 1..."`처럼 의미 없이 단어가 무한 반복되는 기계 생성 데이터 ➔ **제거**

---

## 4. 학습 포맷팅과 프롬프트 손실 마스킹 (Prompt Loss Masking, pp. 401 ~ 404) ⭐

SFT(지도 미세조정) 학습 시, 모델의 예측 손실(Cross-Entropy Loss)을 계산할 때 **사용자 지시문(Prompt) 영역은 손실 가중치를 0으로 마스킹(Prompt Loss Weight = 0)**해야 합니다:

```
[ 프롬프트 손실 마스킹 (Prompt Loss Masking) 구조 ]

입력 텍스트 시퀀스 : "<|im_start|>user\n프랑스의 수도는?<|im_end|>\n<|im_start|>assistant\n파리입니다.<|im_end|>"
역전파 손실 계산   : [        손실 가중치 = 0 (Masked)         ] [    손실 가중치 = 1.0 (Loss 계산)    ]
```

* **이유:**  
  우리의 학습 목표는 모델이 사용자의 질문을 외우게 만드는 것이 아니라, **질문을 보고 정답 응답(Assistant Response)을 올바르게 생성하는 조건부 확률 $P(\text{Response} \mid \text{Prompt})$만을 학습**시키는 것이기 때문입니다. 프롬프트까지 손실을 계산하면 모델이 질문의 패턴에 쓸데없이 과적합되어 제로샷 일반화 능력이 저하됩니다.

---

## 🔗 연관 문서
* [[00-ch08-overview|00. Chapter 8 전체 개요 및 목차]]
* [[01-data-curation-quality-coverage-and-quantity|01. 데이터 큐레이션: 품질, 다양성 및 데이터 규모]]
* [[02-data-augmentation-and-synthesis|02. 데이터 증강과 AI 합성 데이터 (Self-Instruct & Model Collapse)]]
* [[chapter-qa/ch07-fine-tuning-qa/01-finetuning-foundations-and-decision-framework|Ch07-01. 파인튜닝 기초와 엔지니어링 의사결정 프레임워크]]
