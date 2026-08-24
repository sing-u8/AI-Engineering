---
category: foundation-models
title: "03. 어텐션 메커니즘(Attention Mechanism)과 수학적 공식 완벽 분해"
source: "AI Engineering · Chapter 2 (p.60-61)"
tags: [attention, transformer, q-k-v, math, multi-head-attention]
---

# 03. 어텐션 메커니즘(Attention Mechanism)과 수학적 수식 분해

## 📌 핵심 요약
> **어텐션(Attention)은 다음 단어를 예측할 때 "이전 단어들 중 누구에게 얼마나 집중(Attention)할 것인가?"를 계산하는 메커니즘으로, Query(질문), Key(색인), Value(내용) 간의 내적과 소프트맥스 가중합으로 동작합니다.**

---

## 1. 직관적인 비유: 도서관(유튜브) 검색 시스템

| 개념 | 비유 (유튜브/도서관) | 책 요약 비유 (본문) | 수학적/AI적 의미 |
| :--- | :--- | :--- | :--- |
| **Query ($Q$)** | 검색창에 입력한 **검색어** | 정보를 찾는 사람의 **질문** | 지금 단어를 만들기 위해 **"찾고자 하는 정보"** |
| **Key ($K$)** | 동영상들의 **제목 및 태그** | 책의 **페이지 번호/목차** | 과거 단어들이 가진 **"대표 라벨/특징"** |
| **Value ($V$)** | 동영상의 **실제 영상 내용** | 책의 **실제 본문 내용** | 과거 단어들이 가진 **"실제 정보/의미"** |

---

## 2. 어텐션의 공식 수학 수식 (Scaled Dot-Product Attention)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

### 📐 각 기호의 정의와 행렬 크기 (Shape)
단어(토큰) 개수가 $N$개, 각 벡터의 차원 크기가 $d_k$일 때:

* **$Q \in \mathbb{R}^{N \times d_k}$**: Query 행렬 (현재 조회하려는 벡터들)
* **$K \in \mathbb{R}^{N \times d_k}$**: Key 행렬 (대조할 대상 벡터들)
* **$K^T \in \mathbb{R}^{d_k \times N}$**: Key의 전치행렬 (행렬곱을 위해 뒤집음)
* **$d_k$**: Key 벡터의 차원 수 (스칼라 값, 예: 128)
* **$\sqrt{d_k}$**: 스케일링 인자 (Scaling Factor)
* **$V \in \mathbb{R}^{N \times d_v}$**: Value 행렬 (실제 정보가 담긴 벡터들)

---

## 3. 수식의 단계별 연산 과정 (Step-by-Step)

```
[입력 Q, K, V]
      │
      ▼
1단계: S = Q × K^T          (모든 단어 쌍의 내적 유사도 점수 계산: N × N 행렬)
      │
      ▼
2단계: S_scaled = S / √d_k   (차원 크기로 나누어 분산 안정화)
      │
      ▼
3단계: A = softmax(S_scaled) (각 행의 합이 1이 되도록 확률 가중치 변환)
      │
      ▼
4단계: Output = A × V        (가중치대로 Value 벡터들을 가중합)
```

### ① 1단계: 유사도 점수 행렬 계산 ($Q K^T$)
* $(N \times d_k) \times (d_k \times N) = N \times N$ 행렬이 생성됩니다.
* $i$번째 단어의 Query($q_i$)와 $j$번째 단어의 Key($k_j$) 간의 **벡터 내적(Dot Product)**:
  $$S_{ij} = q_i \cdot k_j$$
* 두 벡터가 유사한 방향을 가리킬수록(관련성이 높을수록) $S_{ij}$ 점수가 커집니다.

### ② 2단계: 스케일링 ($\frac{1}{\sqrt{d_k}}$)
* **왜 $\sqrt{d_k}$로 나눌까?**
  * 차원($d_k$)이 커질수록 내적 값의 크기(절댓값)가 지나치게 커집니다.
  * 큰 값이 소프트맥스에 들어가면 출력이 $0$ 또는 $1$로 극단화되어 **기울기 소실(Gradient Vanishing)**이 발생하므로, 분산을 $1$로 정규화해 학습을 안정화합니다.

### ③ 3단계: 가중치 확률 변환 ($\text{softmax}$)
* 각 행(Row)별로 소프트맥스를 취해 점수를 **합이 1(100%)인 확률 가중치**로 변환합니다:
  $$A_{ij} = \frac{\exp\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right)}{\sum_{m=1}^{N} \exp\left(\frac{q_i \cdot k_m}{\sqrt{d_k}}\right)}$$
* $A_{ij}$는 **"$i$번째 단어가 $j$번째 단어에 기울여야 할 주의(Attention) 비율"**이 됩니다.

### ④ 4단계: 가중합으로 최종 벡터 생성 ($A \times V$)
* $(N \times N) \times (N \times d_v) = N \times d_v$ 행렬을 얻습니다.
  $$\text{Output}_i = \sum_{j=1}^{N} A_{ij} v_j$$
* 각 단어는 중요한 이전 단어들의 Value($v$)를 높은 비율로 흡수한 새로운 문맥 벡터가 됩니다.

---

## 4. 멀티헤드 어텐션(Multi-Head Attention)과 Llama 2-7B 수치 분석

문맥을 하나의 관점으로만 보면 단어 간의 다양한 관계를 놓칠 수 있습니다. 따라서 여러 개의 헤드(Head)로 쪼개어 동시에 다양한 각도에서 문맥을 살핍니다.

* **Hidden Dimension ($d_{\text{model}}$) = 4,096:** 단어 1개를 4,096개의 숫자로 표현.
* **Attention Heads = 32개:** 32명의 전문가(헤드)가 동시에 작동.
* **Head Dimension ($d_k$) = $4,096 / 32 = 128$:** 
  * 4,096차원을 32개로 쪼개어, 각 전문가는 128차원 공간에서 각자의 관점(문법, 지칭어, 감정 등)으로 어텐션을 수행.
* **Concat & Output Projection ($W_O$):**
  * 32개 헤드의 결과(각 128차원)를 다시 옆으로 이어붙여 4,096차원으로 복원한 뒤, $4096 \times 4096$ 크기의 출력 행렬($W_O$)을 곱해 다음 층으로 전달.

### 🌟 [심화] 메모리 병목을 해결하는 어텐션 변형 (MQA & GQA)
기본 멀티헤드 어텐션(MHA)은 모든 헤드마다 각자의 Key와 Value 벡터를 가집니다. 이는 모델 추론 시 **막대한 KV 캐시 메모리 용량과 메모리 대역폭을 소모**하는 주범입니다. 이를 해결하기 위해 최신 모델들은 구조를 변형합니다.
* **다중 쿼리 어텐션 (MQA - Multi-Query Attention):** 모든 Query 헤드가 **단 1개의 Key와 Value 쌍을 전체 공유**합니다. (메모리 사용량 급감 및 속도 대폭 향상되나 품질이 다소 하락함)
* **그룹화 쿼리 어텐션 (GQA - Grouped-Query Attention):** MHA와 MQA의 완벽한 절충안으로, 헤드를 N개의 그룹으로 묶고 **그룹당 1개의 Key/Value 쌍을 공유**합니다. Llama 2 (70B) 및 Llama 3에서 채택된 현대 LLM의 표준 방식으로, MHA급의 품질을 유지하면서 MQA급의 극강의 추론 속도를 달성합니다.

---

## 5. 문맥 길이(Context Length) 확장의 메모리 병목 (KV Cache)

* 단어 수가 늘어날 때마다 모든 단어의 $K, V$ 벡터를 저장하고 계산해야 합니다.
* 문맥 길이가 10배 길어지면 필요한 $K, V$ 저장량(KV Cache)과 연산량이 폭증하므로, 이것이 트랜스포머의 문맥 길이를 무한정 늘리기 어려운 핵심 기술적 이유입니다.

---

## 🔗 연관 문서
* [[04-transformer-block-architecture|04. 트랜스포머 블록 구조]]
* [[05-parameters-tokens-flops-and-moe|05. 파라미터와 연산량(FLOPs)]]
