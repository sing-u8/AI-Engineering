---
category: foundation-models
title: "04. 트랜스포머 블록 구조(Transformer Block Architecture)와 모델 크기"
source: "AI Engineering · Chapter 2 (p.62-63)"
tags: [transformer-block, mlp, activation, relu, gelu, llama-specs]
---

# 04. 트랜스포머 블록 구조와 모델 크기 결정 요인

## 📌 핵심 요약
> **트랜스포머 LLM은 `[임베딩 모듈] ➔ [어텐션 + MLP 블록 N개] ➔ [출력 헤드]`의 샌드위치 구조로 이루어져 있으며, 모델 차원, 블록 수, MLP 차원, 어휘 크기에 따라 총 파라미터 수가 결정됩니다.**

---

## 1. 트랜스포머의 전체 아키텍처 구조

```
[입력 텍스트]
      │
      ▼
① 임베딩 모듈 (Embedding Module) : 토큰 임베딩 + 위치(Positional) 임베딩
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ ② 트랜스포머 블록 (Transformer Block)                   │ ◀── (이 블록을 N개 반복 = 모델의 레이어 수)
│   ├─ 어텐션 모듈 (Attention Module) : W_Q, W_K, W_V, W_O │
│   └─ MLP / 피드포워드 모듈 (MLP Module) : Linear + 비선형 활성화 │
└────────────────────────────────────────────────────────┘
      │
      ▼
③ 출력 레이어 (Unembedding / Model Head) : 다음 단어 확률 분포 계산
      │
      ▼
[출력 텍스트]
```

---

## 2. 핵심 구성 모듈 상세 분석

### (1) 트랜스포머 블록 내부 (Transformer Block)

1. **어텐션 모듈 (Attention Module):**
   * $W_Q, W_K, W_V, W_O$ 총 4개의 가중치 행렬로 구성.
   * 문장 내 단어들 간의 관계와 문맥을 파악.
2. **MLP 모듈 (Multi-Layer Perceptron / Feedforward Layer):**
   * 선형 레이어(Linear Layer) 사이에 **비선형 활성화 함수(Activation Function)**를 배치.
   * 복잡한 패턴을 학습하기 위해 선형성을 깨뜨리는 역할.
   * **대표 활성화 함수:**
     * **ReLU:** $\text{ReLU}(x) = \max(0, x)$ (음수는 0으로, 양수는 그대로 통과).
     * **GELU:** GPT-2, GPT-3 등에서 널리 활용된 부드러운 형태의 활성화 함수.
   * *(참고)*: 복잡한 활성화 함수를 쓰더라도 성능 개선은 미미하고 연산/메모리 비용만 급증하므로 단순하고 빠른 함수를 사용하는 것이 표준입니다.

### (2) 블록 앞뒤의 모듈 (입출력)

1. **임베딩 모듈 (Embedding Module - 앞단):**
   * **토큰 임베딩:** 단어 번호를 연속적인 벡터 공간으로 매핑.
   * **위치 임베딩 (Positional Embedding):** 단어의 순서 정보를 부여. 트랜스포머는 단어들을 동시에 처리하므로 위치 정보가 필수적입니다.
   * **🚀 최신 위치 임베딩 기법 (RoPE & ALiBi):**
     * 초기 트랜스포머는 고정된 절대 위치(Absolute) 임베딩을 사용해 문맥 길이가 제한되는 한계가 있었습니다.
     * **RoPE (Rotary Position Embedding):** 단어 벡터를 회전(Rotation)시키는 수학적 연산을 통해 '상대적 거리'를 인코딩합니다. Llama 등 최신 모델의 표준 기법이며, 보지 못한 더 긴 문맥(Context)에 대해서도 유연하게 외삽(Extrapolation)할 수 있는 길을 열었습니다.
     * **ALiBi:** 어텐션 점수 계산 시 두 단어의 거리에 비례하여 페널티를 부여하는 직관적 방식으로 무한에 가까운 문맥 확장을 지원합니다.
2. **출력층 (Output / Unembedding Layer / Model Head - 뒷단):**
   * 최종 은닉 상태 벡터를 어휘 사전(Vocab) 크기의 로짓(Logit)으로 변환하여, 소프트맥스를 통해 다음 단어 생성 확률을 산출.

---

## 3. 모델 크기(파라미터 수)를 결정하는 4대 하이퍼파라미터

1. **Model Dimension ($d_{\text{model}}$):** 블록 내부 벡터들의 기본 차원 (예: 4,096).
2. **# Transformer Blocks ($L$):** 트랜스포머 블록(레이어)을 쌓은 층수 (예: 32층, 80층).
3. **Feedforward Dimension ($d_{\text{ff}}$):** MLP 내부에서 일시적으로 확장하는 차원 크기 (보통 $d_{\text{model}}$의 수 배).
4. **Vocabulary Size ($V$):** 모델이 인식하고 다룰 수 있는 고유 단어(토큰)의 총 개수.

---

## 4. Llama 2 vs Llama 3 하이퍼파라미터 스펙 비교 (Table 2-4)

| 모델 | 블록 수 ($L$) | 모델 차원 ($d_{\text{model}}$) | 피드포워드 차원 ($d_{\text{ff}}$) | 어휘 크기 (Vocab) | 문맥 길이 (Context) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Llama 2-7B** | 32 | 4,096 | 11,008 | 32K | 4K |
| **Llama 2-13B** | 40 | 5,120 | 13,824 | 32K | 4K |
| **Llama 2-70B** | 80 | 8,192 | 22,016 | 32K | 4K |
| **Llama 3-7B** | 32 | 4,096 | 14,336 | **128K** | **128K** |
| **Llama 3-70B** | 80 | 8,192 | 28,672 | **128K** | **128K** |
| **Llama 3-405B** | 126 | 16,384 | 53,248 | **128K** | **128K** |

### 🔍 주목할 포인트:
* **Llama 3의 압도적 확장:** 어휘 사전 크기를 4배(32K ➔ 128K)로 키워 다국어 및 코드 표현력을 극대화했고, 문맥 길이를 32배(4K ➔ 128K)로 늘림.
* **⚠️ 주의 (중요):** **문맥 길이(Context Length)가 늘어나면 추론 시 메모리(VRAM)는 많이 먹지만, 모델 자체의 총 정적 파라미터 수(Total Weight Parameters)는 증가하지 않습니다.**

---

## 🔗 연관 문서
* [[03-attention-mechanism-and-math|03. 어텐션 메커니즘과 수학적 수식]]
* [[05-parameters-tokens-flops-and-moe|05. 파라미터와 연산량(FLOPs)]]
