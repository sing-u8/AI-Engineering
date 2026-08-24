---
category: fine-tuning
title: "02. 학습 메모리 수학과 수치 정밀도 및 양자화 (FP16, BF16, BitNet) (pp. 315-324)"
source: "AI Engineering · Chapter 7 (p.315-324)"
tags: [fine-tuning, memory-math, activation-memory, backpropagation, optimizer-states, fp16, bf16, quantization, bitnet]
---

# 02. 학습 메모리 수학과 수치 정밀도 및 양자화 (FP16, BF16, BitNet)

## 📌 핵심 요약 & 전체 맥락
> **"파인튜닝에서 GPU 메모리가 터지는(OOM, Out Of Memory) 진짜 원인은 모델 가중치가 아니라 옵티마이저 상태(Optimizer States)와 순전파 활성화 메모리(Activations)에 있습니다."**  
> 70억 개(7B) 매개변수를 가진 모델을 16비트(2바이트)로 로드하면 가중치는 단 14GB에 불과하지만, AdamW 옵티마이저를 써서 학습을 시작하는 순간 **가중치(2B) + 그래디언트(2B) + 옵티마이저 상태(12B) = 매개변수당 최소 16바이트**가 필요하여 112GB 이상의 **GPU VRAM (Video Random Access Memory)**이 요구됩니다.  
> 본 섹션에서는 역전파(Backpropagation) 계산 그래프에 기반한 **4대 학습 메모리 수학 공식**, 지수부(Exponent)와 가수부(Fraction)의 비트 구조로 분석하는 **부동소수점 포맷(FP32, FP16, BF16)**, 그리고 가중치를 $\{-1, 0, 1\}$ 삼진수로 극단적 압축한 **BitNet b1.58**의 원리를 완벽히 정리합니다.

---

## 🗺️ 도표(Figure & Table) 색인 및 매핑 가이드

본문의 핵심 원리를 시각적으로 확인할 수 있는 책 내 도표의 위치와 해설 매핑입니다:

| 도표 번호 | 도표 제목 및 핵심 내용 | 책 페이지 | 본문 해당 주제 |
| :---: | :--- | :---: | :--- |
| **Figure 7-4** | 순전파(Forward) 시 활성화값을 저장하고 역전파(Backward) 시 체인 룰을 계산하는 그래프 | **p. 316** | 1. 파인튜닝 메모리의 4대 구성 요소 |
| **Figure 7-5** | 배치 크기와 시퀀스 길이가 증가함에 따라 가중치 메모리를 압도하는 활성화 메모리 비중 | **p. 318** | 2. 활성화 메모리(Activations)의 병목 |
| **Figure 7-6** | 부호(Sign), 지수부(Exponent), 가수부(Fraction)로 구성된 FP32, FP16, BF16 비트 구조도 | **p. 320** | 3. 수치 정밀도와 부동소수점 포맷 |
| **Table 7-3** | 동일한 FP32 수치 값을 FP16, BF16, INT8로 변환할 때 발생하는 반올림 오차 비교표 | **p. 321** | 3. 정밀도 포맷별 오차 및 특징 |
| **Table 7-4** | BitNet b1.58 삼진 모델과 Llama 2 16비트 모델의 연산 속도, 메모리 절감 및 벤치마크 비교표 | **p. 324** | 4. 초저비트 양자화와 BitNet |

---

## 1. 파인튜닝 메모리의 4대 구성 요소와 수학 공식 (pp. 315 ~ 318)

완전 파인튜닝(Full Fine-Tuning)을 수행할 때 모델 가중치 $N$개(예: 7B = $7 \times 10^9$)에 대해 소비되는 정적 메모리는 다음과 같습니다:

```
[ 완전 파인튜닝 시 파라미터당 정적 VRAM 소모량 (AdamW 기준) ]

1. 모델 가중치 (Model Weights, 16-bit FP16/BF16) : 파라미터당 2 바이트
2. 그래디언트 (Gradients, 16-bit FP16/BF16)       : 파라미터당 2 바이트
3. 옵티마이저 상태 (Optimizer States, AdamW 32-bit):
   - FP32 가중치 마스터 카피                     : 파라미터당 4 바이트
   - 1차 모멘텀 (Momentum / Moving Average)      : 파라미터당 4 바이트
   - 2차 모멘텀 (Variance / Squared Average)     : 파라미터당 4 바이트
   ──────────────────────────────────────────────────────────
   합계 (정적 상태 메모리)                        : 파라미터당 16 바이트 (16 Bytes / Param)
```

$$\text{Static Memory (Bytes)} = N \times 16\text{ Bytes}$$

* **7B 모델 기준:** $7 \times 10^9 \times 16\text{ Bytes} \approx 112\text{ GB}$ (A100 80GB GPU 1장으로 학습 불가능!)
* **70B 모델 기준:** $70 \times 10^9 \times 16\text{ Bytes} \approx 1,120\text{ GB}$ (80GB GPU 최소 16장 필요!)

---

## 2. 활성화 메모리(Activations)의 병목과 그래디언트 체크포인팅 (Figure 7-5)

정적 가중치 메모리 외에도, 학습 시 **순전파(Forward Pass) 과정에서 다음 레이어로 전달되는 중간 텐서 값(활성화값)**을 역전파 미분 계산을 위해 메모리에 들고 있어야 합니다:

$$\text{Activation Memory} \propto B \times S \times L \times H \times A$$

* $B$: 배치 크기 (Batch Size)
* $S$: 시퀀스 길이 (Sequence Length)
* $L$: 트랜스포머 레이어 수 (Number of Layers)
* $H$: 은닉 차원 크기 (Hidden Dimension)
* $A$: 어텐션 헤드 수 (Number of Attention Heads)

* **배치 크기와 시퀀스 길이가 늘어날 때 (Figure 7-5):**  
  컨텍스트 길이가 4K에서 32K로 늘어나면 활성화 메모리가 수십 GB 단위로 폭증하여 가중치 메모리보다 훨씬 커집니다.
* 💡 **그래디언트 체크포인팅 (Gradient Checkpointing, 활성화 재계산):**  
  모든 레이어의 활성화값을 메모리에 다 저장하지 않고, 중간중간 체크포인트 레이어만 저장한 뒤, 역전파 시 필요한 활성화값을 순전파로 즉석 재계산합니다. **약 20~30%의 연산 시간이 더 걸리지만 활성화 메모리를 70~80% 이상 극적으로 절감**할 수 있습니다.

---

## 3. 수치 정밀도와 부동소수점 포맷 (Figure 7-6, Table 7-3, pp. 319 ~ 322)

부동소수점 숫자는 **부호(Sign, 1비트)**, **지수부(Exponent, 수의 표현 범위 결정)**, **가수부(Fraction/Mantissa, 수의 정밀도 결정)**의 3요소로 구성됩니다:

```
[ 주요 부동소수점 비트 포맷 구조 (Figure 7-6) ]

• FP32 (32-bit) : [ 부호 1비트 ] [ 지수부 8비트 ] [ 가수부 23비트 ] ➔ 초고정밀도
• FP16 (16-bit) : [ 부호 1비트 ] [ 지수부 5비트 ] [ 가수부 10비트 ] ➔ 지수부가 작아 언더플로/오버플로 취약
• BF16 (16-bit) : [ 부호 1비트 ] [ 지수부 8비트 ] [ 가수부 7비트 ]  ➔ FP32와 동일한 지수부로 안정적 학습
```

| 포맷 | 총 비트 수 | 지수부 (Range) | 가수부 (Precision) | 장단점 및 실무 적용 |
| :--- | :---: | :---: | :---: | :--- |
| **FP32** | 32 bits | 8 bits | 23 bits | 최고 정밀도, 그러나 메모리와 연산 비용이 2배 |
| **FP16** | 16 bits | 5 bits | 10 bits | 가수부가 커서 정밀하지만 지수부가 작아 학습 도중 발산(NaN) 위험 |
| **BF16** | 16 bits | 8 bits | 7 bits | **현대 LLM 학습의 표준 🏆** (Google Brain 개발, 넓은 표현 범위로 수치적 안정성 보장) |

---

## 4. 초저비트 양자화와 BitNet b1.58 (Table 7-4, pp. 323 ~ 324)

* **양자화 (Quantization):**  
  32비트나 16비트 부동소수점 가중치를 8비트 정수(INT8)나 4비트 정수(INT4)로 변환하여 메모리를 $1/2 \sim 1/4$로 압축하고 연산 속도를 가속하는 기법.
* 🚀 **BitNet b1.58 (Microsoft, 2024, Table 7-4):**  
  모든 가중치를 부동소수점이 아닌 **$\{-1, 0, 1\}$ 삼진수(Ternary, 1.58비트)**로만 표현하는 혁신적인 아키텍처.
  * **곱셈 연산의 완전 제거:** 행렬 곱셈($W \times X$)을 비싼 부동소수점 곱셈 대신 **단순 덧셈과 뺄셈(Integer Addition)**으로만 처리하여 에너지 소비를 획기적으로 줄이고 추론 속도를 2~3배 가속합니다.

---

## 🔗 연관 문서
* [[00-ch07-overview|00. Chapter 7 전체 개요 및 목차]]
* [[01-finetuning-foundations-and-decision-framework|01. 파인튜닝 기초와 엔지니어링 의사결정 프레임워크]]
* [[03-peft-lora-and-qlora|03. 매개변수 효율적 파인튜닝(PEFT)과 LoRA / QLoRA]]
* [[chapter-qa/ch09-inference-optimization-qa/01-inference-fundamentals-and-hardware-math|Ch09-01. 추론 기초, 루프라인 모델 및 하드웨어 수학]]
