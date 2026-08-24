---
category: inference-optimization
title: "00. Chapter 9 전체 개요 및 목차 (Inference Optimization)"
source: "AI Engineering · Chapter 9 (p.405-448)"
tags: [inference-optimization, roofline-model, prefill-vs-decode, kv-cache, flashattention, continuous-batching, speculative-decoding, prompt-caching, tensor-parallelism, pipeline-parallelism]
---

# 00. Chapter 9 전체 개요 및 목차 (Inference Optimization)

## 📌 챕터 핵심 요약 (Executive Summary)
> **"모델 학습이 '지능을 만드는 일'이라면, 추론 최적화는 '그 지능을 실제 비즈니스로 지속 가능하게 만드는 일'이다."**  
> 생성형 AI 서비스의 성패는 사용자 경험을 결정짓는 **지연시간(TTFT, ITL)**과 서비스의 경제성을 결정짓는 **비용 및 처리량(Throughput, Goodput)**에 달려 있습니다.  
> LLM 추론은 **병렬 연산 위주의 프리필 (Prefill: Compute-bound) 단계**와 **가중치 로딩 위주의 디코딩 (Decode: Memory bandwidth-bound) 단계**라는 완전히 상이한 두 국면으로 진행됩니다.  
> 본 챕터에서는 **루프라인 모델 (Roofline Model)**을 기반으로 한 하드웨어 한계 분석부터, 메모리 병목을 극복하는 **KV 캐시와 PagedAttention**, IO 연산을 획기적으로 줄인 **FlashAttention (SRAM 타일링)**, 처리량을 10배 이상 끌어올리는 **연속 배치 (Continuous / In-flight Batching)**, 무손실 2~3배 가속을 달성하는 **추측 디코딩 (Speculative Decoding)**, 비용을 90% 절감하는 **프롬프트 캐싱 (Prompt Caching)**, 그리고 수십 대의 GPU를 묶는 **텐서 / 파이프라인 병렬화 (TP, PP)**까지 현존하는 모든 서빙 최적화 기법을 완벽히 정리합니다.

---

## 🗺️ 전체 개념 맵 (Mindmap)

```mermaid
mindmap
  root((Chapter 9. 추론 최적화))
    1. 하드웨어 수학 & 병목 분석
      루프라인 모델 (연산 집약도 vs 대역폭 한계)
      프리필 (Compute-bound) vs 디코드 (Memory-bound)
      추론 4대 메트릭 (TTFT, ITL/TPOT, Throughput, Goodput)
      하드웨어 효율성 (MFU, MBU) 및 메모리 계층
    2. 메모리 & 어텐션 최적화
      KV 캐시 메커니즘과 PagedAttention (vLLM)
      FlashAttention 1/2/3 (SRAM 타일링, 커널 융합)
      배칭의 진화 (정적 ➔ 동적 ➔ 연속/In-flight 배칭)
    3. 고급 가속 & 분산 서빙
      추측 디코딩 (Speculative Decoding & Medusa)
      프롬프트 캐싱 (Prefix / Context Caching)
      분산 병렬화 (텐서 병렬화 TP vs 파이프라인 병렬화 PP)
```

---

## 📑 소챕터 상세 목차 및 도표 색인

| 소챕터 번호 및 파일명 | 핵심 다루는 주제 | 포함된 핵심 Figures & Tables |
| :--- | :--- | :--- |
| **[[01-inference-fundamentals-and-hardware-math\|01. 추론 기초, 루프라인 모델 및 하드웨어 수학]]** | • 단순 추론 서비스 구조<br>• 루프라인 모델과 연산 집약도<br>• 프리필(연산 병목) vs 디코딩(메모리 대역폭 병목)<br>• 추론 4대 메트릭과 Goodput<br>• MFU/MBU 및 GPU 메모리 계층 구조 | • **Figure 9-1**: 단순 추론 서비스 아키텍처<br>• **Figure 9-2**: 루프라인 모델(Roofline Chart)<br>• **Figure 9-3**: 프리필(Prefill)과 디코딩(Decode) 2단계<br>• **Figure 9-4**: Throughput vs Goodput (SLO 준수율)<br>• **Table 9-1**: PaLM 모델의 MFU 실측표<br>• **Figure 9-5**: Llama 2-70B의 칩별 MBU 비교<br>• **Figure 9-6**: 연산 프리미티브 (Scalar, Vector, Tensor)<br>• **Table 9-2**: NVIDIA H100 SXM 정밀도별 FLOP/s 스펙<br>• **Figure 9-7**: GPU 메모리 계층 (SRAM, HBM, RAM) |
| **[[02-kv-cache-flashattention-and-batching\|02. KV 캐시, FlashAttention 및 연속 배칭]]** | • 최적화 기법의 품질 손실 트레이드오프<br>• KV 캐시 수학적 원리와 PagedAttention<br>• FlashAttention IO 최적화 (SRAM 타일링, 커널 융합)<br>• PyTorch 최적화 브레이크다운<br>• 배칭의 진화: 정적 ➔ 동적 ➔ 연속 배칭 | • **Figure 9-8**: 손실 없는 최적화 vs 손실성 최적화 분류<br>• **Figure 9-12**: 디코딩 스텝별 KV 캐시 재사용 구조<br>• **Figure 9-13**: FlashAttention 커널 융합 및 SRAM 타일링<br>• **Figure 9-14**: PyTorch 최적화 단계별 처리량 향상 곡선<br>• **Figure 9-15**: 정적 배칭 vs 동적 배칭 비교<br>• **Figure 9-16**: 연속 배칭 (Continuous/In-flight Batching) |
| **[[03-speculative-decoding-caching-and-parallelism\|03. 추측 디코딩, 프롬프트 캐싱 및 분산 병렬화]]** | • 추측 디코딩 (드래프트-타겟 검증 무손실 가속)<br>• Inference with Reference & Medusa 멀티헤드<br>• 프롬프트 캐싱 (Prefix Caching) 원리와 비용 절감<br>• 텐서 병렬화(TP: Megatron Column/Row Split)<br>• 파이프라인 병렬화(PP: 1F1B 마이크로배치 스케줄링) | • **Figure 9-9**: 추측 디코딩 (Speculative Decoding) 흐름도<br>• **Figure 9-10**: 참조 텍스트 기반 추측 디코딩<br>• **Figure 9-11**: Medusa 멀티헤드 트리 예측<br>• **Figure 9-17**: 프롬프트 캐싱 (Prompt/Prefix Cache)<br>• **Table 9-3**: Anthropic 프롬프트 캐싱 비용/지연시간 절감표<br>• **Figure 9-18**: 텐서 병렬화 (Tensor Parallelism 행렬 분할)<br>• **Figure 9-19**: 파이프라인 병렬화 (Pipeline Parallelism 1F1B) |

---

## 💡 주요 축약어 원문 및 해설 사전 (Abbreviations Glossary)

* **TTFT (Time To First Token, 첫 토큰 생성 지연시간):** 사용자가 프롬프트를 전송한 후 언어 모델이 첫 번째 출력 글자(토큰)를 렌더링할 때까지 걸리는 시간으로, 프리필(Prefill) 연산 속도에 의해 결정됨.
* **ITL (Inter-Token Latency, 토큰 간 생성 지연시간 / TPOT):** 첫 토큰이 나온 이후 다음 글자가 하나씩 생성되는 간격 시간(ms/token)으로, 디코딩(Decode) 메모리 대역폭에 의해 결정됨.
* **TPOT (Time Per Output Token, 출력 토큰당 생성 시간):** ITL과 동의어로, 모델이 1개의 출력 토큰을 뱉어내는 데 걸리는 평균 지연시간.
* **MFU (Model FLOPs Utilization, 모델 연산력 활용도):** GPU 하드웨어가 낼 수 있는 이론상 최대 연산 속도(Peak FLOPs) 대비 모델이 실제 유의미하게 활용한 연산량의 비율.
* **MBU (Memory Bandwidth Utilization, 메모리 대역폭 활용도):** GPU 하드웨어의 이론상 최대 메모리 전송 속도 대비 실제 가중치 및 KV 캐시를 읽어오는 데 사용된 대역폭의 비율.
* **KV Cache (Key-Value Cache, 키-값 캐시):** 이전 스텝들에서 계산된 트랜스포머 어텐션의 Key와 Value 벡터를 GPU VRAM에 저장해 두어, 다음 단어를 예측할 때 이전 토큰들을 중복 재계산하지 않도록 방지하는 메모리 구조.
* **vLLM (Virtual Large Language Model):** 운영체제의 가상 메모리 페이징 기법을 차용하여 KV 캐시 메모리 낭비(단편화)를 거의 0%로 줄인 고성능 LLM 서빙 프레임워크 (PagedAttention 기반).
* **SRAM (Static Random-Access Memory, GPU 정적 메모리):** GPU 칩 내부에 위치하여 대역폭이 19TB/s에 달할 정도로 극도로 빠르지만 용량이 수십~수백 MB로 매우 작은 온칩(On-chip) 메모리.
* **HBM (High Bandwidth Memory, 고대역폭 메모리):** GPU 칩 옆에 쌓아 올린 주 메모리(VRAM)로 용량이 수십 GB로 크지만 SRAM에 비해 데이터 전송 속도가 느린 메모리.
* **TP (Tensor Parallelism, 텐서 병렬화):** 단일 레이어의 거대 행렬(Attention 및 FFN)을 여러 GPU로 가로/세로로 쪼개어 동시에 연산한 뒤 결과를 합치는 모델 병렬화 기법 (Megatron-LM 방식).
* **PP (Pipeline Parallelism, 파이프라인 병렬화):** 모델의 레이어들을 여러 층위로 나누어 서로 다른 GPU에 순차적으로 배치하고 마이크로배치(1F1B)로 실행하는 분산 서빙 기법.
* **SLO (Service Level Objective, 서비스 수준 목표):** 서비스 운영팀이 달성하기로 약속한 정량적 성능 기준(예: "모든 요청의 99%는 TTFT가 200ms 이하여야 한다").
* **Goodput (유효 처리량):** 전체 처리된 토큰/요청 중 사전에 정의된 서비스 수준 목표(SLO)를 만족하면서 성공적으로 반환된 유효 처리량.
