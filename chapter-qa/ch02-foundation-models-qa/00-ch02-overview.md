---
category: foundation-models
title: "Chapter 2: Understanding Foundation Models - 질의응답 및 핵심 개념 요약집"
source: "AI Engineering: Building Applications with Foundation Models (Chip Huyen) · Chapter 2"
created: 2026-08-23
---

# Chapter 2: Understanding Foundation Models (질의응답 및 개념 정리)

> **교재:** *AI Engineering: Building Applications with Foundation Models* (Chip Huyen 저)  
> **범위:** Chapter 2 (Training Data, Modeling Architecture, Compute & Scaling Bottlenecks)

---

## 🗺️ 개념 지도 및 목차

| 번호 | 문서명 | 핵심 주제 | 주요 키워드 |
| :---: | :--- | :--- | :--- |
| **01** | [[01-multilingual-models\|01. 다국어 모델과 언어 불균형]] | 영어가 지배하는 인터넷 데이터와 비영어권 언어의 비용/성능 격차 | `Common Crawl`, `Low-resource`, `Tokenization Latency`, `Token Cost` |
| **02** | [[02-domain-specific-models\|02. 도메인 특화 모델]] | 범용 모델의 한계와 전문 영역(의료, 신약, 제조) 특화 모델 | `Domain-Specific`, `AlphaFold`, `BioNeMo`, `Med-PaLM 2` |
| **03** | [[03-attention-mechanism-and-math\|03. 어텐션 메커니즘과 수학적 수식]] | Q, K, V의 개념과 어텐션 수식의 단계별 수학적 연산 분해 | `Query/Key/Value`, `Scaled Dot-Product`, `Softmax`, `Multi-Head Attention` |
| **04** | [[04-transformer-block-architecture\|04. 트랜스포머 블록 구조]] | 어텐션+MLP 블록 구조와 Llama 2/3의 하이퍼파라미터 스펙 | `Transformer Block`, `MLP`, `Activation(ReLU/GELU)`, `Model Head` |
| **05** | [[05-parameters-tokens-flops-and-moe\|05. 파라미터, 연산량(FLOPs)과 MoE]] | 파라미터의 본질, VRAM 계산, MoE 희소 모델, 학습 비용 추정 | `Parameter 정의`, `VRAM 계산`, `MoE(Mixtral)`, `FLOPs vs FLOP/s`, `Inverse Scaling` |
| **06** | [[06-scaling-extrapolation-and-bottlenecks\|06. 스케일링 외삽과 2대 병목]] | 원샷 하이퍼파라미터 전이와 데이터 고갈/전력 부족 병목 | `Scaling Extrapolation`, `Data Bottleneck`, `Model Collapse`, `Electricity Bottleneck` |
| **07** | [[07-post-training-and-alignment\|07. 포스트 트레이닝과 가치 정렬]] | 2% 연산량으로 대화 능력 잠금 해제, SFT 및 RLHF/DPO 정렬 | `Post-Training`, `SFT`, `Behavior Cloning`, `Reward Model`, `RLHF`, `DPO`, `RLAIF` |
| **08** | [[08-sampling-and-probabilistic-nature\|08. 샘플링과 AI의 확률적 본성]] | 온도/Top-p, 추론 시점 연산, 구조화 출력, 환각/불일치성 대응 (도표 포함) | `Sampling`, `Temperature`, `Top-p`, `Test-Time Compute`, `Logit Masking`, `Hallucination` |
| **09** | [[09-logits-and-softmax-deep-dive\|09. [심화] 로짓(Logits)과 소프트맥스]] | LM Head 출력, 실수 점수 벡터, 온도 연산, 로짓 마스킹(JSON 제약) 원리 | `Logits`, `Softmax`, `LM Head`, `Vocabulary Size`, `Logit Masking`, `Log-Odds` |
| **10** | [[10-deep-dive-adversarial-outputs-in-best-of-n\|10. [심화] 적대적 응답과 보상 해킹]] | Best-of-N에서 N>400 시 성능 하락 원인, 굿하트의 법칙, 검증기 맹점 | `Test-Time Compute`, `Verifier`, `Adversarial Outputs`, `Reward Hacking`, `Goodhart's Law` |

---

## 💡 한눈에 꿰뚫는 핵심 흐름

```mermaid
flowchart TD
    Data[1. 학습 데이터 (Training Data)] --> Multi[다국어 불균형 & 토큰 비용 편차]
    Data --> Domain[도메인 특화 데이터 & 프라이버시]
    
    Arch[2. 모델 아키텍처 (Architecture)] --> Attn[어텐션 메커니즘 Q, K, V]
    Arch --> Block[트랜스포머 블록 = Attention + MLP]
    
    Scale[3. 규모와 연산 (Scale & Compute)] --> Params[파라미터 & VRAM / MoE 희소화]
    Scale --> Compute[연산량 FLOPs & 학습 비용 산정]
    
    Future[4. 미래와 한계 (Bottlenecks)] --> Extrap[스케일링 외삽: 하이퍼파라미터 전이]
    Future --> Bottle[2대 병목: 데이터 고갈 & 전력 한계]

    Post[5. 사후 학습 (Post-Training)] --> SFT[1단계: 지도 미세조정 SFT]
    Post --> Align[2단계: 선호도 정렬 RLHF / DPO / RLAIF]

    Samp[6. 샘플링 & 확률적 본성 (Sampling)] --> Temp[온도 & Top-p / Test-Time 연산]
    Samp --> Struct[구조화 출력 로짓 마스킹]
    Samp --> Hallu[환각 & 불일치성 통제]
```
