---
category: introduction-to-ai-engineering
title: "03. 언어 모델의 기초와 토큰화 (Tokenization)"
source: "AI Engineering · Chapter 1"
tags: [language-model, autoregressive, masked-language-model, tokenization, subword]
---

# 03. 언어 모델의 기초와 토큰화 (Tokenization)

## 📌 핵심 요약 & 전체 맥락
> **"AI는 글자를 읽지 않는다. 그들은 숫자(토큰)를 확률적으로 조립할 뿐이다."**  
> 
> 애플리케이션 계층에서 훌륭한 프롬프트를 짜기 위해서는 밑바탕이 되는 언어 모델(Language Model)이 텍스트를 어떻게 처리하고 생성하는지에 대한 기초적인 이해가 필수적입니다. 이 장에서는 현대 AI의 두 가지 큰 줄기인 **자동회귀(Autoregressive) 모델**과 **마스크드(Masked) 모델**의 차이를 이해하고, 텍스트가 숫자로 변환되는 **토큰화(Tokenization)**의 수학적 효율성을 다룹니다.

---

## 1. 언어 모델의 2가지 핵심 패러다임

언어 모델은 근본적으로 '단어의 연속된 시퀀스 확률을 계산하는 함수'입니다. 학습 방식에 따라 두 가지로 나뉩니다.

### ① 자동회귀 언어 모델 (Autoregressive LM / Causal LM)
* **대표 모델:** GPT 시리즈, Llama, Claude
* **학습 방식 (Next-token prediction):** 앞서 나온 단어들(문맥)을 바탕으로, **바로 다음에 올 한 단어**를 예측하도록 훈련됩니다. 
* **특징:** 과거만 볼 수 있고 미래는 볼 수 없는 '단방향(Uni-directional)' 구조입니다.
* **주요 용도:** 우리가 흔히 아는 챗봇, 텍스트 생성, 코드 작성 등 생성이 필요한 모든 **생성형 AI(Generative AI)**의 근간입니다.

### ② 마스크드 언어 모델 (Masked LM)
* **대표 모델:** BERT, RoBERTa
* **학습 방식 (Fill-in-the-blank):** 문장 중간에 있는 단어를 구멍(Mask) 뚫어 놓고, **양옆의 문맥을 동시에 참조**하여 빈칸의 단어를 맞히도록 훈련됩니다.
* **특징:** 과거와 미래의 문맥을 동시에 보는 '양방향(Bi-directional)' 구조입니다.
* **주요 용도:** 긴 문장을 텍스트로 줄줄 생성하는 데는 젬병입니다. 대신, 문장의 의미를 깊게 파악하여 벡터로 압축(Embedding)하거나, 감성 분석, 스팸 분류 등을 수행하는 데 압도적인 가성비를 자랑합니다.

---

## 2. 텍스트 처리의 근간: 토큰화 (Tokenization)

AI 모델은 '사과'라는 한글이나 'Apple'이라는 알파벳을 그대로 읽을 수 없습니다. 모델은 오직 숫자 행렬(Tensor)만 연산할 수 있습니다. 문자열을 숫자로 변환하는 과정을 **토큰화(Tokenization)**라고 합니다.

### 왜 글자(Character) 단위나 단어(Word) 단위를 쓰지 않을까?
* **글자 단위 (Character-level):** 영어는 a~z 26개 알파벳만 알면 되니 사전(Vocabulary) 크기가 작아 메모리를 덜 먹습니다. 하지만 'A-p-p-l-e' 처럼 글자를 하나하나 조합해 의미를 파악하려면 문맥 길이가 너무 길어지고 학습이 극도로 어려워집니다.
* **단어 단위 (Word-level):** 'Apple', 'Apples', 'Appled' 등 세상의 모든 단어를 등록하려면 사전 크기가 무한대로 커지며, 사전에 없는 신조어(Out-of-vocabulary)를 만나면 에러가 납니다.

### 🌟 현대의 해결책: 서브워드 토큰화 (Subword Tokenization)
현대의 모든 파운데이션 모델(BPE, Tiktoken 등)은 단어를 의미가 있는 작은 조각(Subword)으로 쪼갭니다.
* 예: `unhappiness` ➔ `un` + `happi` + `ness`
* **장점 1:** 자주 쓰이는 단어는 통째로 1개 토큰으로 취급하고, 희귀한 단어는 알파벳 덩어리로 쪼개어 OOV(미등록 단어) 문제를 완벽히 해결합니다.
* **장점 2:** 모델의 사전 크기(보통 3만 ~ 12만 개)를 적절히 유지하여 메모리 효율을 극대화합니다.

> 💡 **비용 계산의 단위:**  
> OpenAI나 Anthropic API의 요금은 글자 수나 단어 수가 아닌 **'토큰 수'**를 기준으로 과금됩니다. 영어의 경우 보통 1토큰은 0.75단어(약 4글자) 수준이며, 한글은 토크나이저 효율이 영어나 코드에 비해 떨어져 동일한 의미여도 2~3배 더 많은 토큰을 소모(과금)하기도 합니다.

---

## 🔗 연관 문서
* [[00-ch01-overview|00. Chapter 1 전체 개요 및 목차]]
* [[chapter-qa/ch02-foundation-models-qa/08-sampling-and-probabilistic-nature|Ch02-08. 샘플링(Sampling)과 확률적 본성]]
* [[chapter-qa/ch08-datasets-and-data-engineering-qa/03-data-processing-deduplication-and-formatting|Ch08-03. 데이터 포맷팅과 채팅 템플릿 토큰]]
