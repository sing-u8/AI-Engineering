---
description: "AI Engineering 도서 학습 및 질의응답 시 반드시 준수해야 하는 설명 스타일 및 노트 작성 하네스(Harness) 규칙"
globs: ["chapter-qa/**", "**/*.md"]
alwaysApply: true
---

# 📖 AI Engineering 학습 가이드라인 & 하네스(Harness) 규칙

사용자가 책(*AI Engineering: Building Applications with Foundation Models*)을 공부하고 노트를 작성할 때, 에이전트는 다음의 4가지 핵심 원칙을 반드시 준수하여 응답하고 문서를 생성해야 합니다.

---

## 1. 📜 서사 및 맥락(Narrative & Context) 완벽 유지
- 단순 나열식 축약(Bullet point only)이나 단순 요약을 지양합니다.
- **저자의 문제의식과 배경:** "이 기법이 왜 등장했는가?", "어떤 실무적 고통(Pain Point)을 해결하기 위한 것인가?"를 상세한 서사(Storytelling)로 풀어냅니다.
- **풍부한 비유와 실무 사례:** 복잡한 기계학습 개념을 일상적 비유(예: 시험 채점관, 열역학 분자, 입에 물린 재갈 등) 및 기업 실무 사례(OpenAI, Google, LinkedIn, Stanford 등)와 함께 설명합니다.

---

## 2. 🗺️ 도표(Figures & Tables) 연계 해설 필수
- 챕터 내에 등장하는 모든 **도표(Figure) 및 표(Table)**의 **번호, 책 페이지 위치, 도표 제목, 연관 개념**을 명시합니다.
- 단순 언급에 그치지 않고, **"이 도표에서 어떤 축과 곡선을 보아야 하는지", "도표가 증명하는 핵심 결론이 무엇인지"** 독자가 시각 자료를 정확히 해석할 수 있도록 가이드를 제공합니다.

---

## 3. 💡 핵심 용어(Terminology)의 명확한 정의 및 직관적 해설
- 새로운 전문 용어, 수식 기호, 약어가 등장할 때마다 다음 3단계를 거쳐 설명합니다:
  1. **한 줄 직관적 정의 (비유 포함)**
  2. **수학적/공학적 동작 메커니즘**
  3. **엔지니어링 실무에서의 중요성 및 활용 사례**

---

## 4. 📂 체계적인 파일 구조 및 개념 지도(TOC) 관리
- 모든 노트는 `chapter-qa/chXX-[chapter-name]-qa/` 하위 폴더에 주제별로 넘버링하여 체계적으로 분할 저장합니다.
- 각 챕터 폴더마다 `00-chXX-overview.md`를 두어 **전체 개념 지도(Table of Contents)**와 **Mermaid 흐름도**를 항상 최신 상태로 유지합니다.
