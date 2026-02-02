# ADR-0002: Balance gating по effect size (SMD / Cramér’s V), а не по p-value

## Status
Accepted

## Context
На больших выборках p-value для тестов баланса почти всегда становится “значимым”
даже при микроскопических различиях. Если ориентироваться только на p-value, баланс будет “вечно провален”.

## Decision
- Numeric ковариаты: KS-test + считаем **SMD** (standardized mean difference).
  Флаг `passes` определяется по `abs(SMD) <= threshold`.
- Categorical ковариаты: χ² + считаем **Cramér’s V**.
  Флаг `passes` определяется по `V <= threshold`.
- p-value сохраняем в отчёте как справочную информацию, но **не используем как gating**.

## Consequences
- Отчёт устойчив к “p-value ловушке” на больших N.
- Порог баланса становится интерпретируемым и детерминированным.
