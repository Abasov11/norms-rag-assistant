"""
Иллюстративный фрагмент из RAG-ассистента по нормативной базе (портфолио-кейс, не рабочая сборка).

Числа и величины — самая опасная зона галлюцинации в инженерных нормах: неправильная
ссылка заметна, а тихо подменённое число («20 м» вместо «12 м») — нет. Тег-заземление
(citation_guard.py) проверяет только «источник существует», но не «то ли число в нём».
Этот слой сверяет числовые величины из утверждения ответа с текстом процитированного
источника детерминированно (без LLM), приводя единицы к общему виду: норма и ответ
могут писать одну и ту же величину по-разному («20 м» / «20 метров»).

Единица измерения берётся только из закрытого списка: без него первым же следующим словом
после числа объявлялось что угодно, и «380 либо» с «380 с» считались разными величинами.
"""
from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

_NUMBER = r"(?:\d+(?:[.,]\d+)?)"
_UNIT = r"(?:м|мм|см|км|кг|т|%|°C|час|мин)"
_QUANTITY = re.compile(rf"(?<![\w.])(?P<value>{_NUMBER})\s*(?P<unit>{_UNIT})?\b")

_UNIT_ALIASES = {
    "метра": "м", "метров": "м", "метр": "м",
    "миллиметра": "мм", "миллиметров": "мм",
    "процента": "%", "процентов": "%",
    "градуса": "°C", "градусов": "°C",
}


def _normalize_number(value: str) -> str:
    compact = value.replace(",", ".")
    try:
        normalized = format(Decimal(compact), "f")
    except InvalidOperation:
        return compact
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def _normalize_unit(unit: str | None) -> str:
    if not unit:
        return ""
    return _UNIT_ALIASES.get(unit.lower(), unit.lower())


def extract_quantities(text: str) -> set[tuple[str, str]]:
    """Множество (число, единица) в тексте, приведённых к общему виду."""
    found = set()
    for match in _QUANTITY.finditer(text):
        found.add((_normalize_number(match.group("value")), _normalize_unit(match.group("unit"))))
    return found


def figures_mismatch(claim: str, source_text: str) -> bool:
    """True, если в утверждении есть числовая величина, которой нет в процитированном
    источнике (после нормализации). Детерминированная проверка — работает даже если
    смысловая LLM-сверка недоступна (fail-open только гасит семантическую часть вердикта,
    числовая сверка от неё не зависит)."""
    claim_figures = extract_quantities(claim)
    source_figures = extract_quantities(source_text)
    return bool(claim_figures) and not claim_figures <= source_figures
