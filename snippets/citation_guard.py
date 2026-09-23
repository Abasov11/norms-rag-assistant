"""
Иллюстративный фрагмент из RAG-ассистента по нормативной базе (портфолио-кейс, не рабочая сборка).

Заземление ответа тегами источников. Корпус — сотни документов, и один и тот же номер
пункта («5.2») встречается почти в каждом из них, поэтому сверять цитату по номеру
глобально нельзя. Решение: каждому найденному фрагменту присваивается тег [И1]…[Иn], модель
обязана ссылаться только тегами, а код проверяет, что все теги в ответе существуют — сослаться
на несуществующий тег («выдумать» источник) технически невозможно, а не «запрещено промптом».

Второй узел — детектор отказа. Маркер отказа должен идти ПЕРВЫМ словом ответа, а не просто
где-то встречаться: на живом прогоне правило «маркер и ни одной цитаты» давало 2 верных
отказа из 15 вопросов вне корпуса, потому что модель иногда прикладывала к отказу цитаты
смежных норм с объяснением, почему они не подходят — после смены правила на «начинается с
маркера» тот же прогон дал 14 из 15.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ABSTAIN_MARK = "НЕДОСТАТОЧНО ОСНОВАНИЙ"
RX_TAG = re.compile(r"И\s*(\d+)")


@dataclass
class Verdict:
    ok: bool
    cited: list[int]
    hallucinated: list[int]   # теги вне диапазона переданных источников
    abstained: bool
    n_sources: int = 0


def is_abstain(answer: str) -> bool:
    """Отказ — ответ, который НАЧИНАЕТСЯ с маркера, а не любой, где маркер встретился где-то
    внутри (иначе отказ с объяснением через смежные нормы ошибочно засчитывается ответом)."""
    head = answer.lstrip().lstrip("#*_ \t")
    return head.upper().startswith(ABSTAIN_MARK)


def verify(answer: str, n_sources: int) -> Verdict:
    cited = sorted({int(tag) for tag in RX_TAG.findall(answer)})
    hallucinated = [tag for tag in cited if tag < 1 or tag > n_sources]
    abstained = is_abstain(answer)
    ok = not hallucinated and (abstained or bool(cited))
    return Verdict(ok=ok, cited=cited, hallucinated=hallucinated,
                    abstained=abstained, n_sources=n_sources)


def expand_tags(answer: str, sources: dict[int, str]) -> str:
    """[И1] / [И1, И3] -> человекочитаемая ссылка «(СП …, п. 5.2.1)» в финальном тексте."""
    group = re.compile(r"\[\s*((?:И\s*\d+\s*[,;]?\s*)+)\]")

    def repl(match: re.Match) -> str:
        refs = [sources[int(tag)] for tag in RX_TAG.findall(match.group(1)) if int(tag) in sources]
        return f"({'; '.join(refs)})" if refs else match.group(0)

    return group.sub(repl, answer)
