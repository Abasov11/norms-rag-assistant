"""
Иллюстративный фрагмент из RAG-ассистента по нормативной базе (портфолио-кейс, не рабочая сборка).

Гибридный поиск по корпусу из сотен нормативных документов: вектор (эмбеддинги) + лексика
(BM25) + label-дорожка (прямая ссылка «п. 5.2.1» / «статья N»), слитые Reciprocal Rank Fusion.

Почему не «чистый вектор»: инженерные вопросы часто называют номер пункта или документа
напрямую — тут точный лексический матч надёжнее семантики. Вес лексической дорожки (0.5,
не 1.0) — не интуиция, а результат замера на эвал-наборе: равный вес роняет recall (OR по
лексемам тащит шум и вытесняет векторных кандидатов в RRF), заниженный вес — нейтрален и
служит страховкой под точные термины и номера.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

RX_ARTICLE_REF = re.compile(r"\b(?:пункт|п\.?)\s*№?\s*(\d+(?:\.\d+)*)", re.I)
RRF_K = 60
LEX_WEIGHT = 0.5  # см. обоснование в докстринге модуля


@dataclass
class Hit:
    chunk_id: str
    score: float
    row: dict
    lanes: tuple[str, ...]


def reciprocal_rank_fusion(
    ranklists: dict[str, list[str]], weights: dict[str, float] | None = None, k: int = RRF_K,
) -> dict[str, float]:
    """RRF: чем выше кандидат в исходном списке, тем больше вклад 1/(k+rank)."""
    weights = weights or {}
    scores: dict[str, float] = {}
    for lane_name, ids in ranklists.items():
        weight = weights.get(lane_name, 1.0)
        for rank, chunk_id in enumerate(ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + rank + 1)
    return scores


def search(query: str, *, vector_search, lexical_search, get_by_article, k: int = 8, pool: int = 100) -> list[Hit]:
    """vector_search/lexical_search/get_by_article — адаптеры к хранилищу (внедряются извне,
    чтобы функцию можно было тестировать без базы и без сети)."""
    vec_ids = [cid for cid, _ in vector_search(query, pool)]
    lex_ids = [cid for cid, _ in lexical_search(query, pool // 3)]
    lanes: dict[str, list[str]] = {"vector": vec_ids, "lexical": lex_ids}

    # Label-дорожка: явная ссылка на пункт — тянем его чанк напрямую, детерминированно.
    label_ids = [row["id"] for num in RX_ARTICLE_REF.findall(query) for row in get_by_article(num)]
    if label_ids:
        lanes["label"] = label_ids

    fused = reciprocal_rank_fusion(lanes, weights={"lexical": LEX_WEIGHT})
    # Прямая ссылка на пункт должна пробиться в топ независимо от того, как её оценил RRF.
    for cid in label_ids:
        fused[cid] = fused.get(cid, 0.0) + 1.0

    top_ids = [cid for cid, _ in sorted(fused.items(), key=lambda item: -item[1])[:k]]
    found_in = {name: set(ids) for name, ids in lanes.items()}
    return [
        Hit(chunk_id=cid, score=fused[cid], row={"id": cid},
            lanes=tuple(name for name, ids in found_in.items() if cid in ids))
        for cid in top_ids
    ]
