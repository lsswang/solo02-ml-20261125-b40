"""Smoothed first-order word-transition models for home and school.

The input is intentionally plain JSON so that another file with the same
schema can replace the sample without changing code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from typing import Any, Iterable

START = "⟦开头⟧"


def dedupe(items: Iterable[str]) -> list[str]:
    """Preserve first occurrence while removing duplicates."""
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item is None:
            continue
        word = str(item)
        if word not in seen:
            seen.add(word)
            out.append(word)
    return out


def compatible_orders(words: list[str]) -> list[tuple[str, ...]]:
    """All distinct word orders compatible with an unordered group record."""
    if not words:
        return []
    # dict.fromkeys removes duplicate permutations caused by repeated words.
    return list(dict.fromkeys(itertools.permutations(str(w) for w in words)))


def record_weight(record: dict[str, Any]) -> float:
    return float(record.get("weight", record.get("count", 1.0)))


@dataclass
class SceneModel:
    scene: str
    vocab: list[str]
    alpha: float = 1.0
    ambiguity_weighting: str = "uniform"
    bos_counts: dict[str, float] = field(default_factory=dict)
    transition_counts: dict[tuple[str, str], float] = field(default_factory=dict)
    context_counts: dict[str, float] = field(default_factory=dict)
    unigram_counts: dict[str, float] = field(default_factory=dict)
    weighted_records: float = 0.0
    weighted_tokens: float = 0.0
    enumerations: list[dict[str, Any]] = field(default_factory=list)

    def _v(self) -> int:
        return max(1, len(self.vocab))

    def start_probability(self, word: str) -> float:
        numerator = self.bos_counts.get(word, 0.0) + self.alpha
        denominator = self.weighted_records + self.alpha * self._v()
        return numerator / denominator

    def probability(self, current: str | None, nxt: str) -> float:
        """Smoothed P(next | current).  current=None means sentence start."""
        if current is None:
            return self.start_probability(nxt)
        numerator = self.transition_counts.get((current, nxt), 0.0) + self.alpha
        denominator = self.context_counts.get(current, 0.0) + self.alpha * self._v()
        return numerator / denominator

    def distribution(self, current: str | None, words: Iterable[str] | None = None) -> dict[str, float]:
        candidates = list(words) if words is not None else list(self.vocab)
        return {w: self.probability(current, w) for w in candidates}

    def normalized_board_distribution(
        self, current: str | None, board_words: list[str]
    ) -> tuple[dict[str, float], float]:
        raw = self.distribution(current, self.vocab)
        on_board = {w: raw[w] for w in board_words if w in raw}
        mass = sum(on_board.values())
        if mass <= 0:
            # Add-alpha guarantees this in ordinary use; keep a safe fallback.
            uniform = 1.0 / max(1, len(board_words))
            return {w: uniform for w in board_words}, 0.0
        return {w: p / mass for w, p in on_board.items()}, mass

    def query(self, current: str | None = None, top_k: int | None = None) -> list[dict[str, Any]]:
        current = None if current in (START, "", None) else str(current)
        probs = self.distribution(current)
        rows = [
            {"word": w, "probability": p}
            for w, p in sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        return rows[:top_k] if top_k else rows

    def top_word(self, current: str | None = None) -> str:
        return self.query(current, top_k=1)[0]["word"]

    def unigram_probability(self, word: str) -> float:
        numerator = self.unigram_counts.get(word, 0.0) + self.alpha
        denominator = self.weighted_tokens + self.alpha * self._v()
        return numerator / denominator

    def unigram_top(self) -> str:
        return max(self.vocab, key=lambda w: (self.unigram_counts.get(w, 0.0), w))

    def as_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "alpha": self.alpha,
            "ambiguity_weighting": self.ambiguity_weighting,
            "vocabulary": self.vocab,
            "weighted_records": self.weighted_records,
            "weighted_tokens": self.weighted_tokens,
            "start_probabilities": {w: self.start_probability(w) for w in self.vocab},
            "unigram_probabilities": {w: self.unigram_probability(w) for w in self.vocab},
            "transition_probabilities": {
                f"{u} -> {w}": self.probability(u, w)
                for u in self.vocab
                for w in self.vocab
            },
            "counts": {
                "bos": self.bos_counts,
                "unigram": self.unigram_counts,
                "context": self.context_counts,
                "transition": {f"{u}->{w}": c for (u, w), c in sorted(self.transition_counts.items())},
            },
            "compatible_order_enumerations": self.enumerations,
        }


def _records_for_scene(corpus: dict[str, Any] | None, scene: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corpus = corpus or {}
    block = corpus.get(scene, {}) if isinstance(corpus, dict) else {}
    sentences = block.get("sentences", block.get("ordered", [])) or []
    groups = block.get("groups", block.get("unordered", [])) or []

    sentence_records: list[dict[str, Any]] = []
    for item in sentences:
        if isinstance(item, dict):
            words = item.get("words", item.get("sentence", []))
            rec = dict(item)
        else:
            words = item
            rec = {}
        sentence_records.append({"kind": "sentence", "words": [str(w) for w in words], **rec})

    group_records: list[dict[str, Any]] = []
    for item in groups:
        if isinstance(item, dict):
            words = item.get("words", item.get("group", []))
            rec = dict(item)
        else:
            words = item
            rec = {}
        group_records.append({"kind": "group", "words": [str(w) for w in words], **rec})
    return sentence_records, group_records


def gather_vocabulary(data: dict[str, Any]) -> list[str]:
    words: list[Any] = []

    def add_seq(seq: Any) -> None:
        if isinstance(seq, dict):
            seq = seq.get("words", seq.get("sentence", seq.get("group", [])))
        if isinstance(seq, (list, tuple)):
            words.extend(seq)

    board = data.get("board", {}) or {}
    for card in board.get("cards", board.get("words", [])):
        if isinstance(card, dict):
            words.append(card.get("word", card.get("text", card.get("词"))))
        else:
            words.append(card)

    for scene_block in (data.get("corpus", {}) or {}).values():
        if not isinstance(scene_block, dict):
            continue
        for key in ("sentences", "groups", "ordered", "unordered"):
            for rec in scene_block.get(key, []) or []:
                add_seq(rec)

    for scene_block in (data.get("heldout", data.get("held_out", {})) or {}).values():
        for rec in scene_block or []:
            add_seq(rec)

    tasks = data.get("tasks", {}) or {}
    for value in tasks.values():
        if isinstance(value, list) and value and isinstance(value[0], list):
            for seq in value:
                add_seq(seq)
        else:
            add_seq(value)

    for replay in data.get("replays", []) or []:
        add_seq(replay.get("sentence", []))

    for scenario in data.get("probability_scenarios", []) or []:
        words.extend((scenario.get("distribution", {}) or {}).keys())
        if scenario.get("locked_word"):
            words.append(scenario["locked_word"])

    return dedupe(words)


def train_scene(
    scene: str,
    data: dict[str, Any],
    vocab: list[str],
    alpha: float = 1.0,
    ambiguity_weighting: str = "uniform",
) -> SceneModel:
    model = SceneModel(scene=scene, vocab=vocab, alpha=alpha, ambiguity_weighting=ambiguity_weighting)
    sentences, groups = _records_for_scene(data.get("corpus"), scene)

    for record in sentences + groups:
        raw_words = record.get("words", [])
        orders = [tuple(raw_words)] if record["kind"] == "sentence" else compatible_orders(raw_words)
        if not orders:
            continue
        base_weight = record_weight(record)
        if ambiguity_weighting == "each_order" or record["kind"] == "sentence":
            order_weight = base_weight
        elif ambiguity_weighting == "uniform":
            order_weight = base_weight / len(orders)
        else:
            raise ValueError(f"unknown ambiguity_weighting: {ambiguity_weighting}")

        enum_info = {
            "kind": record["kind"],
            "record": list(raw_words),
            "compatible_orders": [list(o) for o in orders],
            "weight_per_order": order_weight,
        }
        if record.get("id"):
            enum_info["id"] = record["id"]
        model.enumerations.append(enum_info)

        for order in orders:
            model.weighted_records += order_weight
            model.bos_counts[order[0]] = model.bos_counts.get(order[0], 0.0) + order_weight
            for word in order:
                model.unigram_counts[word] = model.unigram_counts.get(word, 0.0) + order_weight
                model.weighted_tokens += order_weight
            for left, right in zip(order, order[1:]):
                key = (left, right)
                model.transition_counts[key] = model.transition_counts.get(key, 0.0) + order_weight
                model.context_counts[left] = model.context_counts.get(left, 0.0) + order_weight
            # Without an explicit EOS token, the final word is not added as a
            # transition context; add-alpha still gives its unseen successors mass.

    return model


def _sentence_log_probability(model: SceneModel, sentence: list[str]) -> tuple[float, float]:
    if not sentence:
        return 0.0, 0.0
    total = math.log(model.start_probability(sentence[0]))
    for left, right in zip(sentence, sentence[1:]):
        total += math.log(model.probability(left, right))
    return total, float(len(sentence))


def evaluate_scene(model: SceneModel, heldout: list[Any]) -> dict[str, Any]:
    sentences: list[list[str]] = []
    for rec in heldout or []:
        if isinstance(rec, dict):
            seq = rec.get("words", rec.get("sentence", []))
        else:
            seq = rec
        sentences.append([str(w) for w in seq])

    token_count = 0
    transition_correct = 0
    unigram_correct = 0
    total_nll_transition = 0.0
    total_nll_unigram = 0.0
    sentence_reports: list[dict[str, Any]] = []

    for sentence in sentences:
        rows = []
        for i, word in enumerate(sentence):
            context = None if i == 0 else sentence[i - 1]
            trans_guess = model.top_word(context)
            uni_guess = model.unigram_top()
            token_count += 1
            transition_correct += int(trans_guess == word)
            unigram_correct += int(uni_guess == word)
            rows.append(
                {
                    "position": i + 1,
                    "context": START if context is None else context,
                    "actual": word,
                    "transition_guess": trans_guess,
                    "transition_correct": trans_guess == word,
                    "frequency_guess": uni_guess,
                    "frequency_correct": uni_guess == word,
                    "transition_probability": model.probability(context, word),
                    "unigram_probability": model.unigram_probability(word),
                }
            )
        trans_logp, n = _sentence_log_probability(model, sentence)
        uni_logp = sum(math.log(model.unigram_probability(w)) for w in sentence)
        total_nll_transition += -trans_logp
        total_nll_unigram += -uni_logp
        sentence_reports.append(
            {
                "sentence": sentence,
                "transition_log_probability": trans_logp,
                "unigram_log_probability": uni_logp,
                "rows": rows,
            }
        )

    def perplexity(nll: float, n: int) -> float | None:
        return math.exp(nll / n) if n else None

    return {
        "scene": model.scene,
        "sentence_count": len(sentences),
        "token_count": token_count,
        "transition_top1_accuracy": transition_correct / token_count if token_count else None,
        "frequency_top1_accuracy": unigram_correct / token_count if token_count else None,
        "transition_total_nll": total_nll_transition,
        "frequency_total_nll": total_nll_unigram,
        "transition_perplexity": perplexity(total_nll_transition, token_count),
        "frequency_perplexity": perplexity(total_nll_unigram, token_count),
        "sentences": sentence_reports,
    }


def train_both(data: dict[str, Any]) -> tuple[dict[str, SceneModel], dict[str, Any]]:
    config = data.get("config", {}) or {}
    alpha = float(config.get("alpha", config.get("smoothing", 1.0)))
    weighting = config.get("ambiguity_weighting", "uniform")
    vocab = gather_vocabulary(data)
    models = {
        scene: train_scene(scene, data, vocab, alpha=alpha, ambiguity_weighting=weighting)
        for scene in ("home", "school")
    }
    evaluation = {
        scene: evaluate_scene(models[scene], (data.get("heldout", data.get("held_out", {})) or {}).get(scene, []))
        for scene in ("home", "school")
    }
    return models, evaluation
