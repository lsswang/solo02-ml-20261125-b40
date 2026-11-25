from __future__ import annotations

import math
import random
import unicodedata
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


LEVELS = ("language", "wording", "last", "site_last")
KEY_SEP = ""


def norm_appellation(value: str) -> str:
    """Normalize Chinese/English address forms for matching."""
    if value is None:
        return ""
    s = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    # Keep Unicode word characters (Chinese is included), drop spaces/punctuation.
    return "".join(ch for ch in s if unicodedata.category(ch)[0] not in {"P", "Z", "C"})


def make_key(parts: Iterable[str]) -> str:
    return KEY_SEP.join("" if p is None else str(p) for p in parts)


def group_holdout(script_ids: Iterable[str], seed: int, fraction: float) -> Dict[str, List[str]]:
    groups = sorted(set(script_ids))
    if len(groups) < 2:
        return {"train": groups, "test": []}
    rng = random.Random(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    n_test = int(round(float(fraction) * len(shuffled)))
    n_test = max(1, min(n_test, len(shuffled) - 1))
    return {"train": sorted(shuffled[n_test:]), "test": sorted(shuffled[:n_test])}


def _empty_counts() -> Dict[str, Dict[str, Dict[str, int]]]:
    return {level: defaultdict(lambda: defaultdict(int)) for level in LEVELS}


def fit_referent_model(
    bundle: Dict[str, Any],
    records: Optional[List[Dict[str, Any]]] = None,
    fitted_on: str = "all",
) -> Dict[str, Any]:
    cfg = bundle["config"]
    alpha = float(cfg.get("smoothing_alpha", 0.5))
    used_records = bundle["records"] if records is None else records
    counts = _empty_counts()

    for r in used_records:
        lang = r["language"]
        w = norm_appellation(r["heard_wording"])
        last = r.get("last_person_id") or "∅"
        site = r.get("site_id") or "∅"
        chosen = r["chosen_person_id"]
        key_sets = {
            "language": (lang,),
            "wording": (lang, w),
            "last": (lang, w, last),
            "site_last": (lang, w, site, last),
        }
        for level in LEVELS:
            counts[level][make_key(key_sets[level])][chosen] += 1

    persons = {
        lang: [p["person_id"] for p in people]
        for lang, people in bundle["persons"].items()
    }
    distributions: Dict[str, Dict[str, Dict[str, float]]] = {}
    serial_counts: Dict[str, Dict[str, Dict[str, int]]] = {}
    for level in LEVELS:
        distributions[level] = {}
        serial_counts[level] = {}
        for key, person_counts in counts[level].items():
            lang = key.split(KEY_SEP, 1)[0]
            candidates = persons.get(lang, sorted(person_counts.keys()))
            n = sum(person_counts.values())
            denom = n + alpha * len(candidates)
            distributions[level][key] = {
                pid: (person_counts.get(pid, 0) + alpha) / denom for pid in candidates
            }
            serial_counts[level][key] = {pid: person_counts.get(pid, 0) for pid in candidates}

    appella_index = {}
    for a in bundle["appellations"]:
        appella_index[(a["language"], norm_appellation(a["appellation"]))] = a

    return {
        "model_type": "hierarchical_mixture_dirichlet_v1",
        "fitted_on": fitted_on,
        "n_records": len(used_records),
        "seed": cfg.get("seed"),
        "smoothing_alpha": alpha,
        "mixture_weights": cfg.get("mixture_weights", [0.03, 0.07, 0.25, 0.65]),
        "levels": list(LEVELS),
        "persons": persons,
        "appellation_index": {
            f"{lang}{KEY_SEP}{norm_appellation(a['appellation'])}": a
            for (lang, _norm), a in appella_index.items()
        },
        "counts": serial_counts,
        "distributions": distributions,
    }


def _level_distribution(model: Dict[str, Any], level: str, key_parts: Tuple[str, ...]) -> Dict[str, float]:
    key = make_key(key_parts)
    found = model["distributions"][level].get(key)
    if found is not None:
        return dict(found)
    # Unseen key: uniform over language candidates.
    lang = key_parts[0]
    candidates = model["persons"].get(lang, [])
    return {pid: 1.0 / len(candidates) for pid in candidates} if candidates else {}


def distribution_for_query(
    model: Dict[str, Any],
    language: str,
    wording: str,
    site_id: str = "",
    last_person_id: str = "",
) -> Dict[str, Any]:
    w = norm_appellation(wording)
    last = last_person_id or "∅"
    site = site_id or "∅"
    keys = {
        "language": (language,),
        "wording": (language, w),
        "last": (language, w, last),
        "site_last": (language, w, site, last),
    }
    weights = model.get("mixture_weights") or [0.25] * len(LEVELS)
    if len(weights) != len(LEVELS):
        raise ValueError("mixture_weights must have four entries")
    if not math.isclose(sum(weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        total = sum(weights)
        weights = [x / total for x in weights]

    candidates = model["persons"].get(language, [])
    mixture = {pid: 0.0 for pid in candidates}
    components = {}
    for level, weight in zip(LEVELS, weights):
        dist = _level_distribution(model, level, keys[level])
        components[level] = {"key": make_key(keys[level]), "weight": weight, "distribution": dist}
        for pid in candidates:
            mixture[pid] += weight * dist.get(pid, 0.0)
    total = sum(mixture.values())
    if total > 0:
        mixture = {pid: p / total for pid, p in mixture.items()}
    ranked = sorted(mixture.items(), key=lambda kv: (-kv[1], kv[0]))
    return {"distribution": mixture, "ranked": ranked, "components": components}


def get_appellation(model: Dict[str, Any], language: str, wording: str) -> Optional[Dict[str, Any]]:
    key = f"{language}{KEY_SEP}{norm_appellation(wording)}"
    return model.get("appellation_index", {}).get(key)


def evaluate_reference(
    bundle: Dict[str, Any],
    model: Dict[str, Any],
    language: str,
    wording: str,
    intended_person_id: str = "",
    site_id: str = "",
    last_person_id: str = "",
) -> Dict[str, Any]:
    """Return full distribution, rank and museum-threshold decision."""
    threshold = float(bundle["config"]["short_form_threshold"].get(language, 0.8))
    appella = get_appellation(model, language, wording)
    candidates = model["persons"].get(language, [])

    if appella and appella.get("kind") == "full" and appella.get("person_id"):
        pid = appella["person_id"]
        dist = {c: (1.0 if c == pid else 0.0) for c in candidates}
        ranked = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))
        rank = [p for p, _ in ranked].index(intended_person_id) + 1 if intended_person_id in dist else None
        target_p = dist.get(intended_person_id) if intended_person_id else None
        top = ranked[0][0] if ranked else pid
        approved = (not intended_person_id) or (pid == intended_person_id)
        return {
            "language": language,
            "wording": wording,
            "normalized_wording": norm_appellation(wording),
            "reference_rule": "full_name_unique",
            "mapped_person_id": pid,
            "intended_person_id": intended_person_id,
            "site_id": site_id,
            "last_person_id": last_person_id,
            "distribution": dist,
            "ranking": [p for p, _ in ranked],
            "top_person_id": top,
            "intended_rank": rank,
            "intended_probability": target_p,
            "threshold": threshold,
            "top_is_intended": bool(intended_person_id and top == intended_person_id),
            "passes_threshold": approved,
            "approved": approved,
            "reason": "full name maps to exactly one person" if approved else "full name maps to a different person",
        }

    q = distribution_for_query(model, language, wording, site_id, last_person_id)
    dist = q["distribution"]
    ranked = q["ranked"]
    top = ranked[0][0] if ranked else ""
    rank = [p for p, _ in ranked].index(intended_person_id) + 1 if intended_person_id in dist else None
    target_p = dist.get(intended_person_id, 0.0) if intended_person_id else None
    top_ok = bool(intended_person_id and top == intended_person_id)
    threshold_ok = target_p is not None and target_p + 1e-12 >= threshold
    approved = top_ok and threshold_ok
    if not intended_person_id:
        approved = True
        reason = "neutral wording has no person target"
    elif approved:
        reason = "intended person ranks first and reaches threshold"
    elif not top_ok:
        reason = f"top candidate is {top}, not intended {intended_person_id}"
    else:
        reason = f"intended probability {target_p:.4f} below threshold {threshold:.2f}"
    return {
        "language": language,
        "wording": wording,
        "normalized_wording": norm_appellation(wording),
        "reference_rule": "learned_short_form",
        "appellation_metadata": appella,
        "intended_person_id": intended_person_id,
        "site_id": site_id,
        "last_person_id": last_person_id,
        "distribution": dist,
        "ranking": [p for p, _ in ranked],
        "top_person_id": top,
        "intended_rank": rank,
        "intended_probability": target_p,
        "threshold": threshold,
        "top_is_intended": top_ok,
        "passes_threshold": bool(threshold_ok),
        "approved": approved,
        "reason": reason,
        "components": q["components"],
    }


def evaluate_holdout(bundle: Dict[str, Any], seed: Optional[int] = None) -> Dict[str, Any]:
    cfg = bundle["config"]
    seed = int(cfg["seed"] if seed is None else seed)
    split = group_holdout(
        [r["script_id"] for r in bundle["records"]],
        seed=seed,
        fraction=float(cfg["holdout_fraction"]),
    )
    train_records = [r for r in bundle["records"] if r["script_id"] in set(split["train"])]
    test_records = [r for r in bundle["records"] if r["script_id"] in set(split["test"])]
    train_model = fit_referent_model(bundle, train_records, fitted_on="train_holdout")

    losses = []
    per_record = []
    for r in test_records:
        q = distribution_for_query(
            train_model,
            r["language"],
            r["heard_wording"],
            r["site_id"],
            r.get("last_person_id", ""),
        )
        p_chosen = q["distribution"].get(r["chosen_person_id"], 1e-12)
        loss = -math.log(max(p_chosen, 1e-300))
        row = {
            **r,
            "probability_assigned_to_chosen": p_chosen,
            "log_loss_nats": loss,
            "distribution": q["distribution"],
            "model_top_person_id": q["ranked"][0][0] if q["ranked"] else "",
            "model_top_probability": q["ranked"][0][1] if q["ranked"] else None,
        }
        losses.append(loss)
        per_record.append(row)

    mean_nats = sum(losses) / len(losses) if losses else None
    mishearings = [r for r in per_record if r.get("intended_person_id") and r["chosen_person_id"] != r["intended_person_id"]]
    mishearings_sorted = sorted(mishearings, key=lambda r: (-r["log_loss_nats"], r["record_id"]))
    wanted = int(cfg.get("mishearing_count", 2))
    selected = mishearings_sorted[:wanted]
    scope = "held_out_scripts"
    if len(selected) < wanted:
        seen = {r["record_id"] for r in selected}
        all_model = fit_referent_model(bundle, fitted_on="all_for_mishearing_fallback")
        fallback = []
        for r in bundle["records"]:
            if not r.get("intended_person_id") or r["chosen_person_id"] == r["intended_person_id"]:
                continue
            if r["record_id"] in seen:
                continue
            q = distribution_for_query(all_model, r["language"], r["heard_wording"], r["site_id"], r.get("last_person_id", ""))
            p = q["distribution"].get(r["chosen_person_id"], 1e-12)
            fallback.append({**r, "probability_assigned_to_chosen": p, "log_loss_nats": -math.log(max(p, 1e-300)),
                             "distribution": q["distribution"],
                             "model_top_person_id": q["ranked"][0][0] if q["ranked"] else "",
                             "model_top_probability": q["ranked"][0][1] if q["ranked"] else None,
                             "holdout_scope": "all_scripts_fallback"})
        fallback.sort(key=lambda r: (-r["log_loss_nats"], r["record_id"]))
        selected.extend(fallback[: wanted - len(selected)])
        scope = "held_out_then_all_scripts_fallback"

    for row in selected:
        row.setdefault("holdout_scope", "held_out_scripts")

    language_metrics = {}
    for lang in sorted({r["language"] for r in test_records}):
        rows = [r for r in per_record if r["language"] == lang]
        lang_losses = [r["log_loss_nats"] for r in rows]
        lang_errors = [
            r for r in rows
            if r.get("intended_person_id") and r["chosen_person_id"] != r["intended_person_id"]
        ]
        lang_selected = sorted(lang_errors, key=lambda r: (-r["log_loss_nats"], r["record_id"]))[:wanted]
        language_metrics[lang] = {
            "n_heldout_records": len(rows),
            "mean_log_loss_nats": sum(lang_losses) / len(lang_losses) if lang_losses else None,
            "mean_log_loss_bits": (sum(lang_losses) / len(lang_losses) / math.log(2)) if lang_losses else None,
            "n_actual_mishearings": len(lang_errors),
            "mishearing_records": lang_selected,
        }

    # The headline deliverable is two records; choose one reproducible error per
    # language when both languages have one, otherwise retain the global top two.
    balanced = []
    for lang in ("zh", "en"):
        lang_rows = language_metrics.get(lang, {}).get("mishearing_records", [])
        if lang_rows:
            balanced.append(lang_rows[0])
    if len(balanced) == 2:
        selected = balanced
        scope = "one_per_language_held_out_scripts"

    return {
        "seed": seed,
        "split": split,
        "n_train_records": len(train_records),
        "n_heldout_records": len(test_records),
        "mean_log_loss_nats": mean_nats,
        "mean_log_loss_bits": mean_nats / math.log(2) if mean_nats is not None else None,
        "per_language": language_metrics,
        "n_actual_mishearings_in_holdout": len(mishearings),
        "mishearing_selection_scope": scope,
        "selected_mishearing_records": selected,
        "heldout_record_losses": sorted(per_record, key=lambda r: r["record_id"]),
        "train_model": train_model,
    }
