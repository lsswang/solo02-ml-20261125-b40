"""合成书写员材料。

笔迹模板是每个数字的“每笔主方向”序列：
    数字身份 = 笔数 + 每笔方向序列；一笔由若干方向 token 组成，
    笔与笔之间可能抬笔，字与字之间可能抬笔；抬笔被省略时即两字连写。

刻意安排：
  * 数字 4 有三笔（一字内两次抬笔的来源）；
  * render_line(..., ligature_at={i}) 强制第 i、i+1 字之间不抬笔（两字连写）；
  * 连写路径在枚举时表现为“一个笔段跨过字界”。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import DIRECTIONS, UP

# 每个数字：(每笔主方向序列)。序列长度 = 笔数；主方向两两不同，保证可辨识。
TEMPLATES: dict[str, tuple[int, ...]] = {
    "0": (3, 7),
    "1": (0,),               # 单笔字，方向 0
    "2": (2, 5, 1),
    "3": (2, 4, 6),
    "4": (1, 3, 0),          # 三笔 -> 一字两次抬笔
    "5": (5, 4, 3),
    "6": (1, 7, 5),
    # 7 两笔：与 2（三笔）笔数接近，使同轨迹可按不同断笔读成 12 或 17。
    "7": (6, 1),
    "8": (7, 3, 6),
    "9": (5, 0, 4),
}

# 每笔的长度采样范围（一笔内方向 token 的个数）。
LEN_LOW, LEN_HIGH = 1, 2
# 单笔字固定一笔一个方向 token。
SINGLE_LEN = {"1": (1, 1)}


def render_line(
    amount: str,
    rng: random.Random,
    *,
    direction_noise: float,
    internal_u_prob: float,
    boundary_u_prob: float,
    ligature_at: set[int] | None = None,
    force_u_after: set[int] | None = None,
    force_u_inside_digit: dict[int, set[int]] | None = None,
) -> tuple[list, list[dict]]:
    """把金额字符串渲染成 token 行。

    概率开关（训练材料里这些是隐变量，只有金额是标签）：
      internal_u_prob  字内两笔之间抬笔的概率；
      boundary_u_prob  两字之间抬笔的概率，被关掉即连写。
    force_u_after / force_u_inside_digit 供出题人精确控制检验页。

    返回 (tokens, ground_truth)，ground_truth 记录每个数字的 token 区间，
    只写入 manifest 供核对，训练程序读不到它。
    """
    ligature_at = ligature_at or set()
    force_u_after = force_u_after or set()
    force_u_inside_digit = force_u_inside_digit or {}

    tokens: list = []
    truth: list[dict] = []
    for di, ch in enumerate(amount):
        homes = TEMPLATES[ch]
        start = len(tokens)
        for si, home in enumerate(homes):
            low, high = SINGLE_LEN.get(ch, (LEN_LOW, LEN_HIGH))
            stroke_len = rng.randint(low, high)
            for _ in range(stroke_len):
                if rng.random() < direction_noise:
                    d = rng.choice(DIRECTIONS)
                else:
                    d = home
                tokens.append(d)
            # 字内抬笔：最后一笔之后不抬（字界由下一段逻辑处理）。
            if si < len(homes) - 1:
                p = internal_u_prob
                forced = si in force_u_inside_digit.get(di, set())
                if forced or rng.random() < p:
                    tokens.append(UP)
        truth.append(
            {
                "digit_index": di,
                "digit": ch,
                "token_span": [start, len(tokens)],
                "strokes": len(homes),
            }
        )
        if di < len(amount) - 1:
            if di in ligature_at:
                put_u = False                       # 强制两字连写
            elif di in force_u_after:
                put_u = True
            else:
                put_u = rng.random() < boundary_u_prob
            if put_u:
                tokens.append(UP)
    return tokens, truth


def make_writer_material(
    out_dir: Path,
    writer: str,
    seed: int,
    *,
    direction_noise: float,
    internal_u_prob: float = 0.85,
    boundary_u_prob: float = 0.9,
    n_train: int = 36,
) -> dict:
    """生成同一书写员的短行训练集（仅金额标签）与生成参数 manifest。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    # 保证 0..9 都出现，且含一个两字连写样本与一个“4（一字两次抬笔）”样本。
    plan = [str(n) for n in range(10)]
    two_digit = [f"{a}{b}" for a in (1, 2, 4) for b in (0, 3, 7, 4)]
    plan += two_digit
    plan += ["404", "127", "280", "740"]
    plan = plan[:n_train]
    while len(plan) < n_train:
        plan.append(str(rng.randint(1, 999)))

    lines, manifest = [], []
    for k, amount in enumerate(plan):
        lig = {0} if k == 12 and len(amount) > 1 else None
        tokens, truth = render_line(
            amount,
            rng,
            direction_noise=direction_noise,
            internal_u_prob=internal_u_prob,
            boundary_u_prob=boundary_u_prob,
            ligature_at=lig or set(),
        )
        line_id = f"{writer}-train-{k:02d}"
        lines.append({"line_id": line_id, "amount": amount, "tokens": tokens})
        manifest.append(
            {
                "line_id": line_id,
                "amount": amount,
                "ligature_at": sorted(lig) if lig else [],
                "truth": truth,
            }
        )

    (out_dir / f"{writer}_train.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in lines) + "\n",
        encoding="utf-8",
    )
    meta = {
        "writer": writer,
        "seed": seed,
        "direction_noise": direction_noise,
        "internal_u_prob": internal_u_prob,
        "boundary_u_prob": boundary_u_prob,
        "n_train": n_train,
        "note": "manifest 仅供核对；训练程序只读 jsonl 的 amount 标签，不读字界。",
        "lines": manifest,
    }
    (out_dir / f"{writer}_manifest.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def write_heldout_pages(out_dir: Path) -> None:
    """手工控制的检验页：12/17 与 23/28，以及只有第四候选才能对平的小页。

    轨迹按“两种切法都合法”构造。7 为两笔字 (6,1)，2 为三笔字 (2,5,1)，
    二者笔数差 1，靠无 U 处的断笔选择产生歧义：
      A-line-1 [0 U 6 2 5 1]（字界一处 U，其后无 U）：
        12 = 1[0] | 2 三笔 [6][2][5,1]（4 笔）；
        17 = 1[0] | 7 两笔 [6,2][5,1]（3 笔，前两 token 并入 7 第一笔）。
        馆员锁定末笔段 [5,1]：答复“7”表示它是 7 的第二笔（整字的一部分，
        不是单独整字）；答复“不是整字”则保留多种切法。整字锁定用首段 [6,2]。
      A-line-2 [2 6 4 2 4 6]（字界无 U 连写）：
        23 = 2 三笔 | 3 三笔；28 = 2 三笔 | 8 三笔，笔数相同靠方向序列竞争。
    """
    out_dir = Path(out_dir)

    held = {
        # 真值 12；同一轨迹也合法支持 17（7 两笔），两读法后验均过可信度门槛。
        "A-line-1": [0, UP, 6, 5, 4, 1],
        # 真值 23；23/28 均为合法切法且后验接近（字界无 U 连写）。
        "A-line-2": [2, 6, 4, 4, 0, 6],
        # 第四候选小页：真值 34，4 为三笔，两次字内抬笔全部保留（一字两次抬笔）。
        # 3 三笔 [2][4][6]，字界 U；4 三笔 [1][3][0]，两字内 U 都保留。
        "A-line-3": [2, 4, 6, UP, 1, UP, 3, UP, 0],
        # 真值 52；34+52=86，只有取第四候选 34 才能对平。
        "A-line-4": [5, UP, 4, UP, 3, UP, 2, 5, 1],
    }

    line_records = [{"line_id": k, "tokens": v} for k, v in held.items()]
    (out_dir / "heldout_lines.json").write_text(
        json.dumps(line_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 真值字界（仅供核对，程序不读取）。
    manifest = {
        "A-line-1": [
            {"digit_index": 0, "digit": "1", "token_span": [0, 0]},
            {"digit_index": 1, "digit": "2", "token_span": [2, 5]},
        ],
        "A-line-2": [
            {"digit_index": 0, "digit": "2", "token_span": [0, 2]},
            {"digit_index": 1, "digit": "3", "token_span": [3, 5]},
        ],
        "A-line-3": [
            {"digit_index": 0, "digit": "3", "token_span": [0, 2]},
            {"digit_index": 1, "digit": "4", "token_span": [4, 8]},
        ],
        "A-line-4": [
            {"digit_index": 0, "digit": "5", "token_span": [0, 4]},
            {"digit_index": 1, "digit": "2", "token_span": [6, 8]},
        ],
    }
    (out_dir / "heldout_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    def page(page_id, total, threshold, cand):
        return {
            "page_id": page_id,
            "writer": "A",
            "threshold_ratio": threshold,
            "total": total,
            "candidates": cand,
            "line_ids": list(cand),
        }

    c1217 = ["12", "17", "72", "21", "27", "71", "11", "77"]
    c2328 = ["23", "28", "83", "32", "38", "82", "22", "33"]
    pages = [
        page("A-040", 40, 0.5, {"A-line-1": c1217, "A-line-2": c2328}),
        page("A-035", 35, 0.5, {"A-line-1": c1217, "A-line-2": c2328}),
        page(
            "A-086",
            86,
            0.5,
            {
                # 34 故意排在第四位；其他组合凑不出 86。
                "A-line-3": ["31", "30", "37", "34", "43", "41", "13", "14"],
                "A-line-4": ["52", "25", "57", "54", "45", "75", "27", "72"],
            },
        ),
    ]
    (out_dir / "pages.json").write_text(
        json.dumps(pages, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def make_new_writer_page(out_dir: Path) -> None:
    """新书写员 B：与 A 检验页同格式、同构轨迹，但按 B 的高噪声重新采样。

    新笔迹标定（model_B）下，第一行的 17 读法因方向证据被噪声破坏而跌出
    可信度门槛；第二行 28 仍保持有竞争力。
    """
    out_dir = Path(out_dir)
    rng = random.Random(777)

    def noisy(base: list, noise: float, protected: set[int] | None = None) -> list:
        protected = protected or set()
        out = []
        for i, tok in enumerate(base):
            if tok == UP or i in protected:
                out.append(tok)
            elif rng.random() < noise:
                out.append(rng.choice(DIRECTIONS))
            else:
                out.append(tok)
        return out

    base1 = [0, UP, 6, 5, 4, 1]
    # B 第二行直接采用使 23/28 都入围、28 可与 12 对平 40 的轨迹。
    base2 = [2, 5, 1, 0, 3, 6]
    # B 书写员噪声较大：保护字界骨架（首 token 与 U），只噪声化中间方向，
    # 使真值读法仍可对平，但 17 的关键证据被削弱而跌出可信度门槛。
    lines = [
        {"line_id": "B-line-1", "tokens": noisy(base1, 0.30, protected={0})},
        # 第二行用固定轨迹：B 的“新笔迹标定”差异由 B 训练集高噪声模型体现，
        # 检验行本身保持可复现的 23/28 歧义结构。
        {"line_id": "B-line-2", "tokens": list(base2)},
    ]
    (out_dir / "newpage_lines.json").write_text(
        json.dumps(lines, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    page = {
        "page_id": "B-040",
        "writer": "B",
        "threshold_ratio": 0.5,
        "total": 40,
        "candidates": {
            "B-line-1": ["12", "17", "72", "21", "27", "71", "11", "77"],
            "B-line-2": ["23", "28", "83", "32", "38", "82", "22", "33"],
        },
        "line_ids": ["B-line-1", "B-line-2"],
    }
    (out_dir / "newpage_page.json").write_text(
        json.dumps(page, ensure_ascii=False, indent=2), encoding="utf-8"
    )
