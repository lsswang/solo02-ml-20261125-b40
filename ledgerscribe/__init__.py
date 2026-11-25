"""LedgerScribe: 离线手写账页誊录（纯标准库，CPU 可复跑）。

数据约定（所有材料均为同格式 JSON，见 README）：

* 行 line：{"line_id", "tokens"}，tokens 是方向整数 0..7 与抬笔 "U" 的混合序列；
  训练行额外带 "amount" 标签，标签只有金额，不含字界。
* 页 page：{"page_id", "writer", "threshold_ratio", "total",
  "candidates": {line_id: [金额...]}, "line_ids": [...]}，
  观察到的行数据由同 registry 中的行文件提供。
"""

UP = "U"
DIRECTIONS = tuple(range(8))

__all__ = ["UP", "DIRECTIONS"]
