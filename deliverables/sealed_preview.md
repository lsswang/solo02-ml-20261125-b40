# 替代路线预览：销毁读数、封存联合因子

- 封存模型：`cal-v1-initial`
- 模型指纹：`2d7e73780b939bfc…`
- 封存因子重判概率：TOP=0.9817, ZONE=0.0183, NO_ZONE=0.0000
- 证词 A：`T-A-top-witness`，若逐帧为真则类别 `TOP`
- 证词 B：`T-B-zone-coach`，若逐帧为真则类别 `ZONE`
- log Bayes factor A/B：10.392
- 解释：**A is better supported by sealed joint evidence**

## 相抵触帧

| 帧 | 证词 A | 证词 B | 发射似然比 A/B | 转移比 A/B | 联合帧路径比 A/B | 因子更支持 |
|---:|---|---|---:|---:|---:|---|
| 6 | TOP|TOP | ZONE|ZONE | 45.5 | 0.0112 | 0.508 | A |
| 7 | TOP|TOP | ZONE|ZONE | 45.5 | 0.881 | 40 | A |
| 8 | TOP|TOP | ZONE|ZONE | 45.5 | 0.881 | 40 | A |
| 9 | TOP|TOP | ZONE|ZONE | 45.5 | 0.881 | 40 | A |

> Raw readings are absent. A later emission calibration cannot be applied to these factors unless the new likelihoods were also sealed before readings destruction.
