# Route comparison

Chosen complete route: **fixed per-partition tracks**.

| Route / observation | Latency | Result |
|---|---:|---|
| Fixed per-state tracks | 0 samples in the scene gap | partition_open=PASS, partition_half=PASS, partition_closed=PASS |
| One universal fixed track | 0 samples in the scene gap | all-state verification: True |
| Early probe | 160 samples (0.040s) | candidates: partition_half, partition_closed |
| Full probe | 1176 samples (0.294s) | candidates: partition_closed |

The early read reaches a partial decision at 0.040s, but probe echo clearance and safe narration start require 1160 samples (0.290s). The early probe deliberately cannot separate `partition_half` from `partition_closed`; that branch uses one robust WAV checked against both learned rooms rather than an unverified state-specific choice.
