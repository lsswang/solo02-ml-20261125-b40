# Guide Studio run summary

- Seed: `20261125`
- Closed stations: `west_gate`
- Whole-script holdout log loss: `0.1691` nats
- Two highlighted mishearing records: `R-ZH-0078, R-EN-0158`

## Plans
- `wall/en`: objective **46.00** (duration 30.00 + production 16.00), recordings 4
- `wall/zh`: objective **46.00** (duration 30.00 + production 16.00), recordings 4
- `pen/en`: objective **46.00** (duration 30.00 + production 16.00), recordings 4/7
- `pen/zh`: objective **46.00** (duration 30.00 + production 16.00), recordings 4/7

## Portrait station
- wall/en/east: `en_portrait_she` [she], P(intended)=0.874, incoming=`lin`
- wall/zh/east: `zh_portrait_she` [她], P(intended)=0.874, incoming=`lin`
- pen/en/east: `en_portrait_she` [she], P(intended)=0.874, incoming=`lin`
- pen/zh/east: `zh_portrait_she` [她], P(intended)=0.874, incoming=`lin`

## Proofreading
- [PASS] `ML-EN-01` en P=0.874: intended person ranks first and reaches threshold
- [REPLACE] `ML-EN-02` en P=0.100: top candidate is zhou, not intended lin
- [PASS] `ML-EN-03` en P=1.000: full name maps to exactly one person
- [PASS] `ML-ZH-01` zh P=0.874: intended person ranks first and reaches threshold
- [REPLACE] `ML-ZH-02` zh P=0.100: top candidate is zhou, not intended lin
- [PASS] `ML-ZH-03` zh P=1.000: full name maps to exactly one person
