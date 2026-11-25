# Guide Studio run summary

- Seed: `20261125`
- Closed stations: `none`
- Whole-script holdout log loss: `0.1691` nats
- Two highlighted mishearing records: `R-ZH-0078, R-EN-0158`

## Plans
- `wall/en`: objective **88.00** (duration 60.00 + production 28.00), recordings 7, wall bonus at ['silent_porch']
- `wall/zh`: objective **88.00** (duration 60.00 + production 28.00), recordings 7, wall bonus at ['silent_porch']
- `pen/en`: objective **87.00** (duration 59.00 + production 28.00), recordings 7/7
- `pen/zh`: objective **87.00** (duration 59.00 + production 28.00), recordings 7/7

## Portrait station
- wall/en/east: `en_portrait_full` [Lin Lan], P(intended)=1.000, incoming=`lin`
- wall/en/west: `en_portrait_full` [Lin Lan], P(intended)=1.000, incoming=`zhou`
- wall/zh/east: `zh_portrait_full` [林岚], P(intended)=1.000, incoming=`lin`
- wall/zh/west: `zh_portrait_full` [林岚], P(intended)=1.000, incoming=`zhou`
- pen/en/east: `en_portrait_she` [she], P(intended)=0.874, incoming=`lin`
- pen/en/west: `en_portrait_full` [Lin Lan], P(intended)=1.000, incoming=`zhou`
- pen/zh/east: `zh_portrait_she` [她], P(intended)=0.874, incoming=`lin`
- pen/zh/west: `zh_portrait_full` [林岚], P(intended)=1.000, incoming=`zhou`

## Proofreading
- [PASS] `ML-EN-01` en P=0.874: intended person ranks first and reaches threshold
- [PASS] `ML-EN-02` en P=1.000: full name maps to exactly one person
- [PASS] `ML-EN-03` en P=1.000: full name maps to exactly one person
- [PASS] `ML-ZH-01` zh P=0.874: intended person ranks first and reaches threshold
- [PASS] `ML-ZH-02` zh P=1.000: full name maps to exactly one person
- [PASS] `ML-ZH-03` zh P=1.000: full name maps to exactly one person
