# Guide Studio run summary

- Seed: `20261125`
- Closed stations: `none`
- Whole-script holdout log loss: `0.2046` nats
- Two highlighted mishearing records: `R-ZH-0078, R-EN-0158`

## Plans
- `wall/en`: objective **80.00** (duration 52.00 + production 28.00), recordings 7, wall bonus at ['silent_porch']
- `wall/zh`: objective **80.00** (duration 52.00 + production 28.00), recordings 7, wall bonus at ['silent_porch']
- `pen/en`: objective **80.00** (duration 52.00 + production 28.00), recordings 7/7
- `pen/zh`: objective **80.00** (duration 52.00 + production 28.00), recordings 7/7

## Portrait station
- wall/en/east: `en_portrait_she` [she], P(intended)=0.901, incoming=`lin`
- wall/en/west: `en_portrait_she` [she], P(intended)=0.834, incoming=`zhou`
- wall/zh/east: `zh_portrait_she` [她], P(intended)=0.901, incoming=`lin`
- wall/zh/west: `zh_portrait_she` [她], P(intended)=0.834, incoming=`zhou`
- pen/en/east: `en_portrait_she` [she], P(intended)=0.901, incoming=`lin`
- pen/en/west: `en_portrait_she` [she], P(intended)=0.834, incoming=`zhou`
- pen/zh/east: `zh_portrait_she` [她], P(intended)=0.901, incoming=`lin`
- pen/zh/west: `zh_portrait_she` [她], P(intended)=0.834, incoming=`zhou`

## Proofreading
- [PASS] `ML-EN-01` en P=0.901: intended person ranks first and reaches threshold
- [PASS] `ML-EN-02` en P=0.834: intended person ranks first and reaches threshold
- [PASS] `ML-EN-03` en P=1.000: full name maps to exactly one person
- [PASS] `ML-ZH-01` zh P=0.901: intended person ranks first and reaches threshold
- [PASS] `ML-ZH-02` zh P=0.834: intended person ranks first and reaches threshold
- [PASS] `ML-ZH-03` zh P=1.000: full name maps to exactly one person
