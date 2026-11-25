# Required calibration diagnostics

| Case | Speaker peaks | Near RMSE | Far leakage RMS | Verdict |
|---|---:|---:|---:|---|
| Sum/diff original | 0.500, 0.500 | 1.38e-17 | 3.22e-17 | PASS / x1=x2=target/2 |
| Far also receives sum | 0.500, 0.500 | 1.52e-06 | 0.631 | impossible to isolate |
| Speaker 2 unplugged | 0.500, 0 | 0.316 | 0.316 | impossible to satisfy both |
