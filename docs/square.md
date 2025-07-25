# Square Matrix Performance Benchmarks (FP32)

Performance measured on AMD Radeon RX 7900 GRE on WSL2 (Ubuntu 24.04.1 LTS, ROCm 6.4.1) in TFLOPs.

## FP32 Performance Results

| Matrix Size    | Input Layout (A,B) | `rocm_sgemm`<br>(C=col)    | `rocBLAS`<br>(C=col)| Ratio<br>(C=col / rocBLAS)| `rocm_sgemm`<br>(C=row)    | Ratio<br>(C=row / rocBLAS)|
|:---------------|:-------------------|---------------------------:|--------------------:|--------------------------:|---------------------------:|--------------------------:|
| 1024×1024×1024 | col, col           |                      12.69 |               11.51 |                      1.10 |                      11.61 |                      1.01 |
| 1024×1024×1024 | row, col           |                       9.40 |               11.21 |                      0.84 |                       9.82 |                      0.88 |
| 1024×1024×1024 | col, row           |                      11.65 |               11.71 |                      0.99 |                      12.87 |                      1.10 |
| 1024×1024×1024 | row, row           |                      12.55 |               11.52 |                      1.09 |                      11.81 |                      1.02 |
| 2048×2048×2048 | col, col           |                      18.53 |               18.72 |                      0.99 |                      17.95 |                      0.96 |
| 2048×2048×2048 | row, col           |                      15.54 |               16.97 |                      0.92 |                      16.49 |                      0.97 |
| 2048×2048×2048 | col, row           |                      22.08 |               19.48 |                      1.13 |                      22.49 |                      1.15 |
| 2048×2048×2048 | row, row           |                      20.70 |               17.15 |                      1.21 |                      20.53 |                      1.20 |
| 4096×4096×4096 | col, col           |                      24.10 |               20.13 |                      1.20 |                      24.61 |                      1.22 |
| 4096×4096×4096 | row, col           |                      21.22 |               19.26 |                      1.10 |                      21.15 |                      1.10 |
| 4096×4096×4096 | col, row           |                      26.47 |               21.17 |                      1.25 |                      27.84 |                      1.32 |
| 4096×4096×4096 | row, row           |                      25.66 |               20.34 |                      1.26 |                      26.15 |                      1.29 |
| 8192×8192×8192 | col, col           |                      25.38 |               19.33 |                      1.31 |                      26.79 |                      1.39 |
| 8192×8192×8192 | row, col           |                      21.16 |               19.70 |                      1.07 |                      21.26 |                      1.08 |
| 8192×8192×8192 | col, row           |                      27.46 |               20.40 |                      1.35 |                      28.05 |                      1.37 |
| 8192×8192×8192 | row, row           |                      24.27 |               20.12 |                      1.21 |                      26.07 |                      1.30 |
