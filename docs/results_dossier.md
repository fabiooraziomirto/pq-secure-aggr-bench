# Dossier dei risultati (generato il 2026-11-02 01:07)

Raccolta di tutti i numeri disponibili, con fonte, numerosità e limite. Si rigenera con `scripts/build_dossier.py`. Non è il manoscritto: è la base per valutare cosa si può affermare.

## 0. Stato delle evidenze

| Evidenza | Stato | Campione | Fonte |
| --- | --- | --- | --- |
| Costi per round, n = 3, d = 1e5…2,1e7 | disponibile | 30 round (14 per HE a 3,4M), una sessione | runs/campaign-3 |
| Scalabilità in n (5, 10, 20) | disponibile, solo calcolo | 30 round | runs/campaign-3 |
| Replay di rete LAN/metro/WAN | disponibile, 3 celle WAN con perturbazioni | 6–30 campioni, una sessione | runs/campaign-3/net |
| Costo a dimensione reale 8.409.600 e N a codifica uguale | **in attesa (T14)** | - | - |
| Ripetizione celle WAN perturbate e sessioni indipendenti | **in attesa (T15)** | - | - |
| Utilità e attacchi, pilota a 1 seed | disponibile | seed 0 | runs/pilot |
| Utilità e attacchi, campagna finale 5 seed | **non ancora lanciata** | seed 1–5 | runs/final |
| Inversione dei gradienti come controllo positivo | **in attesa (T13b)** | - | - |
| Dropout (esplorativo) | disponibile | 10 sessioni, A e B-BGV | runs/dropout |
| Baseline TLS 1.3 ibrido | disponibile | 40 handshake | runs/tls-baseline.json |
| Contabile DP contro Opacus | fatto | - | docs/decisions.md (1 nov) |
| Revisione della crittografia (combiner, flooding B/B-BGV) | **non fatta** (richiede un esperto) | - | - |

## 1. Costi: cella principale (n = 3, d = 3.407.872, dimensione di un adapter Llama-like)

| Scheme | Round payload (MB) | x A | Setup (MB) | Crypto s (p50) | Peak RSS (MB) | LAN s (p50) | WAN s (p50, n) | WAN x A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | 81.79 | 0.80 | 0.00 | 0.000 | 309 | 0.068 | 19.95 (30) | 0.81 |
| A | 102.24 | 1.00 | 0.02 | 0.185 | 689 | 0.083 | 24.50 (30) | 1.00 |
| C | 204.47 | 2.00 | 0.02 | 1.281 | 929 | 0.147 | 39.77 (30) | 1.62 |
| B | 2540.48 | 24.85 | 3.17 | 45.982 | 3243 | 2.374 | 546.38 (6) | 22.31 |
| B-BGV | 2129.70 | 20.83 | 5.27 | 35.202 | 2756 | 1.992 | 468.67 (7) | 19.13 |

Tempi: somma seriale dei tempi misurati nel simulatore, non un tempo distribuito; la rete è un replay del registro dei trasferimenti tra container sullo stesso host condiviso. WAN: 50 ms, 100 Mbit/s, perdita 0,1 %.

## 2. Costi: tabelle complete della campagna

### 1. Cost per round, three sites (group main)

| scheme | elements | rounds | setup bytes | round MB (p50) | crypto s p50/p95/p99 | site-max s p50 | reducer s p50 | peak RSS MB (whole process) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | 100000 | 30 | 0 | 2.40 | 0.000/0.000/0.000 | 0.000 | 0.001 | 43.9 |
| A | 100000 | 30 | 23,199 | 3.00 | 0.006/0.010/0.010 | 0.013 | 0.007 | 59.2 |
| B | 100000 | 30 | 3,171,328 | 79.23 | 1.417/1.470/1.482 | 0.287 | 0.587 | 161.8 |
| B-BGV | 100000 | 30 | 5,269,728 | 71.32 | 1.244/1.356/1.373 | 0.311 | 0.367 | 163.4 |
| C | 100000 | 30 | 23,199 | 6.00 | 0.039/0.044/0.052 | 0.022 | 0.007 | 67.2 |
| N | 1000000 | 30 | 0 | 24.00 | 0.000/0.000/0.000 | 0.003 | 0.004 | 117.0 |
| A | 1000000 | 30 | 23,199 | 30.00 | 0.052/0.060/0.061 | 0.122 | 0.066 | 228.8 |
| B | 1000000 | 30 | 3,171,328 | 750.97 | 13.586/13.893/14.018 | 2.715 | 5.610 | 1086.3 |
| B-BGV | 1000000 | 30 | 5,269,728 | 634.43 | 10.018/10.978/11.335 | 2.489 | 3.214 | 855.6 |
| C | 1000000 | 30 | 23,199 | 60.00 | 0.370/0.391/0.398 | 0.231 | 0.067 | 301.2 |
| N | 3407872 | 30 | 0 | 81.79 | 0.000/0.000/0.000 | 0.006 | 0.014 | 309.1 |
| A | 3407872 | 30 | 23,199 | 102.24 | 0.185/0.211/0.218 | 0.428 | 0.233 | 689.2 |
| B | 3407872 | 14 | 3,171,328 | 2540.48 | 45.982/47.068/47.253 | 9.163 | 19.031 | 3242.6 |
| B-BGV | 3407872 | 14 | 5,269,728 | 2129.70 | 35.202/36.265/36.463 | 8.414 | 10.849 | 2755.5 |
| C | 3407872 | 30 | 23,199 | 204.47 | 1.281/1.359/1.402 | 0.823 | 0.235 | 929.0 |
| N | 20971520 | 30 | 0 | 503.32 | 0.000/0.000/0.000 | 0.122 | 0.291 | 1497.8 |
| A | 20971520 | 30 | 23,199 | 629.15 | 2.085/2.123/2.128 | 3.262 | 1.613 | 3515.0 |
| C | 20971520 | 30 | 23,199 | 1258.29 | 9.768/9.920/9.950 | 5.844 | 1.590 | 5013.8 |

### 2. Scalability in the number of sites (group scaling, compute only)

| scheme | sites | elements | setup bytes | round MB (p50) | crypto s p50 | cpu total s p50 | peak RSS MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N | 5 | 100000 | 0 | 4.00 | 0.000 | 0.002 | 48.7 |
| A | 5 | 100000 | 77,330 | 5.00 | 0.019 | 0.078 | 61.3 |
| B | 5 | 100000 | 6,342,912 | 132.06 | 2.274 | 2.279 | 198.5 |
| B-BGV | 5 | 100000 | 10,539,816 | 118.87 | 2.048 | 2.061 | 192.8 |
| C | 5 | 100000 | 77,330 | 15.00 | 0.125 | 0.178 | 81.6 |
| N | 10 | 100000 | 0 | 8.00 | 0.000 | 0.002 | 57.4 |
| A | 10 | 100000 | 347,985 | 10.00 | 0.072 | 0.177 | 69.9 |
| B | 10 | 100000 | 14,271,872 | 264.11 | 4.352 | 4.357 | 282.5 |
| B-BGV | 10 | 100000 | 23,715,036 | 237.73 | 4.081 | 4.106 | 274.3 |
| C | 10 | 100000 | 347,985 | 55.00 | 0.576 | 0.687 | 134.1 |
| N | 20 | 100000 | 0 | 16.00 | 0.000 | 0.004 | 71.8 |
| A | 20 | 100000 | 1,469,270 | 20.00 | 0.298 | 0.515 | 83.5 |
| B | 20 | 100000 | 30,129,792 | 528.23 | 8.457 | 8.461 | 454.4 |
| B-BGV | 20 | 100000 | 50,065,476 | 475.47 | 7.969 | 8.015 | 446.5 |
| C | 20 | 100000 | 1,469,270 | 210.01 | 2.383 | 2.593 | 312.5 |

### 3. Network time per round (replay)

| profile | group | scheme | elements | samples | round net s p50/p95/p99 | setup net s p50 | wire/payload | perturbed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lan | main | N | 100000 | 30 | 0.002/0.003/0.004 | 0.000 | 1.004 |  |
| lan | main | A | 100000 | 30 | 0.003/0.004/0.005 | 0.001 | 1.003 |  |
| lan | main | B | 100000 | 30 | 0.072/0.085/0.087 | 0.003 | 1.004 |  |
| lan | main | B-BGV | 100000 | 30 | 0.066/0.076/0.076 | 0.003 | 1.003 |  |
| lan | main | C | 100000 | 30 | 0.006/0.007/0.008 | 0.001 | 1.003 |  |
| lan | main | N | 1000000 | 30 | 0.021/0.026/0.029 | 0.000 | 1.002 |  |
| lan | main | A | 1000000 | 30 | 0.025/0.029/0.030 | 0.001 | 1.003 |  |
| lan | main | B | 1000000 | 30 | 0.718/0.821/0.876 | 0.003 | 1.002 |  |
| lan | main | B-BGV | 1000000 | 30 | 0.616/0.678/0.684 | 0.003 | 1.002 |  |
| lan | main | C | 1000000 | 30 | 0.044/0.055/0.057 | 0.001 | 1.004 |  |
| lan | main | N | 3407872 | 30 | 0.068/0.077/0.083 | 0.000 | 1.002 |  |
| lan | main | A | 3407872 | 30 | 0.083/0.095/0.100 | 0.001 | 1.002 |  |
| lan | main | B | 3407872 | 30 | 2.374/2.661/2.714 | 0.002 | 1.002 |  |
| lan | main | B-BGV | 3407872 | 30 | 1.992/2.202/2.248 | 0.003 | 1.002 |  |
| lan | main | C | 3407872 | 30 | 0.147/0.167/0.171 | 0.001 | 1.003 |  |
| lan | main | N | 20971520 | 30 | 0.444/0.554/0.656 | 0.000 | 1.002 |  |
| lan | main | A | 20971520 | 30 | 0.552/0.630/0.671 | 0.001 | 1.002 |  |
| lan | main | C | 20971520 | 30 | 0.913/1.036/1.049 | 0.001 | 1.002 |  |
| metro | sensitivity | N | 3407872 | 30 | 0.470/0.471/0.471 | 0.000 | 1.003 |  |
| metro | sensitivity | A | 3407872 | 30 | 0.589/0.590/0.590 | 0.031 | 1.003 |  |
| metro | sensitivity | B | 3407872 | 30 | 13.168/13.172/13.173 | 0.044 | 1.003 |  |
| metro | sensitivity | B-BGV | 3407872 | 30 | 11.099/11.103/11.104 | 0.058 | 1.003 |  |
| metro | sensitivity | C | 3407872 | 30 | 0.920/0.938/0.941 | 0.031 | 1.003 |  |
| wan | main | N | 100000 | 30 | 0.857/1.084/1.177 | 0.000 | 1.028 |  |
| wan | main | A | 100000 | 30 | 0.838/1.183/1.395 | 0.151 | 1.025 |  |
| wan | main | B | 100000 | 30 | 20.509/23.166/24.625 | 1.712 | 1.029 |  |
| wan | main | B-BGV | 100000 | 30 | 18.221/22.492/23.407 | 2.991 | 1.028 |  |
| wan | main | C | 100000 | 30 | 1.675/2.166/2.507 | 0.151 | 1.028 |  |
| wan | main | N | 1000000 | 30 | 6.079/8.548/8.887 | 0.000 | 1.028 |  |
| wan | main | A | 1000000 | 30 | 7.803/10.077/10.226 | 0.151 | 1.029 | ⚠ |
| wan | main | B | 1000000 | 20 | 169.504/188.190/190.949 | 2.038 | 1.029 | ⚠ |
| wan | main | B-BGV | 1000000 | 24 | 143.109/155.288/156.539 | 3.119 | 1.030 | ⚠ |
| wan | main | C | 1000000 | 30 | 12.526/14.401/15.519 | 0.151 | 1.029 |  |
| wan | main | N | 3407872 | 30 | 19.949/23.531/24.718 | 0.000 | 1.029 |  |
| wan | main | A | 3407872 | 30 | 24.495/27.262/27.659 | 0.151 | 1.029 |  |
| wan | main | B | 3407872 | 6 | 546.379/562.453/564.837 | 2.061 | 1.030 |  |
| wan | main | B-BGV | 3407872 | 7 | 468.667/483.361/483.647 | 3.037 | 1.030 |  |
| wan | main | C | 3407872 | 30 | 39.767/43.878/44.276 | 0.151 | 1.030 |  |
| wan | main | N | 20971520 | 29 | 112.750/123.361/123.586 | 0.000 | 1.030 |  |
| wan | main | A | 20971520 | 23 | 141.817/154.957/156.564 | 0.151 | 1.030 |  |
| wan | main | C | 20971520 | 15 | 215.119/220.412/221.119 | 0.151 | 1.030 |  |

wire/payload = interface transmit bytes per replay iteration (all actors, includes TCP/IP overhead and control traffic) over the payload bytes of one round.

### 4. Handshake cost (numerator of H6)

| scheme | rekey every | handshakes | bytes p50 | compute s p50 | profile | network s p50 | total s | round time (s) above which it is < 1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 1 | 30 | 23,199 | 0.0028 | lan | 0.001 | 0.004 | 0 |
| A | 1 | 30 | 23,199 | 0.0028 | wan | 0.151 | 0.154 | 15 |
| B | 1 | 30 | 3,171,328 | 0.0754 | lan | 0.003 | 0.078 | 8 |
| B | 1 | 30 | 3,171,328 | 0.0754 | wan | 1.712 | 1.787 | 179 |
| B-BGV | 1 | 30 | 5,269,728 | 0.1238 | lan | 0.003 | 0.127 | 13 |
| B-BGV | 1 | 30 | 5,269,728 | 0.1238 | wan | 2.991 | 3.115 | 311 |
| C | 1 | 30 | 23,199 | 0.0027 | lan | 0.001 | 0.004 | 0 |
| C | 1 | 30 | 23,199 | 0.0027 | wan | 0.151 | 0.154 | 15 |
| A | 10 | 3 | 23,199 | 0.0029 | lan | 0.001 | 0.004 | 0 |
| A | 10 | 3 | 23,199 | 0.0029 | wan | 0.151 | 0.154 | 15 |
| B | 10 | 3 | 3,171,328 | 0.0764 | lan | 0.003 | 0.079 | 8 |
| B | 10 | 3 | 3,171,328 | 0.0764 | wan | 1.712 | 1.788 | 179 |
| B-BGV | 10 | 3 | 5,269,728 | 0.1200 | lan | 0.003 | 0.123 | 12 |
| B-BGV | 10 | 3 | 5,269,728 | 0.1200 | wan | 2.991 | 3.111 | 311 |
| C | 10 | 3 | 23,199 | 0.0039 | lan | 0.001 | 0.005 | 1 |
| C | 10 | 3 | 23,199 | 0.0039 | wan | 0.151 | 0.155 | 16 |

The denominator of H6 is the time of a full round including training, which is not measured here: the last column is the round time that would make the handshake cost under 1 %.


### 5. Perturbation windows and flagged cells

- 21:55:10-22:07:20: pilot 4B-Base: download then GPU run (may perturb the network phase)
- 22:21:25-22:23:26: pilot 4B-Base batch 2 and 1 (GPU run, cap 20 GiB)
- 23:27:35-23:36:04: pilot 2B-Base: download then GPU runs (may perturb the network phase)

- net wan main A-n3-v1000000: overlaps pilot 4B-Base: download then GPU run (may perturb the network phase)
- net wan main B-n3-v1000000: overlaps pilot 4B-Base: download then GPU run (may perturb the network phase), pilot 4B-Base batch 2 and 1 (GPU run, cap 20 GiB)
- net wan main B-BGV-n3-v1000000: overlaps pilot 2B-Base: download then GPU runs (may perturb the network phase)

### 6. Limits

- B and B-BGV at 20971520 elements were skipped (about 2.4 GB of ciphertext per site): extrapolate.
- Heavy wan cells use fewer repetitions (see samples); their p95/p99 are coarse.
- The host is shared (load average 10 on average, cpu busy 20 % on average): timings are not those of a dedicated node.
- Peak RSS is that of the whole simulator process, not per actor; GPU memory is not measured here.


## 2b. Costi a dimensione reale dell'adapter (T14)

| Scheme | Round payload (MB) | x A | Setup (MB) | Crypto s (p50) | Peak RSS (MB) | LAN s (p50) | WAN s (p50, n) | WAN x A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | 201.83 | 0.80 | 0.00 | 0.000 | 622 | 0.192 | 46.67 (26) | 0.80 |
| N-FP | 252.29 | 1.00 | 0.00 | 0.000 | 847 | 0.231 | 59.28 (21) | 1.01 |
| A | 252.29 | 1.00 | 0.02 | 0.849 | 1441 | 0.217 | 58.68 (21) | 1.00 |
| C | 504.58 | 2.00 | 0.02 | 3.971 | 2039 | 0.368 | 87.99 (14) | 1.50 |
| B | 6271.73 | 24.86 | 3.17 | 115.655 | 7842 | 5.953 | 1339.96 (3) | 22.83 |
| B-BGV | 5262.52 | 20.86 | 5.27 | 87.754 | 6575 | 4.936 | 1126.31 (3) | 19.19 |

### 1. Cost per round, three sites (group main)

| scheme | elements | rounds | setup bytes | round MB (p50) | crypto s p50/p95/p99 | site-max s p50 | reducer s p50 | peak RSS MB (whole process) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | 8409600 | 30 | 0 | 201.83 | 0.000/0.000/0.000 | 0.051 | 0.122 | 622.4 |
| N-FP | 8409600 | 30 | 0 | 252.29 | 0.000/0.000/0.000 | 0.452 | 0.829 | 846.6 |
| A | 8409600 | 30 | 23,199 | 252.29 | 0.849/0.861/0.865 | 1.295 | 0.647 | 1441.3 |
| B | 8409600 | 5 | 3,171,328 | 6271.73 | 115.655/116.649/116.758 | 23.243 | 48.107 | 7841.7 |
| B-BGV | 8409600 | 5 | 5,269,728 | 5262.52 | 87.754/88.701/88.792 | 21.000 | 27.389 | 6575.4 |
| C | 8409600 | 30 | 23,199 | 504.58 | 3.971/4.031/4.036 | 2.361 | 0.651 | 2039.1 |


## 2c. Ripetizioni e sessioni indipendenti (T15)

### 3. Network time per round (replay)

| profile | group | scheme | elements | samples | round net s p50/p95/p99 | setup net s p50 | wire/payload | perturbed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wan | heavy | N | 3407872 | 30 | 20.064/23.689/24.590 | 0.000 | n/a |  |
| wan | heavy | A | 3407872 | 30 | 24.290/26.417/28.820 | 0.151 | n/a |  |
| wan | heavy | B | 3407872 | 3 | 551.728/558.695/559.315 | 2.014 | n/a |  |
| wan | heavy | B-BGV | 3407872 | 3 | 467.078/468.704/468.848 | 3.318 | n/a |  |
| wan | heavy | C | 3407872 | 30 | 38.065/42.912/44.461 | 0.151 | n/a |  |
| wan | main | A | 1000000 | 30 | 7.449/9.908/10.230 | 0.151 | 1.026 |  |
| wan | main | B | 1000000 | 8 | 173.323/179.813/181.075 | 0.840 | 1.030 |  |
| wan | main | B-BGV | 1000000 | 10 | 141.432/144.293/144.421 | 2.736 | 1.030 |  |

wire/payload = interface transmit bytes per replay iteration (all actors, includes TCP/IP overhead and control traffic) over the payload bytes of one round.


## 3. Utilità e attacchi (training)

Protocollo: Qwen3.5-2B-Base, LoRA rango 8 (8.409.600 parametri), MedMCQA in tre siti per materia, 4 round, lr 2e-4, batch 32, limite per elemento 0,02 per tutti gli schemi, clip DP C = 4.

### 3.1 Pilota (seed 0)

Nota: nella cartella del pilota ci sono due prove N con eps 8 (batch 32 e 256), quindi la riga N/eps8 mescola due configurazioni; le tabelle finali non avranno questo effetto.

# Training results (17 runs, seeds [0])

Intervals: percentile bootstrap over seeds, 95 %; none with a single seed.

## Per configuration

| Scheme | DP | n | Accuracy | Perplexity | MIA AUC | TPR 1 % | Canary 16 (bit) | Extraction 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | none | 1 | 0.5104 | 7.715 | 0.6304 | 0.0155 | 7.58 | 0.20 |
| N | eps8 | 2 | 0.4946 [0.4944, 0.4949] | 9.392 [9.300, 9.485] | 0.5020 [0.5011, 0.5029] | 0.0088 [0.0087, 0.0088] | 1.70 [1.70, 1.71] | 0.00 [0.00, 0.00] |
| N | eps2 | 1 | 0.4872 | 9.453 | 0.5015 | 0.0080 | 1.60 | 0.00 |
| A | none | 1 | 0.5090 | 7.710 | 0.6303 | 0.0157 | 7.15 | 0.07 |
| A | eps8 | 1 | 0.4944 | 9.300 | 0.5030 | 0.0087 | 1.60 | 0.00 |
| A | eps2 | 1 | 0.4879 | 9.452 | 0.5015 | 0.0067 | 1.59 | 0.00 |
| C | none | 1 | 0.5104 | 7.710 | 0.6297 | 0.0158 | 7.25 | 0.00 |
| C | eps8 | 1 | 0.4908 | 9.300 | 0.5029 | 0.0090 | 1.59 | 0.00 |
| C | eps2 | 1 | 0.4858 | 9.452 | 0.5016 | 0.0080 | 1.61 | 0.00 |
| B | none | 1 | 0.5066 | 7.707 | 0.6301 | 0.0157 | 7.76 | 0.00 |
| B | eps8 | 1 | 0.4932 | 9.300 | 0.5030 | 0.0090 | 1.61 | 0.00 |
| B | eps2 | 1 | 0.4882 | 9.453 | 0.5015 | 0.0087 | 1.66 | 0.00 |
| B-BGV | none | 1 | 0.5116 | 7.714 | 0.6302 | 0.0155 | 7.73 | 0.07 |
| B-BGV | eps8 | 1 | 0.4908 | 9.300 | 0.5030 | 0.0093 | 1.59 | 0.00 |
| B-BGV | eps2 | 1 | 0.4858 | 9.454 | 0.5015 | 0.0072 | 1.57 | 0.00 |
| Z | none | 1 | 0.4970 | 9.100 | 0.8532 | 0.0328 | 8.01 | 1.00 |

## H3: utility without DP against N (margins: 1 accuracy point, 2 % relative perplexity)

| Scheme | n | Accuracy difference | Relative perplexity difference | Verdict |
| --- | --- | --- | --- | --- |
| A | 1 | -0.0014 | -0.0007 | n/a (one seed) |
| C | 1 | 0.0000 | -0.0007 | n/a (one seed) |
| B | 1 | -0.0038 | -0.0010 | n/a (one seed) |
| B-BGV | 1 | 0.0012 | -0.0002 | n/a (one seed) |

## H4: attack success against epsilon

| Scheme | Metric | none | eps8 | eps2 | Monotone | none vs eps2 separated |
| --- | --- | --- | --- | --- | --- | --- |
| N | membership AUC (loss) | 0.630 | 0.503 | 0.501 | True | None |
| N | TPR at FPR 1 % | 0.015 | 0.009 | 0.008 | True | None |
| N | canary exposure, level 16 | 7.583 | 1.696 | 1.601 | True | None |
| N | extraction rate, level 16 | 0.200 | 0.000 | 0.000 | True | None |
| A | membership AUC (loss) | 0.630 | 0.503 | 0.501 | True | None |
| A | TPR at FPR 1 % | 0.016 | 0.009 | 0.007 | True | None |
| A | canary exposure, level 16 | 7.149 | 1.598 | 1.586 | True | None |
| A | extraction rate, level 16 | 0.067 | 0.000 | 0.000 | True | None |
| C | membership AUC (loss) | 0.630 | 0.503 | 0.502 | True | None |
| C | TPR at FPR 1 % | 0.016 | 0.009 | 0.008 | True | None |
| C | canary exposure, level 16 | 7.245 | 1.591 | 1.614 | False | None |
| C | extraction rate, level 16 | 0.000 | 0.000 | 0.000 | True | None |
| B | membership AUC (loss) | 0.630 | 0.503 | 0.501 | True | None |
| B | TPR at FPR 1 % | 0.016 | 0.009 | 0.009 | True | None |
| B | canary exposure, level 16 | 7.763 | 1.610 | 1.657 | False | None |
| B | extraction rate, level 16 | 0.000 | 0.000 | 0.000 | True | None |
| B-BGV | membership AUC (loss) | 0.630 | 0.503 | 0.501 | True | None |
| B-BGV | TPR at FPR 1 % | 0.015 | 0.009 | 0.007 | True | None |
| B-BGV | canary exposure, level 16 | 7.730 | 1.588 | 1.567 | True | None |
| B-BGV | extraction rate, level 16 | 0.067 | 0.000 | 0.000 | True | None |

## H5: attack success of each scheme against N at the same DP level

| DP | Scheme | Metric | n | Difference | Interval contains 0 |
| --- | --- | --- | --- | --- | --- |
| none | A | membership AUC (loss) | 1 | -0.0001 | None |
| none | A | TPR at FPR 1 % | 1 | 0.0002 | None |
| none | A | canary exposure, level 16 | 1 | -0.4334 | None |
| none | A | extraction rate, level 16 | 1 | -0.1333 | None |
| none | C | membership AUC (loss) | 1 | -0.0007 | None |
| none | C | TPR at FPR 1 % | 1 | 0.0003 | None |
| none | C | canary exposure, level 16 | 1 | -0.3371 | None |
| none | C | extraction rate, level 16 | 1 | -0.2000 | None |
| none | B | membership AUC (loss) | 1 | -0.0003 | None |
| none | B | TPR at FPR 1 % | 1 | 0.0002 | None |
| none | B | canary exposure, level 16 | 1 | 0.1802 | None |
| none | B | extraction rate, level 16 | 1 | -0.2000 | None |
| none | B-BGV | membership AUC (loss) | 1 | -0.0001 | None |
| none | B-BGV | TPR at FPR 1 % | 1 | 0.0000 | None |
| none | B-BGV | canary exposure, level 16 | 1 | 0.1474 | None |
| none | B-BGV | extraction rate, level 16 | 1 | -0.1333 | None |
| eps8 | A | membership AUC (loss) | 1 | 0.0002 | None |
| eps8 | A | TPR at FPR 1 % | 1 | 0.0000 | None |
| eps8 | A | canary exposure, level 16 | 1 | -0.0978 | None |
| eps8 | A | extraction rate, level 16 | 1 | 0.0000 | None |
| eps8 | C | membership AUC (loss) | 1 | 0.0000 | None |
| eps8 | C | TPR at FPR 1 % | 1 | 0.0003 | None |
| eps8 | C | canary exposure, level 16 | 1 | -0.1045 | None |
| eps8 | C | extraction rate, level 16 | 1 | 0.0000 | None |
| eps8 | B | membership AUC (loss) | 1 | 0.0001 | None |
| eps8 | B | TPR at FPR 1 % | 1 | 0.0003 | None |
| eps8 | B | canary exposure, level 16 | 1 | -0.0860 | None |
| eps8 | B | extraction rate, level 16 | 1 | 0.0000 | None |
| eps8 | B-BGV | membership AUC (loss) | 1 | 0.0001 | None |
| eps8 | B-BGV | TPR at FPR 1 % | 1 | 0.0007 | None |
| eps8 | B-BGV | canary exposure, level 16 | 1 | -0.1080 | None |
| eps8 | B-BGV | extraction rate, level 16 | 1 | 0.0000 | None |
| eps2 | A | membership AUC (loss) | 1 | -0.0000 | None |
| eps2 | A | TPR at FPR 1 % | 1 | -0.0013 | None |
| eps2 | A | canary exposure, level 16 | 1 | -0.0154 | None |
| eps2 | A | extraction rate, level 16 | 1 | 0.0000 | None |
| eps2 | C | membership AUC (loss) | 1 | 0.0001 | None |
| eps2 | C | TPR at FPR 1 % | 1 | 0.0000 | None |
| eps2 | C | canary exposure, level 16 | 1 | 0.0131 | None |
| eps2 | C | extraction rate, level 16 | 1 | 0.0000 | None |
| eps2 | B | membership AUC (loss) | 1 | 0.0000 | None |
| eps2 | B | TPR at FPR 1 % | 1 | 0.0007 | None |
| eps2 | B | canary exposure, level 16 | 1 | 0.0554 | None |
| eps2 | B | extraction rate, level 16 | 1 | 0.0000 | None |
| eps2 | B-BGV | membership AUC (loss) | 1 | 0.0000 | None |
| eps2 | B-BGV | TPR at FPR 1 % | 1 | -0.0008 | None |
| eps2 | B-BGV | canary exposure, level 16 | 1 | -0.0342 | None |
| eps2 | B-BGV | extraction rate, level 16 | 1 | 0.0000 | None |


### 3.3 Controlli di validità

- Modello base (nessun addestramento): accuratezza 0.4910 su 4183, perplexity 11.02, AUC membership 0.495, esposizione canary 1.34 bit (livello 0) e 1.54 bit (livello 16), estrazione 0: i controlli nulli passano.
- Controllo positivo: addestrando senza DP il livello 16 sale a circa 7,5 bit (soglia nulla 95° percentile ~4–5 bit) e l'AUC a 0,63.
- Con DP eps 8 AUC e canary tornano al livello nullo (0,50 e ~1,7 bit).

### 3.5 Secondo modello: Qwen3.5-0.8B-Base (pilota, seed 0, protocollo del 2B)

Base non addestrato: accuratezza 0,3988, perplexity 14,94, AUC 0,4999 (controllo nullo superato).

# Training results (2 runs, seeds [0])

Intervals: percentile bootstrap over seeds, 95 %; none with a single seed.

## Per configuration

| Scheme | DP | n | Accuracy | Perplexity | MIA AUC | TPR 1 % | Canary 16 (bit) | Extraction 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N | none | 1 | 0.4377 | 9.693 | 0.6103 | 0.0127 | 6.22 | 0.00 |
| N | eps8 | 1 | 0.4176 | 12.080 | 0.5016 | 0.0107 | 1.63 | 0.00 |

## H3: utility without DP against N (margins: 1 accuracy point, 2 % relative perplexity)

| Scheme | n | Accuracy difference | Relative perplexity difference | Verdict |
| --- | --- | --- | --- | --- |

## H4: attack success against epsilon

| Scheme | Metric | none | eps8 | eps2 | Monotone | none vs eps2 separated |
| --- | --- | --- | --- | --- | --- | --- |
| N | membership AUC (loss) | 0.610 | 0.502 | - | True | None |
| N | TPR at FPR 1 % | 0.013 | 0.011 | - | True | None |
| N | canary exposure, level 16 | 6.222 | 1.627 | - | True | None |
| N | extraction rate, level 16 | 0.000 | 0.000 | - | True | None |

## H5: attack success of each scheme against N at the same DP level

| DP | Scheme | Metric | n | Difference | Interval contains 0 |
| --- | --- | --- | --- | --- | --- |


### 3.4 Inversione dei gradienti (controllo positivo su N, 4 round, un esempio)

| Prova | Passi | lr | Temperatura | Distanza coseno | Recall token | Soglia nulla (95°) | Supera il caso | s/passo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| inversion-conv-1 | 300 | 0.1 | 1.0 | 0.713 | 0.125 | 0.188 | False | 2.61 |
| inversion-conv-2 | 400 | 0.3 | 0.5 | 0.926 | 0.062 | 0.188 | False | 2.61 |

Controllo di coerenza sul modello reale (cosine tra il gradiente osservato e quello ricostruito con i percorsi morbidi): esempio 1: token veri 1.0000, un token sbagliato 0.07, token casuali -0.04; esempio 2: token veri 1.0000, un token sbagliato 0.40, token casuali -0.01; esempio 3: token veri 1.0000, un token sbagliato 0.24, token casuali 0.01. L'obiettivo è corretto e molto selettivo; la difficoltà è nell'ottimizzazione.

Ricerca a coordinate (inversion-coord-1, top-32, 2 passate, 993 valutazioni): coseno finale 0.103; recall 0.375 contro 0.375 della continuazione del modello senza gradiente (soglia nulla 0.188): il valore sopra il caso viene dalla memorizzazione, non dall'inversione.

L'attacco non supera il caso con questi budget (<= 17 minuti per esempio, un solo esempio): non è un controllo positivo riuscito e non va letto come prova di privacy.

## 4. Limiti da dichiarare

- Un solo modello (2B), un solo dataset, tre siti, regime LoRA di rango 8; senza DP il guadagno di utilità è modesto (accuratezza 0,49 → 0,51, perplexity 11,0 → 7,7).
- Costi misurati su host condiviso (carico medio ~10): non sono quelli di un nodo dedicato. I tempi di rete sono replay, non cifratura distribuita.
- HE alla dimensione massima non misurato; B e B-BGV a 3,4M hanno 14 round e a WAN 6–7 campioni: solo sintesi descrittive.
- I tre schemi senza DP calcolano la stessa somma: l'equivalenza di utilità è attesa; il contributo è quantificarla e misurare i costi.
- DP: ε da RDP con ordini interi (conservativo); rumore da seme pubblico, quindi nessuna garanzia formale di rilascio; esempio-livello.
- Gli attacchi sono specifici (loss e riferimento, canary con riferimenti finiti), non LiRA; un risultato nullo non prova la privacy.
- Sicurezza di B/B-BGV (flooding), combiner ibrido, recupero del dropout: da far rivedere; il recupero resta esplorativo.
- Un seed nel pilota: nessun intervallo; con cinque seed gli intervalli bootstrap sono grossolani.

