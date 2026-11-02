# Registro del pilota (30 ott – 1 nov 2026): risultati positivi e negativi

Documento interno per la discussione con gli altri autori. Non è il manoscritto. Raccoglie ciò che è stato misurato,
ciò che non ha funzionato, gli errori trovati (anche nostri) e le decisioni ancora aperte. Le tabelle numeriche
complete sono in `docs/results_dossier.md` (rigenerabile con `scripts/build_dossier.py`); il registro cronologico
delle decisioni è `docs/decisions.md`; i checksum dei file di prova sono in `docs/pilot_manifest.json` (generato a
fine pilota con `scripts/build_pilot_manifest.py`).

## 0. Ambito e stato

- Pilota a **un seed (0)**, Qwen3.5-2B-Base (LoRA rango 8, 8.409.600 parametri), MedMCQA in tre siti per materia,
  schemi N, A, B (CKKS), B-BGV, C e riferimento centralizzato Z. Nessun intervallo di confidenza è possibile con un seed.
- Protocollo congelato dal pilota: 4 round, lr 2e-4, microbatch 4, batch logico 32, limite per elemento 0,02 per tutti gli
  schemi, clip DP C = 4, δ = 1e-5, ε ∈ {8, 2}.
- **La campagna finale (cinque seed, 80 run) non è stata lanciata.** È preparata (`configs/final_campaign.yaml`,
  `scripts/submit_final.py`) e richiede un via esplicito.
- Stato (2 nov 2026, 01:30): **pilota concluso.** Training: tutti i run del 2B e il pilota del secondo modello (0,8B);
  costi: T14 (dimensione reale) e T15 (ripetizioni e sessioni indipendenti) completate. GPU rilasciata. Restano le decisioni
  della sezione 7.

## 1. Risultati positivi (seed 0)

### 1.1 Utilità e attacchi

| Configurazione | Accuratezza | Perplexity holdout | AUC membership | Canary liv. 16 (bit) |
| --- | --- | --- | --- | --- |
| Modello base | 0,4910 | 11,02 | 0,495 | 1,5 |
| N senza DP (con limite 0,02) | 0,5104 | 7,715 | 0,6304 | 7,58 |
| A senza DP | 0,5090 | 7,710 | 0,6303 | 7,15 |
| C senza DP | 0,5104 | 7,710 | 0,6297 | 7,25 |
| B senza DP | 0,5066 | 7,707 | 0,6301 | 7,76 |
| B-BGV senza DP | 0,5116 | 7,714 | 0,6302 | 7,73 |
| N, A, B, C, B-BGV con ε = 8 | 0,4908 – 0,4944 | 9,300 | 0,5029 – 0,5030 | 1,59 – 1,70 |
| N, A, B, C, B-BGV con ε = 2 | 0,4858 – 0,4882 | 9,452 – 9,454 | 0,5015 – 0,5016 | 1,57 – 1,66 |
| Z (centralizzato, 4 round, senza DP) | 0,4970 | 9,100 | 0,8532 | 8,01 |

- **H3 (equivalenza di utilità senza DP):** i quattro schemi protetti differiscono da N di meno di 0,4 punti di
  accuratezza e dello 0,1 % di perplexity (entro il margine del protocollo, 1 punto e 2 %), a un seed.
- **H5 (stessi attacchi a parità di DP):** AUC e canary coincidono tra gli schemi a ogni livello di DP.
- **H4 (DP riduce gli attacchi):** a ε = 8 AUC e canary tornano al livello nullo del modello base; la differenza con
  la DP spenta è netta (0,630 → 0,503; 7,5 → 1,6 bit).
- **Controlli di validità:** il modello base dà AUC 0,495 ed esposizione dei canary nulla (nessuna estrazione); con
  l'addestramento senza DP i canary di livello 16 salgono a ~7,5 bit (soglia nulla 95° percentile ~4–5 bit).
- **Contabile DP:** coincide con quello di Opacus sulla stessa griglia di ordini (differenza relativa < 1e-12, ε identico
  7,9825); il nostro ε è prudente (Opacus RDP con ordini frazionari 7,66; PRV 6,43 per lo stesso σ).
- **Costi (campagna 3, n = 3, d = 3.407.872):** C = 2,0 × byte di A; B e B-BGV = 24,8 e 20,8 × A; N-float32 = 0,8 × A.
  WAN (50 ms, 100 Mbit/s, 0,1 % perdita, replay): N 19,9 s, A 24,5 s, C 39,8 s, B 546 s, B-BGV 469 s per round.
  L'handshake ibrido di A costa ~154 ms a WAN (somma delle mediane), trascurabile oltre ~15 s di round.
- **Dropout (esplorativo):** il recupero costa byte extra piccoli in A (82 byte) e ~656 KB in B-BGV-threshold; vedi limiti.

### 1.2 Costi a dimensione reale dell'adapter (T14, 8.409.600 elementi, n = 3; `runs/campaign-4`)

| Schema | Payload per round (MB) | × A | Crypto s p50 | RSS di picco (MB) | LAN s p50 | WAN s p50 (campioni) |
| --- | --- | --- | --- | --- | --- | --- |
| N (float32) | 201,8 | 0,80 | 0 | 622 | 0,192 | 46,7 (26) |
| N-FP (stessa codifica di A, nessuna protezione) | 252,3 | 1,00 | 0 | 847 | 0,231 | 59,3 (21) |
| A | 252,3 | 1,00 | 0,85 | 1441 | 0,217 | 58,7 (21) |
| C | 504,6 | 2,00 | 3,97 | 2039 | 0,368 | 88,0 (14) |
| B (CKKS) | 6271,7 | 24,9 | 115,7 | 7842 | 5,95 | 1340 (3) |
| B-BGV | 5262,5 | 20,9 | 87,8 | 6575 | 4,94 | 1126 (3) |

- **Baseline a codifica uguale (voce M03):** A manda gli stessi byte di N-FP e a WAN non è più lento (58,7 contro 59,3 s): il
  sovraccosto di A rispetto a N float32 (+25 % in byte e tempo WAN) è tutto della codifica a 5 byte, non della protezione; il costo
  della protezione di A è nel calcolo (0,85 s di crittografia per round).
- **Riproducibilità (T15, voci M02 e R01):** tre celle WAN a 10⁶ sovrapposte a prove nostre in campagna 3 sono confermate entro il
  5 % da una ripetizione pulita; tra sessioni indipendenti le mediane WAN variano ≤ 4,5 % (≈ 1 % per B e B-BGV a 3,4 M, tre sessioni).
  Dettaglio in `docs/decisions.md` (T15) e `runs/campaign-5`, `runs/campaign-6`.

### 1.3 Secondo modello (T16, Qwen3.5-0.8B-Base, pilota a un seed, protocollo del 2B senza ri-taratura)

| Configurazione | Accuratezza | Perplexity | AUC loss (rif.) | Canary liv. 16 (bit) | Estrazione liv. 16 |
| --- | --- | --- | --- | --- | --- |
| Base non addestrato | 0,3988 | 14,94 | 0,4999 | 1,62 | 0 |
| N senza DP | 0,4377 | 9,693 | 0,6103 (0,6406) | 6,22 | 0 |
| N con ε = 8 (C = 7,5) | 0,4176 | 12,08 | 0,5016 (0,5027) | 1,63 | 0 |

Il protocollo si trasferisce: guadagno di utilità senza DP (+3,9 punti), memorizzazione misurabile (AUC 0,61; canary 6,2 bit contro
soglia nulla 3,6) azzerata da ε = 8, che conserva +1,9 punti di accuratezza. Solo N, un seed.

### 1.4 Cosa il pilota permette di dire (e cosa no)
Permette di dire che a questo seed gli schemi protetti riproducono N in utilità e attacchi, che la DP con ε = 8 azzera la
memorizzazione misurata e costa utilità, e quantifica i costi. Non permette intervalli, non distingue ε = 8 da ε = 2 negli
attacchi (entrambi al livello nullo), non dice nulla su altri modelli o dataset.

## 2. Risultati negativi o nulli (da non perdere)

1. **Inversione dei gradienti: nessun controllo positivo riuscito.** Due configurazioni a gradiente su un esempio (300 passi lr 0,1:
   coseno 0,713, recall 0,125; 400 passi lr 0,3: coseno 0,926, recall 0,0625; soglia nulla 0,1875) e una ricerca a coordinate
   (coseno 0,103 invariato; il recall 0,375 sopra il caso è identico alla continuazione del modello senza gradiente, quindi
   viene dalla memorizzazione, non dall'inversione). Il controllo di coerenza sul modello reale mostra che l'obiettivo è corretto
   (coseno 1,0000 con i token veri, 0,07–0,40 con un token sbagliato, ~0 con token casuali): la difficoltà è nell'ottimizzazione.
   **Non si può dedurre che il modello sia protetto dall'inversione.**
2. **CKKS senza flooding:** il flooding di OpenFHE non ha prodotto alcun effetto misurabile (errore a precisione di macchina);
   per B non si rivendica protezione IND-CPA-D. B-BGV ha il flooding visibile (ciphertext più grandi) ma non è verificato.
3. **Modelli più grandi:** Qwen3.5-9B non è entrato in memoria sulla GPU condivisa (OOM, non un verdetto sul modello); il 4B era
   troppo lento per il budget. Si è usato il 2B.
4. **Pilota a 10 round (N e Z) fuori protocollo:** sovradattamento forte (perplexity holdout 10,2 per N e 28,3 per Z, AUC 0,87 e
   0,98): il guadagno di utilità sparisce. Sostituito da 4 round. Anche Z a 4 round si adatta più dei federati (fa il triplo dei
   passi per round), quindi **non è un limite superiore di utilità**.
5. **Batch 256 con DP:** non migliora (perplexity 9,485 contro 9,300, accuratezza uguale); il protocollo resta batch 32.
6. **Utilità limitata:** l'accuratezza passa da 0,491 (base) a 0,51 senza DP; con DP (ε = 8) torna a 0,49: il DP cancella il
   guadagno di accuratezza; resta un guadagno di perplexity (11,0 → 9,3).
7. **Membership inference debole a 4 round:** AUC 0,63 senza DP; il TPR a FPR 1 % è 0,016 e a 0,1 % è 0,0025: le conclusioni su
   questi attacchi sono quasi interamente dei canary.
8. **Potere discriminante basso tra schemi con DP:** gli schemi sono indistinguibili in tutto (supporta H5) quindi il confronto
   tra schemi si gioca sui costi; ε = 8 e ε = 2 non si separano negli attacchi.
9. **Costi: limiti di misura.** Tre celle WAN a d = 10⁶ si sovrappongono a prove nostre (A, B, B-BGV, flaggate); B e B-BGV a WAN
   hanno 6–7 campioni a 3,4 M e nessuna misura a 20,9 M; tutto su host condiviso (carico medio ~10); una sessione per cella.
10. **Il modello di stima della rete sottostimava di 2,8 volte** (corretto con un modello per flusso con limite di Mathis).
11. **Dropout:** modello crash-stop; in A rivelare il seed di coppia espone gli upload precedenti dell'epoca; le share di
    B-BGV-threshold sono segnaposto di dimensione fedele. Resta esplorativo.
12. **Dimensione reale (T14/T15): costo HE molto alto e poco campionato.** B e B-BGV a 8,4 M elementi mandano ~6,3 e 5,3 GB per round,
    usano 7,8 e 6,6 GB di memoria in un solo processo e a WAN impiegano ~22 e ~19 minuti per round; ogni cella ha 3 campioni per
    sessione (6–9 in totale a 3,4 M): solo sintesi descrittive, nessuna coda affidabile. Nessuna cella saltata.
13. **Secondo modello: la prova è parziale.** Solo N (nessuno schema protetto), un seed, accuratezza assoluta bassa (0,44 senza DP),
    estrazione esatta nulla anche senza DP (nel 2B fino al 73 %): il segnale di memorizzazione è più debole, quindi gli attacchi
    discriminano meno.
14. **Host sempre condiviso:** carico medio 8–14 (CPU occupata 13–36 %) durante T14 e T15; un vicino usa la GPU (538 MiB). Le misure
    sono ripetibili (≈ 1–4 %) ma non sono quelle di un nodo dedicato.

## 3. Errori nostri nella comunicazione dei risultati (corretti)

- Avevo previsto che il DP con batch 32 e 8,4 M parametri non avrebbe appreso (segnale/rumore ~1/50 per passo): era sbagliato,
  impara (il rumore isotropo si media sui passi).
- Avevo atteso che i duplicati tra membri e non membri schiacciassero il TPR a basso FPR: l'effetto misurato è trascurabile
  (AUC 0,6305 contro 0,6301; TPR 1 % invariato).
- I checkpoint sono numerati da 0: "round 2/3/4" nelle prime tabelle erano dopo 3/4/5 round. Il protocollo a 4 round è
  round-0003.pt.
- L'accuratezza della valutazione rapida per round (500 domande) differisce dalla finale (4183): vale la finale.
- Le ore di alcune annotazioni in `decisions.md` erano imprecise e sono state allineate ai timestamp dei job.
- Primo avvio di T14 fallito: avevo creato io la cartella di output, che lo script rifiuta; spostata in `runs/old` e rilanciata.
- Un mio primo controllo di coerenza dell'inversione aveva un errore di forma dei tensori (`inversion-selfcheck-1`, in `runs/old`).
- Le sostituzioni dei kernel per la prova di inversione erano permanenti nel processo GPU: corrette (ripristino) e processo riavviato.

## 4. Difetti trovati dalla revisione del codice e loro effetto sui risultati già ottenuti

| Difetto | Effetto sui risultati del pilota |
| --- | --- |
| Ripresa da checkpoint senza controllo della configurazione | Nessuno (nessun run è stato ripreso); ora rifiutata |
| Picco di memoria GPU cumulato tra job | I valori di memoria dei pilota precedenti sono stale e non vanno usati |
| Perplexity holdout con 25 duplicati dei membri | +0,006/0,007 dopo correzione; ordine tra schemi invariato |
| Soglia FPR con errore in virgola mobile | Nessuno con le nostre numerosità |
| Limite di gruppo DP senza termine δ | Non usato in alcun risultato |
| Aggregatori A e C consumavano il round prima di validare | Nessuno (i vettori erano in range); corretto |
| Sostituzione permanente dei kernel dopo la prova di inversione | Evitata: worker riavviato; nessun training l'ha usata |

## 5. Stato dei compiti

- **T14 (costi a dimensione reale 8.409.600, con N-FP):** completata (`runs/campaign-4`).
- **T15 (ripetizione celle WAN perturbate e sessioni indipendenti):** completata (`runs/campaign-5`, `runs/campaign-6`).
- **T16 (pilota del secondo modello Qwen3.5-0.8B-Base):** completata (`runs/pilot-q08b`); solo pilota, nessuna campagna.
- **T17 (consolidamento):** dossier, registro, istantanea del codice e manifesto dei checksum.
- Non fatto: revisione crittografica di B, B-BGV, combiner e recupero (pacchetto per l'esperto in `docs/pqfl-crypto-review.zip`);
  attacchi più forti (LiRA, inversione ottimizzata); intervalli con cinque seed.

## 6. Provenienza

- Codice: nessun commit (richiesto dall'utente); stato di lavoro con modifiche non versionate; snapshot e checksum dei file di
  prova in `docs/pilot_manifest.json`. Campagna 3: `runs/campaign-3/code-snapshot.tar.gz` (sha256 in `code-snapshot.sha256`),
  immagine `sha256:11d93f7c…`.
- Dati: partizione per materia con seed 2026 (`configs/data/medmcqa-manifest.json`), canary in `configs/data/canaries.json`.
- Ambiente: Python 3.10.13, liboqs 0.16.0 (binding 0.16.0.1), cryptography 50.0.2, OpenFHE wheel 1.6.1.0.22.4, torch 2.14.1,
  transformers 5.20.2; GPU L40 condivisa; host condiviso.

## 7. Decisioni per gli autori

1. Lanciare la campagna finale a cinque seed (80 run, ~82 h di GPU) con il protocollo congelato? Il seed del pilota è escluso.
2. Includere B-BGV nella tabella principale (16 configurazioni invece di 13)?
3. Secondo modello: il pilota (N senza DP e con ε = 8) conferma la tendenza del 2B; eseguire la campagna ridotta (N, A, B senza DP;
   N, A con ε = 8; tre seed, `configs/second_model_campaign.yaml`, C = 7,5 provvisorio, ~15 run) oppure tenerlo come pilota dichiarato?
4. Inversione dei gradienti: dichiararla come limite (con l'esito nullo) oppure toglierla dal paper senza affermazioni di protezione;
   eventualmente investire in un attacco più forte.
5. Revisione crittografica: esito dell'esperto prima di riportare risultati di B e B-BGV; se cambiano i parametri HE, T14 va
   ripetuto per quei due schemi.
6. Seme del rumore DP: per affermazioni formali serve rumore da seme segreto (opzione già nel codice).
