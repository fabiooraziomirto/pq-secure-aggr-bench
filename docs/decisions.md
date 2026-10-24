# Registro delle decisioni

| Data | Decisione | Motivo | Stato |
| --- | --- | --- | --- |
| 2026-10-24 | Modello base: TODO | Compatibilità LoRA, licenza e risorse da valutare | Aperta |
| 2026-10-24 | Libreria KEM: TODO (liboqs/OpenSSL) | Supporto verificato ML-KEM-768 e interoperabilità | Aperta |
| 2026-10-24 | Libreria HE: TODO | CKKS a soglia e parametri del collega | Aperta |
| 2026-10-24 | DP di record o sito: TODO | Unità protetta, clipping e accounting da concordare | Aperta |
| 2026-10-24 | Emulazione di rete: TODO | Topologia, tc e permessi del cluster da verificare | Aperta |
| 2026-10-24 | Formato container: TODO | Docker/Apptainer secondo il cluster | Aperta |
| 2026-10-24 | scale_bits e modulus_bits: TODO | Range e tolleranza dipendono dagli aggiornamenti | Aperta |
| 2026-10-24 | Autenticazione e combiner ibrido: TODO | Specifica standard e revisione richieste | Aperta |
| 2026-10-24 | Versioni, lock e digest immagini: TODO | Non inventare pin non verificati | Aperta |
| 2026-10-24 | Licenza: TODO | Scelta esplicita ancora necessaria | Aperta |
| 2026-10-24 | NumPy come Tensor del riferimento CPU | Test leggeri senza training o GPU | Provvisoria |
| 2026-10-24 | JSONL schema_version=1, run_seed dalla configurazione | Riproducibilità pubblica separata da tutti i segreti crittografici; sostituisce seed=null | Adottata |
| 2026-10-24 | A come simulatore centralizzato senza dropout | Verifica funzionale delle interfacce, non deployment sicuro | Adottata |
| 2026-10-24 | Libreria KEM scelta: liboqs-python/liboqs candidata; OpenSSL TODO | Rilevamento delle capacità, preferenza ML-KEM-768; Kyber768 solo con opt-in esplicito, mai equiparato allo standard | Da decidere dopo check_env sul cluster |
| 2026-10-24 | Combiner ibrido: X25519 || segreto PQ, poi HKDF-SHA256 con trascrizione | Segue la logica dei progetti di key exchange ibrido; costruzione sperimentale non rivista formalmente. TODO: revisione da parte di un esperto | Da rivedere |
| 2026-10-24 | Trasporto del canale: TLS 1.3 ibrido via OpenSSL oppure canale applicativo | HybridKEM non fornisce autenticazione, framing di rete o protezione replay; nessuna equivalenza con un profilo TLS standard | Aperto |
| 2026-10-24 | Setup A: generare n−1 coppie di chiavi | Prima si generavano n coppie, ma il sito con ID minimo non è mai destinatario; timer solo per le chiavi necessarie e invio della chiave pubblica a ogni coppia interessata | Adottata |
| 2026-10-24 | Payload serializzati e diffusione inclusa, reducer come attore | Byte da len dei messaggi; nessun framing di trasporto o tempo comm fittizio | Adottata |
| 2026-10-24 | JSONL schema_version=2, site, epoch e delta dei byte | Tempi attribuibili per sito, righe sommabili per operazione; sostituisce gli snapshot cumulativi | Adottata |
| 2026-10-24 | Politica di rekey: frequenza TODO, chiamata esplicita | Nuove chiavi e seed per tutte le coppie, epoch crescente, round univoci nell’intera sessione | Aperta |
| 2026-10-24 | Conferma bidirezionale HKDF/HMAC con ruoli e trascrizione | Rilevare segreti diversi da rifiuto implicito senza confronto diretto dei seed; contare i tag | Adottata |
| 2026-10-24 | clip_bound opzionale e controllo esatto dell’overflow nel simulatore | Il costruttore non conosce n: valida il limite minimo per 2 siti; setup valida n e margine di arrotondamento. In produzione richiede un limite verificato localmente | Adottata |
