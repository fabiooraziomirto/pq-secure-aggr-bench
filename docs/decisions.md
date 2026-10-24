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
| 2026-10-24 | JSONL schema_version=1, run_seed dalla configurazione | Riproducibilità pubblica separata da tutti i segreti crittografici; sostituisce seed=null | Superata da schema v2 |
| 2026-10-24 | A come simulatore centralizzato senza dropout | Verifica funzionale delle interfacce, non deployment sicuro | Adottata |
| 2026-10-24 | Libreria KEM scelta: liboqs-python/liboqs candidata; OpenSSL TODO | Rilevamento delle capacità, preferenza ML-KEM-768; Kyber768 solo con opt-in esplicito, mai equiparato allo standard | Da decidere dopo check_env sul cluster |
| 2026-10-24 | Combiner ibrido: X25519 + segreto PQ, poi HKDF-SHA256 con trascrizione | Segue la logica dei progetti di key exchange ibrido; costruzione sperimentale non rivista formalmente. TODO: revisione da parte di un esperto | Da rivedere |
| 2026-10-24 | Trasporto del canale: TLS 1.3 ibrido via OpenSSL oppure canale applicativo | HybridKEM non fornisce autenticazione, framing di rete o protezione replay; nessuna equivalenza con un profilo TLS standard | Aperto |
| 2026-10-24 | Setup A: generare n−1 coppie di chiavi | Prima si generavano n coppie, ma il sito con ID minimo non è mai destinatario; timer solo per le chiavi necessarie e invio della chiave pubblica a ogni coppia interessata | Adottata |
| 2026-10-24 | Payload serializzati e diffusione inclusa, reducer come attore | Byte da len dei messaggi; nessun framing di trasporto o tempo comm fittizio | Adottata |
| 2026-10-24 | JSONL schema_version=2, site, epoch e delta dei byte | Tempi attribuibili per sito, righe sommabili per operazione; sostituisce gli snapshot cumulativi | Adottata |
| 2026-10-24 | Politica di rekey: frequenza TODO, chiamata esplicita | Nuove chiavi e seed per tutte le coppie, epoch crescente, round univoci nell’intera sessione | Aperta |
| 2026-10-24 | Conferma bidirezionale HKDF/HMAC con ruoli e trascrizione | Rilevare segreti diversi da rifiuto implicito senza confronto diretto dei seed; contare i tag | Adottata |
| 2026-10-24 | clip_bound opzionale e controllo esatto dell’overflow nel simulatore | Il costruttore non conosce n: valida il limite minimo per 2 siti; setup valida n e margine di arrotondamento. In produzione richiede un limite verificato localmente | Adottata |
| 2026-10-24 | Vettore dorato mask/v2 generato dalla funzione corrente | Regressione di compatibilità, non controllo indipendente né vettore NIST | Adottata |
| 2026-10-24 | Interoperabilità liboqs/OpenSSL condizionata alle capacità reali | Opzioni encap/decap verificate nell’help locale; resta da eseguire con binding e libreria sul cluster | Da verificare |

## Decisioni D/E/F

| Data | Decisione | Motivo | Stato |
| --- | --- | --- | --- |
| 2026-10-24 | D: binding liboqs-python 0.16.0.1 e liboqs 0.16.0 candidata | Binding installato, build nativa ferma per CMake assente; nessuna nuova dipendenza oltre al binding | Bloccata sul runtime attuale |
| 2026-10-24 | D: requires_oqs e make test-real senza skip ammessi | Evitare che test reali non eseguiti producano un esito di readiness | Adottata |
| 2026-10-24 | D: env opzionale nel setup, schema_version resta 2 | Metadati pubblici identici al collector check_env.py | Adottata |

### D: comandi eseguiti e provenienza

Comandi effettivamente usati (percorsi di questa istanza, non del cluster):

```bash
/workspace/pqfl-venv/bin/python -m pip install liboqs-python
git clone --depth 1 --branch 0.16.0 https://github.com/open-quantum-safe/liboqs.git /tmp/pqfl-liboqs-0.16.0
cmake -S /tmp/pqfl-liboqs-0.16.0 -B /tmp/pqfl-liboqs-build -DCMAKE_INSTALL_PREFIX=/workspace/pqfl-oqs -DBUILD_SHARED_LIBS=ON -DOQS_BUILD_ONLY_LIB=ON -DOQS_ENABLE_KEM_ml_kem_768=ON -DOQS_ENABLE_SIG_ml_dsa_65=ON
```

Binding installato: 0.16.0.1, senza dipendenze aggiuntive. Sorgenti liboqs tag
0.16.0, commit `5a1a854b0dc9f2141bdc771c555ee60c37950183`.
Configurazione terminata con exit 127: `cmake: command not found`. Nessuna
compilazione/installazione nativa eseguita; non si è attivato l’installer
automatico del binding e non si dichiara ML-KEM/ML-DSA funzionante.
CMake non è stato installato per rispettare il vincolo sulle nuove dipendenze.
Dopo aver predisposto il toolchain sul cluster, i passi non ancora eseguiti sono
`cmake --build /tmp/pqfl-liboqs-build --parallel 2` e
`cmake --install /tmp/pqfl-liboqs-build`; poi impostare OQS_INSTALL_PATH e
verificare algoritmi abilitati e `make test-real`.

KAT ufficiali scaricati fuori dal repository in `/workspace/pqfl-kats/ML-KEM-768`:
NIST ACVP-Server commit `975de31eb83d87039ec88934fdc47d8c312b892d`, percorso
`gen-val/json-files/ML-KEM-encapDecap-FIPS203`, via HTTPS raw.githubusercontent.com.
`prompt.json` SHA256 `998e22dfb12efb14ce9fdff911ca634b13612819a1806f25da69adba7e16db91`;
`expectedResults.json` SHA256 `9089ec6ff2424da9f2782b89b2f831a329a3e28d6e5e24b802b78ff36ac61cdf`.
Usare PQFL_KAT_DIR con la directory padre. Vettori non fabbricati né aggiunti
al repository; non ancora verificati contro liboqs perché manca il backend.
I cinque test reali restano saltati per libreria nativa non caricabile:
component_sizes, real_round_trip, hybrid_round_trip, known_answers e
mlkem_liboqs_openssl_interoperability. Il binding Python da solo non basta.
