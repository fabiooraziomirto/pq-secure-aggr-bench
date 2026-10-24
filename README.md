# Aggregazione post-quantum di aggiornamenti LoRA

Base di ricerca per tre siti A, B, C. Nessun dato clinico incluso.
Licenza: TODO. Versione 0.0.0: segnaposto del pacchetto, non release scientifica.

## Setup e test

Python >=3.10 (minimo richiesto dalle annotazioni); versione del runtime di
riferimento e versioni riproducibili delle dipendenze: TODO.

```bash
cd /workspace/pq-secure-aggr-bench
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

I test sono sintetici, CPU e leggeri. Nessun training o benchmark.
Il simulatore usa array NumPy come `Tensor`; la pipeline LoRA dovrà convertire
esplicitamente forma, ordine e dtype degli adapter. Non richiede PyTorch o GPU.
FakeKEM esiste solo in `tests/conftest.py`, non nel pacchetto installato, ed è
INSECURE. È disponibile un adapter reale liboqs con import lazy, da verificare sul cluster;
non è un deployment pronto per produzione.

## Aggiungere uno schema

Implementare `Aggregator` in `src/pqfl/aggregators/`, con la stessa API e la
strumentazione comune. B (CKKS a soglia) e C (MPC a tre parti) sono stub che
sollevano `NotImplementedError`. Specifica, esempi e criteri di consegna:
[interfaccia](docs/interface.md). Consultare anche il
[modello di minaccia](docs/threat_model.md) e le [decisioni](docs/decisions.md).

`configs/experiment.yaml` è un template con TODO, non una configurazione eseguibile.
Non sono presenti runner di esperimenti. Container Docker e Apptainer sono
bozze non compilabili finché i TODO non sono risolti; non sono stati costruiti.
`scripts/check_env.sh` va eseguito sul cluster: controlla OpenSSL/ML-KEM,
liboqs e tc; non certifica sicurezza, interoperabilità o permessi di rete.

Non salvare dati clinici, chiavi, segreti o credenziali. Usare identificatori
neutri e non aggiungere autori, affiliazioni o ringraziamenti per double-blind.

I log JSONL usano `schema_version: 2` e `run_seed` pubblico dalla configurazione
YAML (intero non negativo da scegliere al posto di TODO). Non viene usato per
inizializzare KEM reali; nessun materiale crittografico deve essere registrato.

## Backend reale e verifica sul cluster

Non è stata installata liboqs in questa sessione. Il pacchetto non scarica o
compila automaticamente backend: predisporre sul cluster una versione revisionata
di **liboqs** e del binding ufficiale **liboqs-python** (`import oqs`). Non usare
pacchetti omonimi di origine diversa. Versioni, pin e percorso dei moduli: TODO.
Configurare il loader nativo del cluster (`LD_LIBRARY_PATH` o `OQS_INSTALL_PATH`)
prima di avviare Python; il preflight impedisce l'import se liboqs non è rilevabile.

Ordine di verifica (dalla root, nell'ambiente Python attivo):

1. Installare il pacchetto con `python -m pip install -e '.[test]'` e predisporre
   liboqs/binding tramite gli strumenti autorizzati del cluster.
2. Eseguire `bash scripts/check_env.sh`: stampa OpenSSL, capacità ML-KEM, liboqs,
   binding e meccanismi abilitati, oltre a `tc`. Un esito negativo va diagnosticato;
   `tc` e OpenSSL non sono prerequisiti del solo adapter liboqs.
3. Predisporre vettori ufficiali esterni al repository, impostare `PQFL_KAT_DIR`.
4. Eseguire `python -m pytest -q -rs`. Test reali saltati NON certificano il backend.
5. Solo dopo revisione e autorizzazione agli esperimenti, scegliere parametri e
   valutare l'harness descritto sotto. In questa sessione non va eseguito.

`LibOQSKEM()` preferisce `ML-KEM-768`. Se esiste solo `Kyber768`, segnala
l'assenza dello standard; `LibOQSKEM(allow_legacy=True)` consente verifiche legacy
esplicitamente etichettate e con warning. Kyber768 non è interoperabile per
assunzione con ML-KEM-768. Nessun fallback a FakeKEM.
`HybridKEM()` implementa la stessa API e può essere passato a `SchemeA`.
Composizione sperimentale: richiede revisione da parte di un esperto e non
sostituisce il trasporto autenticato.

### Vettori di risposta nota

Fonti ufficiali: progetto NIST
[ACVP-Server](https://github.com/usnistgov/ACVP-Server/tree/master/gen-val/json-files/ML-KEM-encapDecap-FIPS203),
esempi JSON ML-KEM encapDecap FIPS203, e progetto
[liboqs](https://github.com/open-quantum-safe/liboqs), strumenti KAT e vettori
previsti dalla versione selezionata. Verificare algoritmo, revisione e provenienza;
conservare esternamente URL, commit e checksum. Non ricostruire vettori a memoria.
Hash di riferimento da soli non sono vettori di decapsulazione.

Il test verifica **decapsulazione con risposta nota**, non keygen/encaps
deterministici (il binding comune non espone qui un'interfaccia di casualità
controllata). Struttura supportata:

```text
$PQFL_KAT_DIR/
  ML-KEM-768/
    prompt.json
    expectedResults.json
    # oppure file ufficiali *.rsp con record sk, ct, ss
```

Per ACVP usare la coppia originale prompt/expectedResults ML-KEM, gruppi
`parameterSet = ML-KEM-768`, `function = decapsulation`, campi `dk`, `c` nel
prompt e `k` nella risposta, associati tramite `tgId` e `tcId`. Sono supportati
anche i wrapper JSON ACVP con intestazione di versione. Nei file RSP usare
record separati da righe vuote con `sk`, `ct`, `ss` esadecimali. Nessun vettore
è incluso. Un formato diverso richiede un adapter revisionato: non viene
ignorato silenziosamente. Directory impostata ma vuota/invalida causa fallimento.
Senza backend o variabile d'ambiente il test salta con motivo esplicito.

### Harness di handshake (non eseguito)

```bash
# Sostituire tutti i TODO con scelte concordate; non è un comando pronto all'uso.
python scripts/bench_handshake.py --config configs/experiment.yaml \
  --output runs/handshake.jsonl --kem hybrid \
  --repetitions TODO --warmup TODO --percentiles TODO
```

Creare prima la directory di output. `--kem` accetta `mlkem` o `hybrid`.
Tutti i conteggi di ripetizioni, warm-up e percentili sono obbligatori, senza
default. Compilare `run_seed` e metadati pubblici nel YAML: il seed non influenza
la casualità dei KEM reali. Le quattro righe JSONL sono riepiloghi di keygen,
encaps, decaps e handshake, con mediana in `duration_s`, percentili nel campo
opzionale `summary` e `phase=crypto`. I round sono `null`.

Ogni campione esegue keygen, encaps, decaps una volta; i tempi delle componenti
sono intervalli della stessa sequenza e handshake è il tempo totale. Esclusi
import, selezione backend, I/O, controllo di uguaglianza e statistiche. Incluse
serializzazione del wrapper e derivazione ibrida. Nessuna autenticazione,
trasmissione di rete, rotazione o TLS viene misurata: “handshake” significa qui
solo round trip KEM locale. I byte sono payload per singolo campione, non totali
delle ripetizioni: keygen/decaps zero, encaps ciphertext iniziatore→risponditore,
handshake anche chiave pubblica risponditore→iniziatore. Non sommare le quattro
righe. Per confrontare B/C adottare identica frontiera API, percentili, warm-up,
ruoli, hardware e politica dei byte; confronti wire richiedono un harness distinto.
