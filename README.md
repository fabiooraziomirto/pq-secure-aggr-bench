# Aggregazione post-quantum di aggiornamenti LoRA

Base di ricerca per tre siti A, B, C. Nessun dato clinico incluso.
Licenza: Apache-2.0 (file `LICENSE`, testo ufficiale non modificato). Versione 0.0.0: segnaposto del pacchetto, non release scientifica.

## Setup e test

Python >=3.10 (minimo richiesto dalle annotazioni); versione del runtime di
riferimento e versioni riproducibili delle dipendenze: TODO.

```bash
cd pq-secure-aggr-bench
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

## Come aggiungere uno schema

1. Crea il modulo dell’aggregatore in src/pqfl/aggregators.
2. Implementa setup, rekey, aggregate_round e stats dell’ABC.
3. Dichiara actors, participation_policy e rekey_policy.
4. Elenca i communication_edges per setup, round e rekey.
5. Serializza ogni payload e consegnalo tramite ByteCounter.send.
6. Includi upload e diffusione del risultato nel registro dei trasferimenti.
7. Usa i timer comuni auth, crypto, compute e comm per attore.
8. Dichiara tolleranza numerica, librerie e stato di sicurezza del core.
9. Aggiungi la classe ad AGGREGATOR_REGISTRY e configura il factory dei test.
10. Esegui la suite di conformità e make test-real per ogni backend nativo.

C implementa secret sharing additivo (`scheme_c.py`, `masking/sharing.py`) sopra il
setup PQ autenticato di A. B implementa CKKS a soglia n-di-n con OpenFHE
(`scheme_b.py`, `he/ckks.py`), da installare in un ambiente separato
(`pip install '.[he]'`, wheel ufficiale openfhe 1.6.1.0.22.4); i test B si saltano
senza il wheel. `InsecureSumStub` resta solo come baseline etichettata. Il traffico dello stub non rappresenta costi CKKS e non può essere usato come risultato
del paper. Specifica in [docs/interface.md](docs/interface.md), assunzioni in
[docs/threat_model.md](docs/threat_model.md) e decisioni in
[docs/decisions.md](docs/decisions.md).

`configs/experiment.yaml` è un template con TODO per gli esperimenti finali LoRA: il
caricatore lo rifiuta. `configs/sim_costs.yaml` è completo e serve ai run di costo
del simulatore su aggiornamenti sintetici (nessun modello, dati clinici o DP).

```bash
# Un ambiente con liboqs-python e openfhe; OQS_INSTALL_PATH/LD_LIBRARY_PATH per liboqs.
python scripts/run_experiment.py --config configs/sim_costs.yaml --scheme A \
  --output runs/A.jsonl --rounds 5 --warmup 1 --vector-size 100000   # A, B, C, STUB
python scripts/summarize_runs.py runs/A.jsonl runs/B.jsonl --percentiles 50 95
python scripts/bench_scaling.py --config configs/sim_costs.yaml --output-dir runs/scaling \
  --schemes STUB A C B --sites 3 5 10 --vector-sizes 10000 100000 \
  --repetitions 5 --warmup 1 --percentiles 50 95
```

`--output` e `--output-dir` devono essere nuovi (`runs/` è ignorata da git). Rounds,
warm-up, siti, dimensioni e percentili sono obbligatori e senza default. Ogni cella
dello sweep è una sessione nuova (setup registrato, warm-up non registrato) con un
JSONL proprio e una riga di `summary.csv`. Il runner verifica la somma a ogni round.
I tempi sono somme dei timer del simulatore seriale in-process: `cpu_total_s` su tutti
gli attori, `site_max_s` per il sito più carico, `reducer_s`; non sono un wall-clock
distribuito. I byte sono la somma inviata da tutti gli attori per round. Senza
rete reale né ripetizione tra sessioni/hardware, restano misure del simulatore.

## Misure di rete con i container

Gli schemi restano simulatori in-process; la rete si misura **riproducendo il registro
dei trasferimenti** (solo mittente, destinatario, dimensione e stadio causale, senza
payload) tra agenti TCP reali, uno per attore. Payload di zeri: il costo di rete
dipende solo dalle dimensioni. Il tempo di un trasferimento va dal primo byte
all'ack di 1 byte del destinatario (include il mezzo RTT di ritorno); il tempo di uno
stadio è il più lento dei suoi trasferimenti concorrenti; il tempo di round è la somma
degli stadi. Il calcolo non è incluso: un'end-to-end stimata è `cpu_total_s` del
simulatore più il tempo di rete, un limite superiore perché il calcolo simulato è seriale.

```bash
python scripts/run_experiment.py --config configs/sim_costs.yaml --scheme C \
  --output runs/C.jsonl --rounds 5 --warmup 1 --vector-size 100000 \
  --ledger-output runs/C-ledger.json
# agenti: uno per container (container/compose.yaml) oppure python scripts/net_agent.py
python scripts/net_replay.py --ledger runs/C-ledger.json --profile wan \
  --repetitions 30 --warmup 3 --output runs/C-wan-net.json \
  --agents A=site-a:9000 B=site-b:9000 C=site-c:9000 reducer=reducer:9000
```

`net_replay.py` fallisce se i byte ricevuti dagli agenti non coincidono con il registro.
Profili in `configs/network_profiles.yaml` (`lan`, `metro`, `wan`, `constrained`):
`scripts/netem_commands.py` stampa i comandi `tc` (non li esegue); l'entrypoint
del container li applica con NET_ADMIN. I parametri sono ipotesi plausibili, da
dichiarare come emulazione. `container/Dockerfile`, `entrypoint.sh` e `compose.yaml`
(CPU, ubuntu:22.04, liboqs al commit verificato, OpenFHE e liboqs-python con versione
fissata, `cpuset` per container); il lockfile con hash resta da generare.
`container/pqfl.def` (Apptainer) resta una bozza non usata.

Immagine costruita e compose provato il 2026-10-30 (digest, versioni effettive e
verifica di `tc` in `docs/decisions.md`). Imposta `CPUSET_A`, `CPUSET_B`, `CPUSET_C`,
`CPUSET_REDUCER` con core esistenti sull'host prima di `docker compose up`.
Attenzione: sui trasferimenti da centinaia di KB con RTT alti il tempo dipende da slow
start e congestion control di TCP, e ha mostrato variabilità tra sessioni; nella
campagna finale registra `tcp_congestion_control`, `tcp_slow_start_after_idle` e ripeti le
sessioni.
`scripts/check_env.sh` va eseguito sul cluster: controlla OpenSSL/ML-KEM,
liboqs e tc; non certifica sicurezza, interoperabilità o permessi di rete.

Non salvare dati clinici, chiavi, segreti o credenziali. Usare identificatori
neutri e non aggiungere autori, affiliazioni o ringraziamenti per double-blind.

I log JSONL usano `schema_version: 2`, `site`, `epoch`, delta dei byte sommabili
per operazione e `run_seed` pubblico dalla configurazione YAML (intero non negativo da scegliere al posto di TODO). Non viene usato per
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
# Replace every TODO with agreed choices; this is not a ready-to-run command.
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

## Contratto aggiornato dello schema A

Il reducer è un attore separato: i byte comprendono upload e diffusione del
risultato, calcolati con len dei messaggi serializzati. Nessuna riga comm nel
simulatore. Le righe Metrics riportano delta per attore; `stats()` riporta totali
dell’ultima operazione. I tempi crypto sono separati per sito, con ogni lato
che calcola le proprie copie delle maschere.

`rekey()` rigenera le chiavi e i seed di tutte le coppie, incrementa epoch e
conserva l’univocità dei round. `rekey_every_rounds: TODO` non attiva alcuna
politica automatica. L’info delle maschere passa a mask/v2 con epoch esplicita.
La conferma bidirezionale HMAC rileva segreti non corrispondenti e può sollevare
`KeyConfirmationError`; non autentica la provenienza delle chiavi pubbliche.
`clip_bound` è obbligatorio e keyword-only, senza default, e rifiuta aggiornamenti oltre il limite; setup ne verifica
la sicurezza per il numero effettivo di siti. `AggregateOverflowError` rileva una
somma quantizzata fuori range soltanto nel simulatore che vede i testi in chiaro.

I test extra includono vettori sintetici di 100000 elementi, controllo dei bit,
rekey, conferma, overflow, payload/delta e rifiuto X25519 di ordine basso.
Il vettore dorato di mask/v2 è una regressione prodotta dall’implementazione,
non un vettore crittografico indipendente. Questi controlli non provano sicurezza.

`tests/test_kem_interop.py` rileva opzioni reali di `openssl pkeyutl -help` e
supporto ML-KEM-768 prima di usare genpkey/pkey/pkeyutl per due round trip
incrociati con liboqs. Conserva gli identificatori ASN.1 esportati da OpenSSL,
senza ipotizzare codifiche della chiave privata. I file temporanei sono esterni
al repository. Se CLI/capacità/binding mancano, salta con motivo e TODO cluster.
Eseguire sul cluster `python -m pytest -q -rs`; non installare nulla da rete
nell’ambiente di questa sessione e non avviare l’harness di benchmark.

## Test reali obbligatori

Con Python dell’ambiente attivo, `make test-real` esegue tutti i test marcati
`requires_oqs` e fallisce anche se uno solo è saltato (inclusa l’assenza dei KAT).
Per la suite completa usare `make test`. Non è un benchmark.
`scripts/check_env.py` stampa versioni effettive Python/cryptography/liboqs,
CPU e algoritmi KEM/SIG; lo stesso oggetto pubblico appare nel campo opzionale
`env` del primo record di setup di ogni run, mantenendo schema_version=2.
Installazione reale tentata ed esito sono documentati in `docs/decisions.md`.
Il 2026-10-30 liboqs 0.16.0 (commit 5a1a854b) e il binding 0.16.0.1 sono stati
compilati e verificati: suite completa 81 passati, 1 saltato. Il test saltato è
l'interoperabilità liboqs/OpenSSL, perché OpenSSL 3.0.12 non supporta ML-KEM
(serve ≥3.5). `make test-real` fallisce finché quel test è saltato.

## Chiavi autenticate e policy di partecipazione

SchemeA richiede un clip_bound numerico esplicito e un registro attendibile:
PinnedFingerprintAuth è il default, ma senza pin il setup fallisce. Per chiavi
pre-provisionate usare il key_provider e registri per epoca; in alternativa
iniettare MLDSAAuth con identità ML-DSA-65 e chiavi pubbliche pinnate fuori banda.
Non fornire segreti nel YAML. I test funzionali non dedicati all’autenticazione
iniettano un bypass INSECURE confinato ai test, mai scelto dal pacchetto.
La fase auth conta chiavi identità e firme; schema_version resta 2.
Il nome API esatto è `participation_policy="all_or_abort"`;
siti mancanti causano MissingParticipantError e nessun risultato parziale.
I test ML-DSA richiedono liboqs nativa e sono marcati requires_oqs.

## Suite di conformità

`python -m pytest -q tests/test_aggregator_contract.py` verifica registro e byte,
somma, diffusione, policy di partecipazione e rekey/no-op. A reale usa ML-KEM
ibrida e identità ML-DSA: è un test requires_oqs, non sostituito da FakeKEM se
manca il backend. A funzionale è un caso aggiuntivo esplicitamente INSECURE.
B/C restano stub senza crittografia. Il controllo di positività riguarda i
collegamenti effettivi dichiarati, senza imporre traffico full mesh a una stella.
Restano aperti il modello di collusione dei core B/C e la presentazione di H6.
