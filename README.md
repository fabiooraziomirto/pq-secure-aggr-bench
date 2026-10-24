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
INSECURE. Non è disponibile alcun backend crittografico pronto per produzione.

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

I log JSONL usano `schema_version: 1` e `run_seed` pubblico dalla configurazione
YAML (intero non negativo da scegliere al posto di TODO). Non viene usato per
inizializzare KEM reali; nessun materiale crittografico deve essere registrato.
