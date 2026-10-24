# Interfaccia comune

## Contratto

```python
class KEM(Protocol):
    def keygen(self) -> tuple[bytes, bytes]: ...  # public, secret
    def encaps(self, public: bytes) -> tuple[bytes, bytes]: ...  # ciphertext, shared_secret
    def decaps(self, secret: bytes, ciphertext: bytes) -> bytes: ...

class Aggregator(ABC):
    def setup(self, site_ids: list[str]) -> None: ...
    def aggregate_round(self, round_idx: int, updates: dict[str, Tensor]) -> Tensor: ...
    def stats(self) -> RoundStats: ...
```

`Tensor` è attualmente un array NumPy reale, convertito in float64 in ingresso;
forme identiche, tutti i siti presenti e ID stringa univoci non vuoti.
L'output è la **somma**, mai la media, nella forma iniziale e in float64.
La pipeline deve concordare separatamente ordinamento degli adapter, flattening,
dtype e metadati di forma. Nessun dato clinico attraversa l'interfaccia.
Il simulatore riceve aggiornamenti sintetici; in produzione gli aggiornamenti
rimangono nel processo locale fino alla protezione prevista dal protocollo.

`setup` è chiamato una volta per istanza/sessione. I round sono interi non
negativi univoci nella sessione. Riavviare una sessione reale richiederà nuove
chiavi e un identificatore di sessione concordato; la gestione persistente è TODO.
`stats()` restituisce una copia dei contatori per sito dell'ultima operazione:
dopo setup sono costi setup, dopo un round solo i costi di quel round.
Lo schema A conserva anche `setup_stats`; non è un'estensione obbligatoria API.

## Schema A

`SchemeA(kem, scale_bits=..., modulus_bits=..., writer=None)` richiede entrambi
i parametri senza default: `0 <= scale_bits < modulus_bits <= 62`.
Quantizzazione: arrotondamento al più vicino con pareggi al pari, passo
`2**(-scale_bits)`, errore individuale al massimo mezzo passo. Ogni aggiornamento
deve rientrare nell'intervallo con segno; anche la somma deve rientrarvi per
interpretare correttamente la dequantizzazione. La somma modulare non rileva
l'overflow aggregato: scelta dei limiti e clipping sono TODO del protocollo.

Ogni coppia ordinata lessicograficamente esegue una KEM in setup. HKDF-SHA256
usa come info la codifica JSON ASCII compatta della lista
`["pqfl/scheme-a/mask/v1", round_idx, left, right]`, salt assente, output 32 byte.
ChaCha20 di cryptography espande con nonce di 16 byte nullo: la chiave derivata
cambia per coppia/round. Ogni elemento usa 8 byte little-endian troncati ai bit
modulari; l'ID minore aggiunge, il maggiore sottrae. Nessun round si può riusare.
`recover_masks(...)` è uno stub per secret sharing dei seed; dropout rifiutato.
Il canale ibrido autenticato X25519 + ML-KEM-768 e i backend reali restano stub.

Esempio di contratto, da completare con un backend verificato (non eseguibile
finché tale backend non esiste):

```python
import numpy as np
from pqfl.aggregators.scheme_a import SchemeA
from pqfl.logging.metrics import JSONLWriter

# kem = verified_backend  # TODO
# scale_bits, modulus_bits = reviewed_configuration  # TODO
writer = JSONLWriter.from_config("run.jsonl", "configs/experiment.yaml")
# Set run_seed and other public metadata in the YAML before running.
# agg = SchemeA(kem, scale_bits=scale_bits, modulus_bits=modulus_bits, writer=writer)
# agg.setup(["A", "B", "C"])
# result = agg.aggregate_round(0, {s: np.zeros(4) for s in ["A", "B", "C"]})
```

Per un esempio eseguibile usare `tests/test_scheme_a.py`, con FakeKEM INSECURE
confinata ai test. Il seed di riproducibilità usa solo il generatore fittizio;
nessun parametro seed deterministico è previsto per KEM reali.

## Log e misure

Un file nuovo per esecuzione; `JSONLWriter` rifiuta un file esistente.
Usare `Metrics(site_ids, writer)` e `with metrics.phase_timer("crypto"):`;
le altre fasi sono `compute` e `comm`. Assegnare `metrics.round_idx` e usare
`metrics.counter.transfer(sender, receiver, nbytes)` per messaggi tra siti.
I contatori nel JSONL sono snapshot cumulativi dell'operazione, non delta da
sommare tra righe. Setup ha round `null`. Nessuna sovrapposizione intenzionale
tra fasi; i tempi misurano il blocco locale, non prestazioni distribuite.

Campi (schema_version = 1): schema_version, run_id, config_hash, scheme, dp_level,
run_seed, round, phase, duration_s,
bytes_sent, bytes_received, peak_mem_gpu_mb, peak_mem_host_mb, timestamp,
library_versions. `run_seed` è il seed pubblico dell’esperimento letto dalla
configurazione YAML; il vecchio campo `seed` è rimosso. Il template lascia
`run_seed: TODO`: prima di creare un log va scelto un intero non negativo.
Questo valore non inizializza mai la casualità crittografica. Non registrare
seed crittografici, chiavi, segreti condivisi, ciphertext o maschere. Memoria non misurata: `null`, mai zero inventato. Librerie
riportate nelle versioni realmente installate. L’hash SHA256 è calcolato sul JSON della configurazione pubblica, chiavi ordinate
e separatori compatti. Collector memoria e inventario completo backend: TODO. Il chiamante
non deve inserire segreti neppure nei metadati testuali o nel file configurazione.

Contare payload pubblici KEM e ciphertext in setup per coppia, e payload logici
modulari al reducer esterno nei round di A: `numero_elementi * ceil(modulus_bits/8)`.
Il reducer non è un sito; ricezione dei siti pari a zero se non c'è broadcast.
Sono conteggi simulati: non byte effettivamente serializzati o inviati su rete.
Framing, autenticazione e canale vanno aggiunti e misurati separatamente quando
esisterà il trasporto. Non confrontare questi conteggi con traffico wire di B/C.

## Consegna B e C

Consegnare implementazione dell'ABC, configurazioni senza default arbitrari,
serializzazione documentata, adattamento Tensor, test di somma/errore e
strumentazione comune per setup, compute, crypto e comm. Separare messaggi di
protocollo, broadcast e canale, contare invio/ricezione per sito senza doppio
conteggio e documentare l'eventuale coordinatore. CKKS deve dichiarare parametri,
precisione e tolleranza testata; MPC deve dichiarare dominio e conversione.
Dichiarare librerie/versioni realmente validate, numero/soglia di partecipanti,
assunzioni su collusione e dropout. Non consegnare benchmark o dati clinici.
Qualsiasi confronto di prestazioni richiede prima la stessa frontiera di misura.
