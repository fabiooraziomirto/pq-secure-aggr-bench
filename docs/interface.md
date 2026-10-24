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
`stats()` restituisce una copia dei contatori per attore (siti e `reducer`) dell'ultima operazione:
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
Il backend liboqs reale e `pqfl.kem.HybridKEM` implementano ora KEM;
il trasporto autenticato `HybridChannel` e il backend OpenSSL restano stub.

Esempio di contratto da completare con configurazione e backend verificati sul cluster:

```python
import numpy as np
from pqfl.aggregators.scheme_a import SchemeA
from pqfl.logging.metrics import JSONLWriter

from pqfl.kem import HybridKEM
# kem = HybridKEM()  # Requires reviewed native liboqs and Python binding.
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
Usare `Metrics([*site_ids, "reducer"], writer, epoch=0)` e
`with metrics.phase_timer("crypto", site="A"):`;
le altre fasi sono `compute` e `comm`. Assegnare `metrics.round_idx` e usare
`metrics.counter.transfer(sender, receiver, nbytes)` per messaggi tra siti.
Ogni riga riporta il DELTA dei byte dall’ultima riga scritta dalla stessa
istanza Metrics (inizialmente zero per ogni attore). Le righe sono sommabili
per ottenere stats() dell’operazione; i contatori si azzerano con la nuova
Metrics di ciascun round. Setup ha round `null`, epoca 0. `site` attribuisce
il blocco di lavoro a un attore; i delta includono tutti gli attori interessati,
non solo il valore di site. Timer non annidati. Ogni sito quantizza, calcola
le proprie n−1 maschere, applica e serializza; entrambi i lati di ogni coppia
ricalcolano la maschera. Il reducer deserializza, somma e serializza il risultato;
ogni sito lo deserializza e dequantizza. Nessuna riga `comm` nel simulatore:
non c’è trasporto reale. La fase resta valida per implementazioni future.
I tempi sono lavoro locale per attore, non latenza distribuita.

Campi (schema_version = 2): schema_version, run_id, config_hash, scheme, dp_level,
run_seed, round, phase, site, epoch, duration_s,
bytes_sent, bytes_received, peak_mem_gpu_mb, peak_mem_host_mb, timestamp,
library_versions. `run_seed` è il seed pubblico dell’esperimento letto dalla
configurazione YAML; il vecchio campo `seed` è rimosso. Il template lascia
`run_seed: TODO`: prima di creare un log va scelto un intero non negativo.
Questo valore non inizializza mai la casualità crittografica. Non registrare
seed crittografici, chiavi, segreti condivisi, ciphertext, maschere o tag di conferma. Memoria non misurata: `null`, mai zero inventato. Librerie
riportate nelle versioni realmente installate. L’hash SHA256 è calcolato sul JSON della configurazione pubblica, chiavi ordinate
e separatori compatti. Collector memoria e inventario completo backend: TODO. Il chiamante
non deve inserire segreti neppure nei metadati testuali o nel file configurazione.

La frontiera include solo payload dei messaggi di protocollo: chiavi pubbliche,
ciphertext, vettori mascherati e risultato aggregato, con diffusione a tutti i
siti inclusa. Tutti i conteggi usano `len()` dei messaggi realmente serializzati,
mai una formula stimata. Esclusi framing di trasporto, TLS e autenticazione.
`serialize_masked` codifica ogni intero modulare canonico in
`ceil(modulus_bits/8)` byte little-endian, ordine C, senza intestazione.
Forma e bit sono concordati fuori banda; `deserialize_masked` rifiuta lunghezze
inattese e bit alti non canonici. `reducer` è un attore riservato, separato da
site_ids: riceve ciascun upload e invia a ogni sito lo stesso risultato
serializzato. `stats()` include invii e ricezioni di tutti gli attori. Si contano
payload concreti, ma non pacchetti di una rete reale; per B/C vale la stessa
frontiera, non un confronto diretto con byte di trasporto.

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

## Backend e composizione ibrida

`LibOQSKEM` usa context manager del binding liboqs-python per generazione,
esportazione della chiave privata, encaps e decaps. Nessuna installazione
automatica; dipendenze assenti sollevano `BackendUnavailable` con istruzioni.
Preferisce ML-KEM-768. Kyber768 richiede `allow_legacy=True`, produce warning e
mantiene il nome legacy; non è una variante intercambiabile dello standard.
Le versioni reali e il nome del meccanismo sono disponibili in `library_versions`.

`HybridKEM(pq_kem=None)` usa ML-KEM-768 reale per default. L'iniezione esplicita
consente test con FakeKEM confinata ai test. La chiave pubblica contiene nome
algoritmo, chiave X25519 destinatario (32 byte) e chiave PQ; la chiave privata
contiene chiave X25519 privata, chiave PQ privata e chiave pubblica completa.
Il ciphertext contiene chiave X25519 effimera mittente e ciphertext PQ.

Codifica v1: prefisso ASCII `pqfl/hybrid-kem/v1/`, tag `pk`, `sk`, `ct` o
`combiner`, poi campi con lunghezza uint32 big-endian e byte esatti, nel precedente
ordine. Rifiutati tag/versioni errati, troncamenti, byte finali e nomi incompatibili.
Il segreto finale è HKDF-SHA256 a 32 byte, salt assente, input
`shared_X25519 || shared_PQ`. L'info è la codifica `combiner` con due campi:
chiave pubblica ibrida completa e ciphertext ibrido completo. Include così nome,
entrambe le chiavi pubbliche e le componenti ciphertext; in X25519 il ciphertext
KEM è la chiave effimera. Chiavi effimere da cryptography, nessun run_seed usato.
Non è fornita cancellazione garantita dei byte segreti nella memoria Python.

La costruzione non è stata rivista formalmente: TODO revisione da parte di un
esperto. Non fornisce autenticazione, protezione replay o trasporto. La scelta
TLS 1.3 ibrido via OpenSSL/canale applicativo resta aperta.

Nel JSONL `library_versions` può includere versioni native e meccanismo reali.
Per la strumentazione usare `HandshakeSummary`, non payload liberi.
L'harness registra un campo opzionale `summary` con operazione, meccanismo,
ripetizioni, warm-up, percentili in secondi (interpolazione lineare), frontiera
temporale e frontiera dei byte. `duration_s` è la mediana dei campioni; ogni riga
è un riepilogo e non una singola osservazione o un round di aggregazione.
