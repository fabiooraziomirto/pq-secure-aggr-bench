# Interfaccia comune

## Contratto obbligatorio

```python
class Aggregator(ABC):
    actors: list[str]
    participation_policy: str
    rekey_policy: str
    communication_edges: dict[str, set[tuple[str, str]]]

    def setup(self, site_ids: list[str]) -> None: ...
    def rekey(self) -> None: ...
    def aggregate_round(self, round_idx: int, updates: dict[str, Tensor]) -> Tensor: ...
    def stats(self) -> RoundStats: ...
```

`Tensor` è un array NumPy reale, convertito in float64. L’output è la **somma**,
mai la media, con forma preservata e tolleranza numerica dichiarata. Forma,
ordine degli adapter LoRA, dtype e flattening sono concordati fuori banda.
Gli input del simulatore sono sintetici; in produzione ogni aggiornamento resta
locale fino alla protezione prevista dal protocollo. Nessun dato clinico nei test.

`setup` è chiamato una volta, accetta ID stringa univoci non vuoti, almeno due.
`actors` elenca tutti i siti e gli altri attori, qui `reducer` (ID riservato).
`participation_policy="all_or_abort"`: un sito mancante o inatteso solleva
`MissingParticipantError`, sottoclasse di ValueError, prima di traffico, consumo
del round o output parziale. Forma errata solleva ValueError prima del traffico.
I round sono interi non negativi univoci nell’intera sessione, anche dopo rekey;
un round iniziato e poi fallito non deve riutilizzare le maschere.

`rekey_policy="refresh_all_pairs"` in A rigenera n−1 coppie di chiavi e tutti i
seed, incrementa epoch e non azzera used_rounds. `rekey_policy="no_op"` negli
stub B/C dichiara assenza di chiavi: epoch invariata, contatori dell’operazione
azzerati e nessun messaggio. La politica dei futuri core B/C dovrà essere esplicita.
`rekey_every_rounds: TODO` non introduce frequenze automatiche.

`stats()` restituisce una copia isolata di `RoundStats`: bytes_sent e
bytes_received per tutti gli attori, più transfers (lista di Transfer con sender,
receiver e size). Sono totali dell’ultima operazione setup, round o rekey; in A
setup_stats e rekey_stats conservano copie dedicate. Il registro contiene solo
metadati e **non è scritto nel JSONL**, né contiene payload o segreti.

Ogni messaggio passa da `ByteCounter.send(sender, receiver, serialized_bytes)`:
la consegna simulata usa gli stessi byte, chiama counter.transfer con len e
registra il trasferimento. Contatori e registro devono riconciliarsi esattamente.
Inclusa diffusione del risultato; esclusi framing di trasporto e TLS. Si contano
payload concreti, non stime o byte di rete fisica. Un adapter di rete futuro deve
preservare questa frontiera e registrare messaggi aggiuntivi.

`communication_edges` dichiara collegamenti diretti per setup, round e rekey.
A scambia fra coppie di siti in setup/rekey e usa una stella sito↔reducer nei
round. Gli stub B/C usano solo la stella nei round. I test richiedono byte positivi
per **ogni collegamento dichiarato**, assenza di trasferimenti su collegamenti
non dichiarati e corrispondenza tra len consegnati, registro, stats e log.
Non si impone un full mesh a protocolli a stella: richiederlo aggiungerebbe
messaggi artificiali e altererebbe il confronto. Questa precisazione sostituisce
il requisito letterale di positività per ogni possibile coppia di attori.

## Schema A: quantizzazione e maschere v2

`SchemeA(kem, *, scale_bits, modulus_bits, clip_bound, writer=None,
authenticator=None, key_provider=None)` richiede tre parametri numerici espliciti.
`0 <= scale_bits < modulus_bits <= 62`; clip_bound finito, non negativo e mai
None. Nessun valore numerico predefinito. Passo 2^(-scale_bits), arrotondamento al
più vicino con pareggi al pari; errore individuale massimo di mezzo passo.
Il limite non applica clipping: coordinate oltre clip_bound sono rifiutate.

Il costruttore non riceve n e controlla la sicurezza per almeno 2 siti;
setup, appena noto n e prima delle chiavi, richiede
n × clip_bound × 2^scale_bits < 2^(modulus_bits−1), e verifica anche il limite
conservativo n × round-to-even(clip_bound × 2^scale_bits), includendo l’arrotondamento.
Il simulatore controlla inoltre la somma intera quantizzata con interi Python:
fuori da [-2^(modulus_bits−1), 2^(modulus_bits−1)−1] solleva
`AggregateOverflowError`. È possibile solo perché conosce gli input; un reducer
reale richiede il limite verificato localmente. I test di questa difesa aggiuntiva
iniettano deliberatamente una configurazione corrotta, non consentita dal
costruttore normale. Non è un modo per disabilitare il limite in produzione.

Si generano n−1 coppie di chiavi: l’ID minimo non è mai destinatario di encaps.
Ogni coppia ordinata lessicograficamente esegue una KEM. Ciascun lato ricava
la propria maschera: HKDF-SHA256, salt assente, output 32 byte, info JSON ASCII
compatto `["pqfl/scheme-a/mask/v2", epoch, round_idx, left, right]`.
`pairwise_mask` richiede epoch esplicita. ChaCha20 di cryptography espande con
nonce nullo di 16 byte e chiave distinta per epoca/coppia/round. Ogni elemento
usa 8 byte little-endian troncati ai bit modulari. ID minore somma, maggiore
sottrae. Mai riusare un round nella sessione.

Conferma separata HKDF-SHA256 a 32 byte, salt assente, info JSON ASCII compatto
`["pqfl/scheme-a/confirm/v1", epoch, left, right, hex(SHA-256(public || ciphertext))]`.
Ogni lato usa il proprio segreto KEM per HMAC-SHA256 del ruolo ASCII left/right
e verifica il tag dell’altro tramite hmac.compare_digest. Due tag da 32 byte
per coppia, contati con len, senza confronto diretto dei segreti.
`KeyConfirmationError` invalida la disponibilità dell’epoca: un nuovo rekey
riuscito è necessario prima di riprendere. Nessun segreto nei messaggi d’errore.

`serialize_masked` codifica interi modulari canonici in ceil(modulus_bits/8) byte
little-endian, ordine C, senza intestazione; forma e bit sono noti fuori banda.
`deserialize_masked` rifiuta lunghezze inattese e bit alti non canonici.
Ogni sito quantizza, calcola le proprie n−1 maschere, applica e serializza;
il reducer deserializza, somma modulo 2^modulus_bits, serializza e invia a tutti;
ogni sito deserializza e dequantizza. `_masked_site` è un metodo interno di test,
non un endpoint per aggirare l’univocità dei round. `recover_masks` resta uno stub.

## Autenticazione

`Authenticator.issue(owner, epoch, public_key)` produce AuthProof;
`verify(verifier, owner, epoch, public_key, proof)` accetta oppure solleva
`AuthenticationError`, **prima di encaps e della derivazione del seed**.
Ogni verificatore possiede il proprio registro fuori banda.

Default PinnedFingerprintAuth vuoto: fallisce senza pin, senza trust-on-first-use.
from_config legge verifier → epoch → owner → fingerprint SHA256 esadecimale.
Per pinnare chiavi KEM effimere prima dello scambio il chiamante usa
key_provider(owner, epoch), con nuove coppie pre-provisionate e pin autentici.
Non procurarsi chiave e fingerprint dallo stesso collegamento non attendibile.
La generazione offline delle coppie del provider è un costo distinto dalla
lettura del provider misurata in setup e va dichiarata in un confronto reale.

MLDSAAuth usa solo ML-DSA-65 abilitata nel backend reale. Identità a lungo termine,
chiavi segrete da storage sicuro e registri di chiavi identità pubbliche pinnate
sono iniettati dal chiamante; niente segreti nel YAML. Messaggio firmato:
JSON ASCII compatto `["pqfl/auth/v1", epoch, site_id, public_key.hex()]`.
La prova non è valida per un’altra epoca, identità o chiave. Dimensioni da oqs;
il test richiede firma di 3309 byte, ma la misura nativa resta da verificare qui.

Si firma una volta per proprietario/epoca, poi si inviano chiave KEM, chiave
identità e firma a ogni destinatario interessato; si contano i len effettivi
tramite lo stesso counter.transfer (invocato da send). Non ricontare la chiave
KEM nella fase crypto. Nel pinning non ci sono firma e chiave identità sul wire.
Questa autenticazione non sostituisce l’integrità del canale e dei vettori.

## Log e frontiera temporale

Schema_version resta **2**. Campi obbligatori: schema_version, run_id, config_hash,
scheme, dp_level, run_seed, round, phase, site, epoch, duration_s, bytes_sent,
bytes_received, peak_mem_gpu_mb, peak_mem_host_mb, timestamp, library_versions.
Un file nuovo per run; file esistente rifiutato. run_seed è pubblico, dal YAML,
intero non negativo; non inizializza mai un KEM reale. config_hash è SHA256 del
JSON pubblico con chiavi ordinate e separatori compatti. Memoria non misurata null.

Fasi ammesse: **auth, crypto, compute, comm**. Timer non annidati per attore:
`with metrics.phase_timer("auth", site="A"):`. Auth include firma/verifica;
crypto include KEM, conferma e maschere; compute quantizzazione, serializzazione
e reducer. Comm richiede trasporto reale: nessuna riga comm nei simulatori.
Il controllo di overflow aggiunge lavoro al simulatore, non è un costo di un
reducer privato senza accesso agli input.

Ogni riga Metrics riporta il DELTA dei byte dall’ultima riga della stessa istanza,
non dall’ultima riga dello stesso sito. Si possono sommare le righe di
un’operazione per ricostruire stats; nuova Metrics per ogni operazione. Site
attribuisce il lavoro, mentre i delta includono tutti gli attori coinvolti.
Setup epoch=0 e round=null; rekey round=null e nuova epoca per A. Gli stub con
rekey no-op mantengono epoch. Il registro transfers non cambia lo schema JSONL.

Env è opzionale solo nel primo record setup: versioni Python/cryptography,
binding/liboqs, CPU e KEM/SIG abilitati, dal collector scripts/check_env.py.
Native assente: liboqs=null, liste vuote, oqs_available=false. Nessuna variabile
segreta o credenziale letta. L’harness può usare HandshakeSummary (riepiloghi
locali, duration_s mediana, percentili lineari): non sono round di aggregazione
e non si sommano le righe componenti e handshake.

Non registrare seed crittografici, chiavi, segreti condivisi, ciphertext, maschere,
tag o firme, nemmeno in metadati o configurazioni. Il registro conserva solo
attori e dimensioni. Pin pubblici e trust configuration rimangono fuori dai log.

## Backend reale e composizione ibrida

KEM conserva keygen() → (public, secret), encaps(public) → (ciphertext, shared),
decaps(secret, ciphertext) → shared. LibOQSKEM rileva ML-KEM-768; Kyber768 è
legacy diverso e richiede allow_legacy=True esplicito. BackendUnavailable con
istruzioni se manca una capacità; nessuna installazione automatica o FakeKEM
nel pacchetto. Le versioni effettive sono in library_versions.

HybridKEM di default usa ML-KEM-768 reale; l’iniezione FakeKEM esiste solo nei test.
Pack v1: prefisso ASCII pqfl/hybrid-kem/v1/, tag pk/sk/ct/combiner, poi campi
con lunghezza uint32 big-endian. Pk: nome algoritmo, X25519 destinatario, pk PQ;
sk: privato X25519, sk PQ, pk ibrida completa; ct: pubblico X25519 effimero e ct PQ.
HKDF-SHA256 a 32 byte su shared_X25519 concatenato a shared_PQ, salt assente,
info pack combiner contenente pk e ct ibridi completi. Include entrambe le chiavi
pubbliche e componenti ciphertext. Non è un canale autenticato né un profilo TLS
standard; TODO revisione da parte di un esperto. Nessuna cancellazione garantita
delle copie di segreti in memoria Python.

Per nome esatto ML-KEM-768: pk PQ 1184 byte, ct PQ 1088, ss 32; X25519 pubblico 32.
Pack(pk) **1259 byte**, pack(ct) **1149 byte**, verificati sul serializer con
componenti delle dimensioni standard, non su un backend nativo funzionante qui.
I test reali verificano dimensioni e round trip, KAT ufficiali e interop OpenSSL.
make test-real fallisce per qualsiasi skip requires_oqs. Native 0.16.0 non
compilata per CMake mancante; dettagli e provenienza KAT in decisions.md.

## Consegna B e C e conformità

AGGREGATOR_REGISTRY contiene A, B e C. B e C oggi sono **INSECURE_STUB**:
somma in chiaro con float64 little-endian, payload realmente serializzati e
consegnati nella stella, risultato diffuso a tutti. Nessuna HE, MPC, cifratura
o autenticazione. Questi byte descrivono solo gli stub, mai costi CKKS/MPC.
Il chiamante deve etichettare gli esperimenti come stub, non risultati B/C reali.

Sostituire il core mantenendo ABC e suite tests/test_aggregator_contract.py.
Dichiarare numerica/tolleranze, actors, partecipazione, rekey e collegamenti;
registrare ogni messaggio e misurare le stesse fasi. CKKS dovrà dichiarare
parametri e precisione; MPC dominio e conversioni. Le policy degli stub non
pre-approvano le scelte dei colleghi. Nessun benchmark, training o dato clinico
incluso. I test sintetici e il vettore dorato prodotto dalla funzione corrente
sono regressioni, non prove crittografiche indipendenti.

## Punti aperti per i colleghi

1. **Collusione reducer più un sito**: cosa dichiara ciascuno schema reale?
   In A la conoscenza del seed di una coppia consente di togliere quella maschera;
   non basta a recuperare l’update se resta una maschera ignota con il terzo sito.
   Per isolare un singolo update servono tutti i suoi contributi di maschera
   rilevanti (o ulteriori informazioni). Con tre siti e nessuna informazione
   ausiliaria, reducer più un sito apprendono la somma degli altri due. La
   formula “basta il seed della coppia col colluso” è quindi solo necessaria,
   non sufficiente nel caso a tre siti. B e C: dichiarazioni da concordare con
   i responsabili, non dedotte dagli stub.
2. **H6**: riportare il costo handshake come mediana dei componenti o misura
   end-to-end? Decisione aperta con i colleghi. La somma delle mediane non è
   in generale la mediana del totale; l’harness può osservare entrambe le
   quantità ma non decide quale riportare nel paper. Nessun dato prodotto qui.
