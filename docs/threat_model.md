# Modello di minaccia

Tre siti A, B e C addestrano adapter LoRA dello stesso LLM. Dati clinici grezzi,
record e credenziali restano nei rispettivi siti. Anche gli aggiornamenti e gli
aggregati possono contenere informazioni sensibili: l'aggregazione sicura non
costituisce da sola una garanzia contro membership inference o estrazione.

## Assunzioni

Le chiavi sono autenticate tramite Authenticator con trust anchor fuori banda:
fingerprint per verificatore/epoca oppure identità ML-DSA-65 pinnate. Il default
vuoto rifiuta la chiave, senza trust-on-first-use. La conferma HMAC resta distinta
dall’autenticazione. Identità a lungo termine e chiavi pre-provisionate devono
essere custodite e distribuite in modo sicuro. Nessun dropout: all_or_abort,
nessuna ricostruzione di seed e nessun output parziale.
Il canale non è necessario per la confidenzialità dei vettori mascherati sotto
le assunzioni su KEM/maschere, ma è necessario per l’integrità dei messaggi:
le firme sulle chiavi non autenticano automaticamente ciphertext, aggiornamenti,
ordine dei round o risultato. Il simulatore non implementa tale canale.

## Livello esterno: harvest now, decrypt later

Un avversario registra oggi il traffico e dispone in futuro di capacità
quantistiche. Il canale previsto combina X25519 e ML-KEM-768. L'autenticazione,
il transcript binding, il profilo del combiner e il ciclo di rotazione devono
ancora essere revisionati e verificati. HybridKEM implementa un combiner
sperimentale con transcript binding, non un canale autenticato; HybridChannel
resta uno stub. TODO: revisione da parte di un esperto.
Non si rivendica sicurezza post-quantum dell'implementazione corrente.
MLDSAAuth implementa l’autenticazione della chiave KEM con ML-DSA-65;
restano aperti provisioning, rotazione delle identità e canale autenticato.
Si assume esplicitamente che le chiavi pubbliche dei siti siano autentiche:
un reducer che le inoltra potrebbe altrimenti sostituirle. La conferma HMAC
prova il possesso di un segreto condiviso legato alla trascrizione, non
l’identità del peer o l’autenticità iniziale della chiave pubblica.

A usa seed di coppia da ML-KEM e HKDF/ChaCha20 di libreria. B prevede CKKS a
soglia; C secret sharing additivo a tre parti. Per B e C restano da verificare
assunzioni, parametri, distribuzione delle chiavi, collusione e protezione dei
messaggi. Il solo nome dello schema non dimostra resistenza HNDL end-to-end.
La compromissione futura delle chiavi a lungo termine può esporre sessioni
registrate senza una politica adeguata di chiavi effimere e cancellazione.

## Livello interno: honest-but-curious

I partecipanti seguono il protocollo ma cercano di inferire aggiornamenti altrui
tramite il proprio stato e l'aggregato. Con tre siti e senza collusione, in A
ogni singolo sito osserva il proprio input e una somma degli altri due; due siti
collusi possono ricavare il contributo del terzo dalla somma. Nessuna garanzia
contro tale collusione è dichiarata. Con tre siti, reducer più un solo sito
colluso apprendono al massimo la somma degli altri due, sotto le assunzioni
sui seed e sulle chiavi autentiche e senza informazioni ausiliarie; non i due
contributi separati dal solo protocollo. Questa proprietà riguarda un deployment
con isolamento reale: il simulatore centralizzato non la realizza. Non sono gestiti dropout o equivocation.
Il simulatore centralizzato vede tutti gli aggiornamenti e seed: è esclusivamente
un riferimento funzionale e non realizza la separazione di fiducia tra siti.
DP di record o sito, clipping e accounting sono decisioni aperte e non implementate.

## IPOTESI DA VERIFICARE con i responsabili di B e C

In A i vettori in transito sono mascherati con seed derivati da ML-KEM: sotto
le assunzioni crittografiche e di gestione delle chiavi, la loro riservatezza
non dipende dal solo canale. I messaggi di conferma e le chiavi pubbliche non
rendono pubblici i seed; compromissione successiva delle chiavi di decapsulazione,
riuso delle maschere o sostituzione delle chiavi richiedono protezioni distinte.
In B, CKKS a soglia produce ciphertext già su reticoli: parametri, gestione delle
quote e collusione devono essere confermati dal responsabile. In C, se le share
additive inviate fossero protette dal solo canale e un avversario potesse
registrare abbastanza share per ricostruire ogni aggiornamento, un canale
classico violato in futuro rivelerebbe gli aggiornamenti registrati. Verificare
con il responsabile di C topologia e soglia: una singola share non basta.
Queste sono ipotesi di confronto, non proprietà già validate di B e C.

## Fuori scope

Attestazione con catene classiche, side-channel, server malevolo e avvelenamento
a bassa norma sono fuori scope. Sono escluse anche compromissione degli endpoint
e verifica della correttezza del training. Attacchi di membership inference ed
estrazione saranno moduli separati dei colleghi, non eseguiti in questa base.
Nei log è ammesso solo `run_seed`, seed pubblico di riproducibilità. Non conservare
chiavi, seed crittografici, segreti condivisi, ciphertext, maschere o tag di conferma nei log; nessun dato clinico nei test.

Collusione per lo schema C (secret sharing additivo, n siti più reducer onesto-ma-curioso):
il reducer vede solo somme parziali, ciascuna uniforme se almeno una share
ignota la maschera. Reducer più fino a n-2 siti apprendono soltanto ciò che
l'aggregato già rivela; reducer più n-1 siti espongono l'update del sito restante,
inevitabile per una somma. Le share viaggiano sigillate con chiavi derivate dal
KEM ibrido; il canale eredita i limiti di A (combiner sperimentale, nessun framing).
Ambiente di test: container o VM sullo stesso host condividono hardware e, per i
container, il kernel; i risultati sono costi di benchmark, non prova di isolamento.

Schema B (CKKS a soglia n-di-n, semi-honest): il reducer vede ciphertext, la somma
cifrata, le decifrature parziali e il risultato. Con chiave congiunta, un
sottoinsieme di siti più il reducer non decifra i singoli update. Limite aperto,
da portare a revisione esperta: le decifrature parziali non hanno noise flooding
(la modalità NOISE_FLOODING_DECRYPT di OpenFHE non ha mostrato rumore aggiunto nella
decifratura multiparty), e CKKS ha problemi noti di sicurezza IND-CPA-D. Nessun
dropout: serve la partecipazione di tutti i siti.
B-BGV: stesso modello semi-honest n-di-n, ma con noise flooding multiparty di OpenFHE
attivo (verificato dall'aumento del modulo del ciphertext) e somma esatta, senza il
limite CKKS delle decifrature parziali senza flooding. Resta da validare da un esperto
che il flooding sia sufficiente per le garanzie dichiarate (statisticalSecurity e
numAdversarialQueries non sono configurabili da Python per BGV multiparty).
Il baseline `InsecureSumStub` rivela gli aggiornamenti al reducer e non è uno schema. La loro policy
all_or_abort è una scelta funzionale provvisoria; i colleghi dovranno dichiarare
policy e garanzie dei core reali prima di un confronto scientifico.
