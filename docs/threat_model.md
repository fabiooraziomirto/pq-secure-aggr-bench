# Modello di minaccia

Tre siti A, B e C addestrano adapter LoRA dello stesso LLM. Dati clinici grezzi,
record e credenziali restano nei rispettivi siti. Anche gli aggiornamenti e gli
aggregati possono contenere informazioni sensibili: l'aggregazione sicura non
costituisce da sola una garanzia contro membership inference o estrazione.

## Livello esterno: harvest now, decrypt later

Un avversario registra oggi il traffico e dispone in futuro di capacità
quantistiche. Il canale previsto combina X25519 e ML-KEM-768. L'autenticazione,
il transcript binding, il profilo del combiner e il ciclo di rotazione devono
ancora essere revisionati e verificati. HybridKEM implementa un combiner
sperimentale con transcript binding, non un canale autenticato; HybridChannel
resta uno stub. TODO: revisione da parte di un esperto.
Non si rivendica sicurezza post-quantum dell'implementazione corrente.
La scelta delle firme/autenticazione e delle ancore di fiducia resta TODO.
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
