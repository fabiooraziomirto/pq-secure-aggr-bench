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
contro tale collusione è dichiarata. Non sono gestiti dropout o equivocation.
Il simulatore centralizzato vede tutti gli aggiornamenti e seed: è esclusivamente
un riferimento funzionale e non realizza la separazione di fiducia tra siti.
DP di record o sito, clipping e accounting sono decisioni aperte e non implementate.

## Fuori scope

Attestazione con catene classiche, side-channel, server malevolo e avvelenamento
a bassa norma sono fuori scope. Sono escluse anche compromissione degli endpoint
e verifica della correttezza del training. Attacchi di membership inference ed
estrazione saranno moduli separati dei colleghi, non eseguiti in questa base.
Nei log è ammesso solo `run_seed`, seed pubblico di riproducibilità. Non conservare
chiavi, seed crittografici, segreti condivisi, ciphertext o maschere nei log; nessun dato clinico nei test.
