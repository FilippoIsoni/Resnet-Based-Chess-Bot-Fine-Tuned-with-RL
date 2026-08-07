# File golden

File di riferimento versionati, usati dai test di regressione dell'encoder (§2.5 del
piano, Gate 1 della pipeline).

Sono piccoli e **devono** stare in git: senza riferimento, i test di regressione non
hanno nulla con cui confrontare. Il `.gitignore` li esclude esplicitamente dalle regole
sui file di dati.

## Contenuto previsto (Stadio 1)

| File | Contenuto |
|---|---|
| `positions.json` | 20 posizioni note con il tensore 19x8x8 atteso |
| `moves.json` | mosse rappresentative con l'indice 0-4671 atteso |
| `mirror.json` | coppie (posizione, posizione specchiata) con indici mossa attesi |

## Regola d'oro

Un file golden si rigenera **solo** dopo aver capito perche e cambiato. Rigenerarlo
per far passare un test rosso e il modo piu efficace di cancellare l'unica protezione
contro le regressioni silenziose.

Quando si rigenera, il commit deve spiegare cosa e cambiato nell'encoder e perche il
nuovo output e quello giusto.

## Casi che `positions.json` deve coprire

Non vanno scelte 20 posizioni a caso: servono i casi limite che rompono gli encoder.

- posizione iniziale (bianco al tratto) e la stessa con nero al tratto
- tutti e quattro i diritti di arrocco attivi, e nessuno
- en passant disponibile su colonne diverse (inclusa la colonna a e la h)
- contatore delle 50 mosse a 0, a meta, e prossimo al limite
- posizione ripetuta due volte (piano delle ripetizioni)
- finale con pochi pezzi
- posizione con pedone su settima traversa (promozione imminente)
- scacco in corso
