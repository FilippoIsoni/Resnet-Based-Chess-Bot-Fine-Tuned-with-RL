# Diario di lavoro

Note su cose non ovvie imparate strada facendo. Non e un log di commit — quello lo fa
git. Qui vanno i "non me lo aspettavo" e i bug che hanno richiesto tempo per essere
capiti, perche sono esattamente quelli che si ripresentano.

Regola #6: se scopri un bug, il primo commit del fix contiene il test che lo avrebbe
intercettato. Annota qui **quanto tempo e costato trovarlo** — e la misura di quanto
valeva quel test.

---

### 2026-08-07 — Setup del progetto
Struttura, pipeline di check e ambiente definiti prima di scrivere codice di dominio.
La scelta e deliberata: in questo progetto i bug non crashano (encoder sbagliato, segno
invertito nel backup MCTS, leakage nello split), quindi l'infrastruttura di verifica
deve esistere prima delle cose da verificare.

Prossimo passo: Stadio 1 — encoder e codifica mosse, con la batteria di test §2.5 come
Gate 1.
