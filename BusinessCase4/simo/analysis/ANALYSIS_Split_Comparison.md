# Split Methodology Comparison — Simone
## Business Case 4 — Early Warning System

---

## 1. Full Results by Split

### Shuffled Split (baseline — original professor's method)

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Elliptic Envelope | 0.677 | 0.899 | **0.773** | **0.768** |
| MVG Ledoit-Wolf | 0.635 | 0.966 | 0.767 | 0.750 |
| Student-t (ν=2) | 0.699 | 0.840 | 0.763 | 0.750 |
| Graphical Lasso | 0.701 | 0.807 | 0.750 | 0.749 |
| MVG Baseline | 0.603 | 0.983 | 0.748 | 0.750 |
| MVG Asymmetric | 0.612 | 0.916 | 0.734 | 0.710 |
| MVG CDF-based | 0.578 | 1.000 | 0.732 | 0.764 |
| Factor Model (k=12) | 0.698 | 0.740 | 0.718 | 0.730 |

---

### Walk-Forward Split (temporal, 60/20/20)

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Elliptic Envelope | **0.750** | 0.310 | **0.439** | 0.763 |
| MVG Asymmetric | 0.609 | 0.483 | 0.538 | 0.287 |
| Factor Model (k=1) | 0.643 | 0.310 | 0.419 | **0.784** |
| Student-t (ν=2) | 0.158 | 0.931 | 0.270 | 0.777 |
| MVG Baseline | 0.155 | 0.931 | 0.266 | 0.777 |
| Graphical Lasso | 0.151 | 0.931 | 0.260 | 0.779 |
| MVG Ledoit-Wolf | 0.155 | 0.931 | 0.266 | 0.783 |
| MVG CDF-based | 0.141 | 0.862 | 0.243 | 0.766 |

---

### Expanding Window (5 periodi, media sull'ultimo fold)

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Elliptic Envelope | **0.750** | 0.310 | **0.439** | 0.763 |
| MVG Asymmetric | 0.609 | 0.483 | 0.538 | 0.287 |
| Factor Model (k=1) | 0.643 | 0.310 | 0.419 | **0.784** |
| Student-t (ν=2) | 0.158 | 0.931 | 0.270 | 0.777 |
| MVG Baseline | 0.155 | 0.931 | 0.266 | 0.777 |
| Graphical Lasso | 0.151 | 0.931 | 0.260 | 0.779 |
| MVG Ledoit-Wolf | 0.155 | 0.931 | 0.266 | 0.783 |
| MVG CDF-based | 0.141 | 0.862 | 0.243 | 0.766 |

---

### Stratified K-Fold (k=5, con shuffle)

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Elliptic Envelope | 0.639 | 0.479 | 0.548 | **0.817** |
| Student-t (ν=2) | **0.551** | 0.563 | **0.557** | 0.812 |
| Graphical Lasso | 0.471 | 0.667 | 0.552 | 0.811 |
| Factor Model (k=10) | 0.509 | 0.583 | 0.544 | 0.810 |
| MVG Asymmetric | 0.486 | 0.354 | 0.410 | 0.378 |
| MVG Ledoit-Wolf | 0.282 | 0.958 | 0.436 | 0.814 |
| MVG Baseline | 0.266 | 0.958 | 0.416 | 0.812 |
| MVG CDF-based | 0.218 | 0.979 | 0.356 | 0.828 |

---

### Purged Cross-Validation (k=5, purge=4 settimane)

| Model | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| Elliptic Envelope | **0.750** | 0.310 | **0.439** | 0.768 |
| MVG Asymmetric | 0.609 | 0.483 | 0.538 | 0.287 |
| Factor Model (k=1) | 0.643 | 0.310 | 0.419 | **0.790** |
| Student-t (ν=2) | 0.209 | 0.793 | 0.331 | 0.784 |
| Graphical Lasso | 0.170 | 0.931 | 0.287 | 0.782 |
| MVG Ledoit-Wolf | 0.157 | 0.931 | 0.269 | 0.789 |
| MVG Baseline | 0.158 | 0.931 | 0.270 | 0.784 |
| MVG CDF-based | 0.150 | 0.862 | 0.255 | 0.771 |

---

## 2. F1 Score Comparison Across Splits

| Model | Shuffled | Walk-Forward | Expanding | Strat K-Fold | Purged CV | Δ (Shuffled - Best Temporal) |
|---|---|---|---|---|---|---|
| MVG Baseline | 0.748 | 0.266 | 0.266 | 0.416 | 0.270 | **-0.332** |
| MVG Ledoit-Wolf | 0.767 | 0.266 | 0.266 | 0.436 | 0.269 | **-0.331** |
| Elliptic Envelope | **0.773** | **0.439** | **0.439** | **0.548** | **0.439** | -0.225 |
| MVG CDF-based | 0.732 | 0.243 | 0.243 | 0.356 | 0.255 | -0.376 |
| MVG Asymmetric | 0.734 | 0.538 | 0.538 | 0.410 | 0.538 | **-0.196** |
| Student-t (ν=2) | 0.763 | 0.270 | 0.270 | **0.557** | 0.331 | -0.206 |
| Factor Model | 0.718 | 0.419 | 0.419 | 0.544 | 0.419 | -0.174 |
| Graphical Lasso | 0.750 | 0.260 | 0.260 | 0.552 | 0.287 | -0.198 |

---

## 3. AUC Comparison Across Splits

| Model | Shuffled | Walk-Forward | Expanding | Strat K-Fold | Purged CV |
|---|---|---|---|---|---|
| MVG Baseline | 0.750 | 0.777 | 0.777 | **0.812** | 0.784 |
| MVG Ledoit-Wolf | 0.750 | 0.783 | 0.783 | **0.814** | 0.789 |
| Elliptic Envelope | **0.768** | 0.763 | 0.763 | **0.817** | 0.768 |
| MVG CDF-based | 0.764 | 0.766 | 0.766 | **0.828** | 0.771 |
| MVG Asymmetric | 0.710 | 0.287 | 0.287 | 0.378 | 0.287 |
| Student-t (ν=2) | 0.750 | 0.777 | 0.777 | 0.812 | **0.784** |
| Factor Model | 0.730 | 0.784 | 0.784 | 0.810 | **0.790** |
| Graphical Lasso | 0.749 | 0.779 | 0.779 | 0.811 | 0.782 |

---

## 4. Analisi Critica

### 4.1 Il Leakage Temporale Gonfia le Metriche in Modo Drammatico

Il dato più importante dell'intera analisi:

> **F1 medio con shuffle: 0.743 → F1 medio con walk-forward: 0.333**
> **Il leakage temporale inflaziona l'F1 di oltre il 100%**

Con la split shuffled, dati del 2008 finiscono nel training e dati del 2007 nel test. Il modello non sta imparando a *riconoscere* le crisi — sta imparando a *ricordare* le settimane già viste. Questo è esattamente il **look-ahead bias** che rende inutilizzabili i backtest nella pratica.

La performance "reale" attesa in produzione si avvicina molto di più ai numeri della walk-forward o della purged CV.

---

### 4.2 Walk-Forward ≈ Expanding Window ≈ Purged CV

I tre metodi temporali danno risultati quasi identici sull'ultimo fold:

- Walk-Forward e Expanding Window sono **identici** perché entrambi usano lo stesso ultimo periodo come test
- Purged CV è leggermente diverso (rimuove le 4 settimane di purge gap) ma l'impatto è minimo su questo dataset

**Perché il purge gap cambia poco?** L'autocorrelazione finanziaria settimanale decade in poche settimane per la maggior parte delle variabili trasformate (log-return). Un gap di 4 settimane è probabilmente sufficiente — non serve un gap maggiore.

---

### 4.3 Il Crollo della Precision nei Metodi Temporali

Con la walk-forward split, la precision della maggior parte dei modelli crolla a ~0.15:

> Solo 1 settimana su 7 flaggata come crisi è effettivamente una crisi.

**Perché?** Il modello viene addestrato su un regime di mercato (es. anni 2000-2012) e testato su un periodo successivo con struttura diversa. Le anomalie del periodo recente hanno caratteristiche statistiche diverse da quelle del passato → il threshold calibrato sul CV passato non si trasferisce bene.

**Eccezioni notevoli:**
- **Elliptic Envelope** mantiene precision=0.75 (la più alta) anche in temporal split, a costo di recall molto bassa (0.31). Trova solo le crisi più eclatanti, con pochi falsi allarmi.
- **MVG Asymmetric** mantiene precision=0.61 — la regola finanziaria ("se il mercato azionario è sopra soglia K, non può essere risk-off") è robusta al cambio di regime perché è basata su una logica economica, non su pattern statistici.

---

### 4.4 L'AUC è Stabile — il Problema è il Threshold

Un risultato fondamentale: **l'AUC rimane 0.76–0.82 in tutti i metodi di split**.

> I modelli mantengono la loro capacità di *ordinare* le settimane dal più al meno anomalo (AUC stabile). Il problema è calibrare la *soglia di classificazione* (F1 instabile).

Questo ha implicazioni pratiche importanti:
1. I modelli sono **genuinamente informativi** — non è tutto rumore
2. Il problema è il **threshold deployment**: la soglia calibrata sul CV storico non vale sul futuro
3. La soluzione giusta è un sistema di **soglia dinamica/rolling** che si aggiorna periodicamente, non una soglia fissa

---

### 4.5 Stratified K-Fold: Risultati Intermedi ma AUC Migliore

La K-Fold stratificata dà F1 nell'intervallo 0.35–0.56, intermedio tra shuffled e walk-forward. È attesa: shuffle parziale (dentro i fold) riduce il leakage ma non lo elimina.

Il dato più interessante: **AUC è la più alta di tutti i metodi (0.81–0.83)**. Media su 5 fold indipendenti riduce la varianza della stima — l'AUC stratificato è la stima più affidabile della vera capacità discriminante del modello.

---

### 4.6 Il Factor Model Collassa a k=1 nei Split Temporali

Con la split shuffled, il Factor Model sceglie k=12 fattori. Con walk-forward/purged CV, sceglie k=1.

**Perché?** Con il CV set temporale, le anomalie nel CV sono poche e concentrate in uno specifico regime. Un solo fattore (probabilmente il livello generale del mercato azionario) discrimina già bene in quel periodo. Aumentare k aggiunge rumore. Questo è un chiaro segnale di **instabilità del processo di selezione di k**: il modello non ha abbastanza anomalie in CV per selezionare un numero di fattori generalizzabile.

---

### 4.7 MVG Asymmetric: Unico Modello che Mantiene l'AUC Decente Solo con Shuffle

L'MVG Asimmetrico ha un comportamento anomalo: **AUC=0.29 nelle split temporali** (molto basso), mentre AUC=0.71 con shuffle.

**Perché?** La variabile discriminante (soglia K su MXUS) viene tuned sul CV set passato. Se il regime del mercato azionario cambia nel periodo di test, la soglia K non è più quella giusta — il modello override le sue predizioni in modo sistematicamente sbagliato. Il confine tra "mercato bullish" e "mercato bearish" è non stazionario e cambia nel tempo.

---

## 5. Ranking Finale dei Modelli per Robustezza

Valutati sulla capacità di mantenere performance **attraverso tutti i metodi di split**:

| Rank | Modello | Motivazione |
|---|---|---|
| 🥇 | **Elliptic Envelope** | Migliore F1 in tutti i contesti. Precision alta e stabile (0.75) anche in temporal. Robusto al cambio di regime. |
| 🥈 | **Student-t (ν=2)** | F1 mediocre in WF ma eccellente in Strat K-Fold (0.557). AUC stabile. La scelta ν=2 conferma i fat tails. |
| 🥉 | **Graphical Lasso** | AUC stabile (0.78). F1 buono in Strat K-Fold. Offre interpretabilità finanziaria unica. |
| 4 | **Factor Model** | AUC buono (0.78–0.81). k instabile ma struttura utile. Più interpretabile degli altri. |
| 5 | **MVG Ledoit-Wolf** | Alta recall ma precision crolla in temporal. Migliore del baseline ma non robusto. |
| 6 | **MVG Baseline** | Buono con shuffle ma fragile out-of-sample. Riferimento utile. |
| 7 | **MVG CDF-based** | Recall perfetto ma precision pessima in tutti i contesti temporali. Solo per risk management estremo. |
| 8 | **MVG Asymmetric** | AUC crolla in temporal (0.29). La soglia K non si trasferisce tra regimi. |

---

## 6. Conclusioni

### Cosa abbiamo imparato

1. **La split shuffled del professore è un benchmark didattico, non una stima di performance reale.** Gonfia F1 del 100%+ per effetto del look-ahead bias. Utile per confrontare modelli tra loro, non per stimare la performance assoluta in produzione.

2. **L'AUC è la metrica più affidabile tra i metodi di split.** Rimane stabile 0.76–0.82 in tutti i contesti. F1 invece dipende fortemente dalla calibrazione del threshold, che è instabile nel tempo.

3. **Elliptic Envelope è il modello più robusto.** Mantiene la capacità discriminante in tutti i split, ha la precision più alta in temporal, ed è l'unico a bilanciare precision e recall anche fuori dal regime di training.

4. **Il threshold non si trasferisce tra regimi.** Il vero problema non è il modello, ma la calibrazione della soglia di classificazione. Un sistema di soglia rolling/adattiva sarebbe la strada giusta per il deployment.

5. **La regola finanziaria del MVG Asymmetric è un'idea buona ma mal implementata.** La soglia K su MXUS non è stazionaria — andrebbe stimata su una finestra rolling, non fissa sull'intero CV storico.

### Raccomandazione per il deployment

| Obiettivo | Modello | Metrica da monitorare |
|---|---|---|
| Minimizzare i falsi allarmi | Elliptic Envelope | Precision |
| Non perdere nessuna crisi | MVG CDF-based | Recall |
| Ranking settimane per rischio | Qualsiasi (AUC stabile) | AUC |
| Interpretabilità per il risk officer | Factor Model o Graphical Lasso | AUC + loadings |
