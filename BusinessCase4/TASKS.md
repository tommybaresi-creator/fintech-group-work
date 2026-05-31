# Business Case 4 — Task Division

## Shared Pipeline (tutti e 4 — da fare prima)

- [ ] Caricare `Dataset4_EWS.xlsx` e costruire `X_df` e `y`
- [ ] Applicare la stationarity transform (log-diff per indici/valute, first-diff per tassi, as-is per ECSURPUS)
- [ ] Shuffle + split: `X_train` (80% normal), `X_cv` (10% normal + 50% anomalie), `X_test` (10% normal + 50% anomalie)
- [ ] `StandardScaler` (fit su train, transform su cv e test)
- [ ] Funzione `evaluate_model()` → restituisce `{Precision, Recall, F1, AUC}`
- [ ] `results_df` con colonne `['Model', 'Precision', 'Recall', 'F1', 'AUC']` — ogni persona appende i propri risultati

---

## Persona 1 — MVG potenziato + Contesto Finanziario

### MVG Baseline (reference)
- [ ] Confermare che il MVG baseline del notebook gira correttamente
- [ ] Appendere risultati a `results_df`: `MVG Baseline`

### MVG + Ledoit-Wolf
- [ ] Sostituire la covarianza standard con `sklearn.covariance.LedoitWolf`
- [ ] Tuning soglia ε su CV (stessa logica baseline — massimizzare F1)
- [ ] Valutare su test set: Precision, Recall, F1, AUC
- [ ] Appendere a `results_df`: `MVG Ledoit-Wolf`

### Elliptic Envelope
- [ ] Implementare `sklearn.covariance.EllipticEnvelope`
- [ ] Tuning del parametro `contamination` su CV
- [ ] Valutare su test set: Precision, Recall, F1, AUC
- [ ] Appendere a `results_df`: `Elliptic Envelope`

### MVG con CDF invece di PDF
- [ ] Sostituire il scoring con CDF: `Prob(x₁ < p, x₂ < p, ...)` usando la CDF della gaussiana multivariata
- [ ] Tuning soglia su CV
- [ ] Valutare su test set: Precision, Recall, F1, AUC
- [ ] Appendere a `results_df`: `MVG CDF-based`

### MVG Asimmetrico con Variabile Discriminante
- [ ] Scegliere la variabile discriminante (es. MXUS o VIX log-return)
- [ ] Implementare la logica: se discriminante > soglia K → non risk-off (override)
- [ ] Tuning sia di ε che di K su CV (grid search 2D)
- [ ] Valutare su test set: Precision, Recall, F1, AUC
- [ ] Appendere a `results_df`: `MVG Asymmetric`

### Visualizzazioni
- [ ] t-SNE sul test set (colorato per TP/FP/FN/TN)
- [ ] PACMAP sul test set
- [ ] Spectral Embedding sul test set
- [ ] Esperimenti su dati sintetici: variare parametri del DGP (distanza tra normali e anomalie, numero di dimensioni) e osservare come cambiano le performance

### Explainability
- [ ] Feature importance tramite permutation sul MVG (zero-out una feature alla volta, misurare cambio di log-likelihood)
- [ ] Bar chart delle feature più importanti

---

## Persona 2 — Modelli Supervisionati + Classi Sbilanciate

> Training su `X_train + X_cv` (con label), test su `X_test`

### Random Forest Baseline (reference)
- [ ] Confermare che RF del notebook gira correttamente
- [ ] Appendere risultati a `results_df`: `Random Forest Baseline`

### SVM Baseline (reference)
- [ ] Confermare che SVM del notebook gira correttamente
- [ ] Appendere risultati a `results_df`: `SVM Baseline`

### Random Forest + SMOTE
- [ ] Installare `imbalanced-learn`: `pip install imbalanced-learn`
- [ ] Applicare `SMOTE` sul training set prima del fit
- [ ] Riaddestrare RF su dati oversamplati
- [ ] Valutare su test set: Precision, Recall, F1, AUC
- [ ] Appendere a `results_df`: `Random Forest + SMOTE`

### XGBoost con scale_pos_weight
- [ ] Installare `xgboost`: `pip install xgboost`
- [ ] Calcolare `scale_pos_weight = n_normal / n_anomaly` sul training set
- [ ] Addestrare `XGBClassifier` con questo peso
- [ ] Valutare su test set: Precision, Recall, F1, AUC
- [ ] Appendere a `results_df`: `XGBoost`

### XGBoost + SMOTE
- [ ] Applicare SMOTE prima del fit XGBoost
- [ ] Valutare su test set: Precision, Recall, F1, AUC
- [ ] Appendere a `results_df`: `XGBoost + SMOTE`

### Best Supervised Model con Optuna
- [ ] Installare `optuna`: `pip install optuna`
- [ ] Definire la `objective` function che massimizza F1 su CV
- [ ] Ottimizzare iperparametri del modello con F1 migliore (RF o XGBoost)
- [ ] Riaddestrare con i best params e valutare su test set
- [ ] Appendere a `results_df`: `Best Supervised (Optuna)`

### Explainability
- [ ] Calcolare SHAP values sul miglior modello supervisionato (`pip install shap`)
- [ ] Summary plot SHAP (beeswarm o bar)
- [ ] Identificare le 5 feature più importanti per la detection delle anomalie

---

## Persona 3 — Modelli Non Supervisionati Classici + COPOD

> Training su `X_train` (solo normali), soglia tuned su `X_cv`

### Isolation Forest (reference)
- [ ] Confermare che Isolation Forest del notebook gira correttamente
- [ ] Appendere risultati a `results_df`: `Isolation Forest`

### One-Class SVM (reference)
- [ ] Confermare che One-Class SVM del notebook gira correttamente
- [ ] Appendere risultati a `results_df`: `One-Class SVM`

### LOF (reference)
- [ ] Confermare che LOF del notebook gira correttamente
- [ ] Appendere risultati a `results_df`: `LOF`

### GMM 2 componenti (cella incompleta nel notebook)
- [ ] Implementare `GaussianMixture(n_components=2)` da sklearn
- [ ] Score = log-likelihood di ogni punto sotto la GMM
- [ ] Tuning soglia su CV (contamination rate)
- [ ] Valutare su test set: Precision, Recall, F1, AUC
- [ ] Appendere a `results_df`: `GMM 2 components`

### COPOD (PyOD)
- [ ] Installare PyOD: `pip install pyod`
- [ ] Implementare `from pyod.models.copod import COPOD`
- [ ] Tuning del parametro `contamination` su CV
- [ ] Valutare su test set: Precision, Recall, F1, AUC
- [ ] Appendere a `results_df`: `COPOD`

### Isolation Mondrian Forest
- [ ] Installare: `pip install mondrian-forests` (o implementazione alternativa)
- [ ] Addestrare su `X_train`
- [ ] Tuning contamination su CV
- [ ] Valutare su test set: Precision, Recall, F1, AUC
- [ ] Appendere a `results_df`: `Isolation Mondrian Forest`

### Hyperparameter Tuning con Optuna
- [ ] Installare `optuna`: `pip install optuna`
- [ ] Ottimizzare il modello non supervisionato con F1 migliore (es. numero di stimatori, n_neighbors, contamination)
- [ ] Appendere a `results_df`: `Best Unsupervised (Optuna)`

### Tabella riassuntiva modelli non supervisionati
- [ ] Produrre una tabella markdown con pro/contro di ogni metodo (Decision Boundary, Labels needed, Contamination needed)

---

## Persona 4 — Deep Learning + Ensemble + Incertezza

> Training su `X_train` (solo normali), soglia tuned su `X_cv`

### Autoencoder (reference — completare cella esistente)
- [ ] Verificare e completare il codice dell'Autoencoder nel notebook (PyTorch)
- [ ] Training con early stopping (patience=20, max 200 epochs)
- [ ] Tuning soglia su CV (contamination rate)
- [ ] Valutare su test set: Precision, Recall, F1, AUC
- [ ] Plot reconstruction error (normal vs anomalie)
- [ ] Appendere a `results_df`: `Autoencoder`

### Variational Autoencoder (VAE)
- [ ] Implementare VAE in PyTorch (encoder → μ, σ → reparametrization → decoder)
- [ ] Loss = MSE reconstruction + KL divergence
- [ ] Training con early stopping
- [ ] Anomaly score = reconstruction error (o ELBO)
- [ ] Tuning soglia su CV
- [ ] Valutare su test set: Precision, Recall, F1, AUC
- [ ] Appendere a `results_df`: `VAE`

### LSTM Autoencoder
- [ ] Reshaping dati in sequenze temporali (es. finestra di 4 settimane)
- [ ] Implementare LSTM encoder-decoder in PyTorch
- [ ] Training su sequenze normali con early stopping
- [ ] Anomaly score = reconstruction error sulla sequenza
- [ ] Tuning soglia su CV
- [ ] Valutare su test set: Precision, Recall, F1, AUC
- [ ] Appendere a `results_df`: `LSTM Autoencoder`

### Ensemble (AE + LOF + Isolation Forest)
- [ ] Raccogliere gli anomaly score (probabilità o errore normalizzato) dai 3 modelli
- [ ] Combinare con media pesata (o voto a maggioranza)
- [ ] Tuning soglia ensemble su CV
- [ ] Valutare su test set: Precision, Recall, F1, AUC
- [ ] Appendere a `results_df`: `Ensemble (AE + LOF + IF)`

### Uncertainty via Varianza dell'Ensemble
- [ ] Calcolare la varianza degli score tra i modelli dell'ensemble per ogni punto
- [ ] Visualizzare: alta varianza = punto ambiguo (borderline)
- [ ] Plot varianza vs errore di classificazione (FP/FN tendono ad alta varianza?)

### Explainability Deep Learning
- [ ] Completare feature importance per permutation (già parzialmente nel notebook)
- [ ] Gradient-based importance con Captum (`pip install captum`) sul miglior modello

### Merge finale (persona 4 coordina)
- [ ] Raccogliere i `results_df` di tutte e 4 le persone
- [ ] Merge in un unico DataFrame
- [ ] Bar chart di confronto F1 per tutti i modelli
- [ ] ROC curve overlay (tutte le curve in un unico plot)
- [ ] Tabella finale ordinata per F1 decrescente

---

## Output comune atteso da tutti

```
results_df columns: ['Model', 'Precision', 'Recall', 'F1', 'AUC']
```

| Persona | Modelli da consegnare |
|---|---|
| P1 | MVG Baseline, MVG Ledoit-Wolf, Elliptic Envelope, MVG CDF-based, MVG Asymmetric |
| P2 | RF Baseline, SVM Baseline, RF+SMOTE, XGBoost, XGBoost+SMOTE, Best Supervised (Optuna) |
| P3 | Isolation Forest, One-Class SVM, LOF, GMM 2 comp., COPOD, Mondrian Forest, Best Unsupervised (Optuna) |
| P4 | Autoencoder, VAE, LSTM Autoencoder, Ensemble, + merge finale |
