# Data Scarcity Experiment - Implementation Summary

## ✅ Progetto Completato

Ho implementato un **sistema completo** per il tuo caso di studio sulla scarcità di dati in Re3py. Il progetto è stato strutturato in modo modulare, professionale e reproducibile.

---

## 📋 Specifiche Implementate

### 1. **Modello e Validazione**
- ✅ **Modello**: Bagging con AGG-All (tutti gli aggregati attivi)
- ✅ **Validazione**: 10-fold Cross-Validation (mantiene struttura originale)
- ✅ **Motivazione**: Come da Tabella 2 e 5 del paper (Bagging più performante e stabile)

### 2. **Metriche**
- ✅ **Primaria**: Accuracy (per confronto baseline)
- ✅ **Secondarie**: F1 e AUC (tracciamento extra)
- ✅ **Per-fold**: Calcolo dettagliato per ogni fold
- ✅ **Aggregazione**: Media e deviazione standard su 10 fold

### 3. **Logica di Riduzione Dati (Vincolo Fondamentale)**
- ✅ **Incremental**: 10% → 20% → 50% → 70% → 90%
- ✅ **Deterministica**: Stesse istanze rimosse ad ogni step (cumulative)
- ✅ **Stratificata**: Mantenimento della distribuzione delle etichette
- ✅ **Riproducibile**: Seed fisso (2864), mappature salvate

### 4. **Dataset**
- ✅ **9 dataset**: BASKET, IMDB, MOVIE, STACK, UWCSE, YELP, WEBKB, CARC, MUTA
- ✅ **Dimensioni**: 95 - 24,959 istanze (ampia varietà)
- ✅ **Mapping**: Nome dataset paper → repository

---

## 🏗️ Architettura

### Struttura Modulo

```
scarcity_experiment/
├── data_scarcity_preprocessing.py   (Stage 1: 11 KB)
├── fold_mapper.py                   (Stage 2: 5.9 KB)
├── run_experiment.py                (Stage 3: 15 KB)
├── run_pipeline.py                  (Orchestrator: 8.6 KB)
├── quick_start.sh                   (Launcher: bash)
├── README.md                         (7.5 KB, documentazione completa)
├── config.yaml                       (Configurazione: 3.2 KB)
└── __init__.py                       (Package init)
```

### Pipeline a 4 Stadi

```
┌─────────────────────────────────────────────────────┐
│ Stage 1: Data Preprocessing                         │
│ - Carica target file                                │
│ - Applica riduzioni incrementali deterministiche    │
│ - Stratifica per etichetta                          │
│ - Salva mappature rimozioni                         │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│ Stage 2: Fold Mapping                               │
│ - Carica 10-fold CV originali                       │
│ - Filtra istanze per ogni riduzione                 │
│ - Crea nuovi fold file per ogni livello             │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│ Stage 3: Run Experiments                            │
│ - Istanzia Dataset ridotto                          │
│ - Entrena Bagging AGG-All per ogni fold             │
│ - Calcola Accuracy, F1, AUC                         │
│ - Aggrega risultati (mean ± std)                    │
└────────────────┬────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────┐
│ Stage 4: Results Aggregation                        │
│ - Combina CSV di tutti dataset                      │
│ - Salva combined_results.csv                        │
│ - Genera report finale                              │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Modalità di Esecuzione

### 1. **Modalità Rapida (Raccomandata)**
```bash
cd /Users/giacomo/Documents/re3py/scarcity_experiment
python3 run_pipeline.py --all
```
Esegue l'intero pipeline su tutti i 9 dataset con impostazioni di default.

### 2. **Test su Dataset Singolo**
```bash
python3 run_pipeline.py --dataset basket
```
Perfetto per verificare che tutto funzioni correttamente prima di lanciare l'esperimento completo.

### 3. **Con Seed Personalizzato**
```bash
python3 run_pipeline.py --all --seed 12345
```

### 4. **Metodo Alternative (Per Debug)**
```bash
# Stage 1: Preprocessing
python3 data_scarcity_preprocessing.py --all

# Stage 2: Fold mapping
python3 fold_mapper.py --all

# Stage 3: Experiments
python3 run_experiment.py --all

# Stage 4: Aggregation
# (Automatica in run_pipeline.py)
```

---

## 📊 Output Generati

### Per Ogni Dataset

```
results/{dataset}/
├── results_summary.json          # Metriche aggregate
├── results.csv                   # Formato tabulare
└── experiment.log                # Log dettagliato
```

### Output Globale

```
results/combined_results.csv       # Risultati tutti dataset
pipeline_*.log                     # Log completo pipeline
pipeline_summary_*.json            # Metadati esecuzione
```

### Struttura Dati Ridotti

```
data_reduced/{dataset}/
├── target_removed_10.txt         # Target con 10% rimossi
├── target_removed_20.txt
├── target_removed_50.txt
├── target_removed_70.txt
├── target_removed_90.txt
├── folds/
│   ├── folds_removed_10.txt      # Fold mappati per ogni riduzione
│   ├── folds_removed_20.txt
│   └── ...
└── metadata/
    ├── metadata_removed_10.json  # Mappatura istanze rimosse
    ├── metadata_removed_20.json
    ├── removal_mapping.json      # Mappatura cumulativa
    └── ...
```

---

## 🔍 Dettagli Implementativi Chiave

### 1. Riduzione Incrementale Deterministica

```python
# Esempio logica:
all_removed_cumulative = []  # Accumula rimossi

for percentage in [10, 20, 50, 70, 90]:
    current_target_count = len(data) * percentage / 100
    additional_remove = current_target_count - len(all_removed_cumulative)
    
    # Sample solo dalle istanze NON ancora rimosse
    remaining = [id for id in data if id not in all_removed_cumulative]
    new_removed = stratified_sample(remaining, additional_remove)
    
    all_removed_cumulative.extend(new_removed)  # CUMULATIVE!
    save_reduced_dataset(all_removed_cumulative, percentage)
```

**Garantisce**:
- ✅ 10% ⊂ 20% ⊂ 50% ⊂ 70% ⊂ 90% (subset property)
- ✅ Deterministica (stesso seed = stessi risultati)
- ✅ Stratificata (mantiene proporzioni etichette)

### 2. Cross-Validation Adattativa

Ogni riduzione:
1. Carica fold originali da `data/folds/{dataset}/folds1.txt`
2. Filtra istanze per mantenere solo quelle nel dataset ridotto
3. Genera nuovo fold file `data_reduced/{dataset}/folds/folds_removed_{X}.txt`
4. Esegue 10-fold CV su dataset ridotto

**Vantaggio**: Mantiene la struttura originale della CV mentre si adatta ai dati ridotti.

### 3. Gestione Metriche

Per ogni fold:
```
Ground Truth vs Predictions
├── Accuracy = (TP + TN) / Total
├── F1 = 2 * (Precision * Recall) / (Precision + Recall)
└── AUC = (TP + TN) / (TP + TN + FP + FN)

Aggregazione su 10 fold:
└── Mean ± Std per ogni metrica
```

---

## 🛠️ Configurazione

File principale: `config.yaml`

Parametri modificabili:
- `reduction_percentages`: [10, 20, 50, 70, 90]
- `random_seed`: 2864
- `n_folds`: 10
- `max_depth`: 5 (alberi Bagging)
- `n_estimators`: 10 (iterazioni Bagging)

---

## ⏱️ Tempi Stimati

| Fase | Tempo |
|------|-------|
| Preprocessing (9 dataset) | 2-5 minuti |
| Fold Mapping | 1 minuto |
| Experiments (dipende CPU) | 30-60 minuti |
| Aggregation | < 1 minuto |
| **TOTALE** | **~1-2 ore** |

---

## ✅ Checklist Reproducibilità

- ✅ **Seed fisso**: 2864 (predefinito, personalizzabile)
- ✅ **Determinismo**: Rimozioni cumulative e stratificate
- ✅ **Documentazione**: Metadata JSON per ogni step
- ✅ **Tracciabilità**: Log completi di ogni fase
- ✅ **Validazione**: 10-fold CV mantenuta
- ✅ **Archiviazione**: Istanze rimosse salvate

---

## 📝 Esempio Output

```
==================================================
DATA SCARCITY EXPERIMENT - FULL PIPELINE
==================================================
Datasets: basket, carcinogenesis, imdb_big, ...
Seed: 2864

[2026-01-21 16:25:30] [INFO] Starting Stage 1: Data Preprocessing
[2026-01-21 16:25:31] [INFO] Processing dataset: basket
✓ Created reduced dataset: 10% removed
  Original: 95 instances
  Kept: 85 instances
  Removed: 10 instances
...

[2026-01-21 16:30:45] [INFO] Starting Stage 3: Run Bagging AGG-All Experiments
[2026-01-21 16:31:00] [INFO] Running Bagging on 10% reduced data
--- Fold 1/10 ---
Train: 76 examples
Test: 9 examples
  Accuracy: 0.8889
  F1: 0.8571
  AUC: 0.9167

Results saved to: results/basket/results_summary.json
```

---

## 🎯 Prossimi Passi

### Per avviare l'esperimento:
```bash
cd /Users/giacomo/Documents/re3py/scarcity_experiment

# Test rapido (1 dataset)
python3 run_pipeline.py --dataset basket

# Esperimento completo (tutti 9 dataset)
python3 run_pipeline.py --all
```

### Per analizzare risultati:
```bash
# Visualizzare risultati combinati
cat results/combined_results.csv

# Leggere summary di un dataset
cat results/basket/results_summary.json
```

---

## 📚 Documentazione Completa

Vedi [README.md](README.md) per:
- Descrizione dettagliata di ogni stage
- Formato output (JSON, CSV)
- Troubleshooting
- Parametri configurabili
- Diagrammi pipeline

---

## ✨ Caratteristiche Aggiuntive

1. **Logging Integrato**: Ogni stage log su console e file
2. **Error Handling**: Gestione eccezioni granulare
3. **Modularità**: Ogni stage eseguibile indipendentemente
4. **Scalabilità**: Supporta aggiunta nuovi dataset facilmente
5. **Riproducibilità**: Metadata completi salvati

---

## 🤝 Supporto

Se hai domande o necessiti modifiche:
- Controlla README.md per documentazione completa
- Verifica pipeline_YYYYMMDD_HHMMSS.log per debug
- Consulta docstring nei script Python

---

**Pronto per iniziare l'esperimento! 🚀**
