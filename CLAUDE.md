# BirdCLEF+ 2026 Competition

## Competition Overview

- **Platform:** Kaggle
- **Deadline:** June 3, 2026
- **Prize:** $50,000 USD
- **Task:** Multi-label bioacoustic species classification in soundscape recordings
- **Classes:** 234 species (birds, frogs, insects, reptiles, mammals)
- **Location:** Pantanal wetlands, Mato Grosso do Sul, Brazil (coordinates: -16.5 to -21.6 lat, -55.9 to -57.6 lon)

## Data Structure

### train_audio/ (10 GB)
- Short audio clips of individual species
- Source: iNaturalist and Xeno-Canto (crowdsourced)
- Organized by species ID folders (e.g., `1161364/iNat1216197.ogg`)
- Metadata in `train.csv`: species labels, coordinates, author, license, rating

### train_soundscapes/ (5 GB)
- Long continuous field recordings from passive acoustic monitors
- Realistic evaluation scenario with multiple overlapping species
- Labels in `train_soundscapes_labels.csv` with 5-second windows
- Multi-label format: `filename,start,end,primary_label` (e.g., `22961;23158;24321;517063;65380`)

### Key Files
- `train.csv` - Training metadata (6.5 MB)
- `train_soundscapes_labels.csv` - Soundscape annotations (132 KB)
- `taxonomy.csv` - Bird species taxonomy (16 KB)
- `sample_submission.csv` - Submission format
- `recording_location.txt` - Pantanal location info

## Target Species (234 classes)

Not just birds! The taxonomy includes:
- **Aves (birds):** Most species (e.g., `ashgre1`, `yebcar`, `greani1`)
- **Amphibia (frogs):** Whistling Grass Frog, Waxy Monkey Tree Frog, etc.
- **Reptilia:** Southern Spectacled Caiman
- **Mammalia:** Feral Horse
- **Insecta:** Guyalna cuta (cicada)

## Input/Output Structure

### Input
- **Whole soundscape recordings** (variable length, e.g., 60+ seconds)
- Files provided as `.ogg` audio

### Output (Submission)
- **One prediction row per 5-second window**
- **row_id format:** `{filename}_{end_second}`
- **234 probability columns** (one per species, values 0-1)

Example:
```
row_id                                    | species_1 | species_2 | ... | species_234
BC2026_Test_0001_S05_20250227_010002_5    | 0.95      | 0.02      | ... | 0.01   (0-5s)
BC2026_Test_0001_S05_20250227_010002_10   | 0.10      | 0.85      | ... | 0.03   (5-10s)
BC2026_Test_0001_S05_20250227_010002_15   | 0.05      | 0.10      | ... | 0.80   (10-15s)
```

### Inference Pipeline
```
Soundscape (60s) → Segment into 5s chunks → Model → 12 prediction rows
                   [0-5, 5-10, 10-15, ...]       [234 probabilities each]
```

## Challenge

Bridge the domain gap between:
- **Clean training clips** (single species, close recording)
- **Noisy soundscapes** (multiple species, background noise, varying distances)

## Modeling Approaches

### Approach 1: Geolocation Baseline (Implemented)
**Location:** `approaches/approach_1/geolocation_baseline.py`

**Note:** First approach WITHOUT feature engineering. Uses only raw latitude/longitude coordinates to predict species as a baseline to understand geographic predictive power.

**Models:**
- Logistic Regression (multinomial, scaled features)
- Decision Tree (max_depth=20)
- XGBoost (n_estimators=100, max_depth=6)

**Key Features:**
- Adaptive stratified split for handling small classes:
  - 1 sample: duplicated in both train/test
  - 2-4 samples: 50/50 split
  - 5+ samples: 80/20 split
- Metrics: accuracy, top-k accuracy, log loss, balanced accuracy, F1 (macro/weighted), precision, recall, ROC-AUC
- Outputs: `approaches/approach_1/outputs/` (performance_table.csv, plots)

**Results (206 classes, 35,549 samples):**

| Model | Accuracy | Balanced Acc | F1 Macro | ROC-AUC |
|-------|----------|--------------|----------|---------|
| Logistic Regression | 4.9% | 2.0% | 1.2% | 0.75 |
| Decision Tree | **15.0%** | **12.9%** | **11.5%** | 0.73 |
| XGBoost | 14.3% | 12.5% | 10.5% | **0.87** |

**Comparison vs Random Classifier (206 classes):**

| Metric | Random | Best Model | Improvement |
|--------|--------|------------|-------------|
| Accuracy | 0.49% (1/206) | 15.0% (DT) | **31x better** |
| Balanced Accuracy | 0.49% | 12.9% (DT) | **26x better** |
| F1 Macro | ~0.49% | 11.5% (DT) | **24x better** |
| Top-3 Accuracy | 1.46% (3/206) | 27.9% (DT) | **19x better** |
| Top-5 Accuracy | 2.43% (5/206) | 35.9% (XGB) | **15x better** |
| ROC-AUC | 0.50 | 0.87 (XGB) | **+0.37 points** |

**Key Findings:**
1. **Geolocation is significantly better than random** - 15-31x improvement on classification metrics
2. **Decision Tree** wins on classification metrics; **XGBoost** wins on probability calibration (ROC-AUC: 0.87)
3. **Logistic Regression struggles** with non-linear geographic boundaries between species
4. **ROC-AUC of 0.87** confirms geographic location contains real signal for species distribution patterns

**Recommendations:**
- Use geolocation as a supplementary feature, not standalone
- XGBoost probabilities can serve as priors to combine with audio models
- Audio features are essential for accurate species classification

### Approach 2: Direct Multi-Label Classification (Recommended)
1. Convert 5-second audio → mel spectrograms
2. Train CNN/Transformer for multi-label prediction
3. Use sigmoid outputs (not softmax) — probabilities are independent, don't sum to 1
4. Loss function: Binary Cross-Entropy (BCE)

### Approach 3: Segment → Classify
1. Detect/extract individual bird calls
2. Classify each segment
3. Aggregate predictions per window

### Approach 4: Pretrained Audio Models
- BirdNET (bird-specific)
- PANNs (AudioSet pretrained)
- AST/HTS-AT (Audio transformers)

## Winning Strategies (from past competitions)

- **Architecture:** EfficientNet-B0 to B3, ConvNeXt on mel spectrograms
- **Training:** Train on short clips, validate on soundscapes
- **Augmentation:** Mixup, time/frequency masking (SpecAugment), noise injection
- **Inference:** Test-time augmentation, ensemble multiple models

## Project Structure

```
BirdCLEF_2026/
├── approaches/           # Modeling approaches
│   └── approach_1/       # Geolocation baseline
│       ├── geolocation_baseline.py
│       └── outputs/      # Results and plots
├── configs/              # Configuration files
├── data/
│   ├── raw/              # Competition data
│   ├── processed/        # Cleaned/transformed data
│   └── external/         # External datasets
├── logs/                 # Training logs
├── models/               # Saved model weights
├── notebooks/            # Jupyter notebooks
├── reports/figures/      # Generated plots
├── src/
│   ├── data/             # Data loading/processing
│   ├── features/         # Feature engineering (spectrograms)
│   ├── models/           # Model architectures
│   └── visualization/    # Plotting utilities
├── .gitignore
├── requirements.txt
└── CLAUDE.md
```

## Next Steps

1. Exploratory Data Analysis (EDA)
2. Audio preprocessing pipeline (mel spectrograms)
3. Baseline model training
4. Augmentation experiments
5. Validation strategy (soundscape-based)
6. Model ensembling and submission
