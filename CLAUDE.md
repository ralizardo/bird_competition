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

Uses latitude/longitude coordinates to predict species as a baseline to understand geographic predictive power.

**Models:**
- Logistic Regression (multinomial, scaled features)
- Decision Tree (max_depth=20)
- XGBoost (n_estimators=100, max_depth=6)

**Key Features:**
- Adaptive stratified split for handling small classes:
  - 1 sample: train only
  - 2-4 samples: 50/50 split
  - 5+ samples: 80/20 split
- Metrics: accuracy, top-3/top-5 accuracy, log loss
- Outputs: `approaches/approach_1/outputs/` (performance_table.csv, plots)

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
