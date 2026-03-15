"""
Approach 1: Geolocation-based Multi-class Classification

Uses latitude and longitude to predict species (primary_label).
Models: Logistic Regression, Decision Tree, XGBoost

This serves as a baseline to understand how much predictive power
geographic location alone provides for species identification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    log_loss,
    classification_report,
    confusion_matrix
)
import warnings
import os
import sys
warnings.filterwarnings('ignore')

# Detect environment and set paths
def get_project_paths():
    """Get project paths, handling both local and Colab environments."""
    # Check if running in Google Colab
    IN_COLAB = 'google.colab' in sys.modules

    if IN_COLAB:
        # Colab: look for data in common locations
        possible_roots = [
            Path('/content/bird_competition'),
            Path('/content/BirdCLEF_2026'),
            Path('/content'),
        ]
        for root in possible_roots:
            if (root / 'data' / 'raw' / 'train.csv').exists():
                return root, root / 'data' / 'raw' / 'train.csv'

        # Data not found - provide helpful message
        print("="*60)
        print("DATA NOT FOUND - Please download the competition data first:")
        print("="*60)
        print("""
# Run these commands in a Colab cell:

!pip install kaggle
from google.colab import files
files.upload()  # Upload your kaggle.json

!mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle competitions download -c birdclef-2026 -p /content/bird_competition/data/raw/
!unzip -q /content/bird_competition/data/raw/birdclef-2026.zip -d /content/bird_competition/data/raw/
        """)
        sys.exit(1)
    else:
        # Local environment
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "data" / "raw" / "train.csv"
        return project_root, data_path

PROJECT_ROOT, DATA_PATH = get_project_paths()
OUTPUT_DIR = Path(__file__).parent / "outputs" if '__file__' in dir() else Path('/content/bird_competition/approaches/approach_1/outputs')
OUTPUT_DIR.mkdir(exist_ok=True)

def load_and_prepare_data():
    """Load train.csv and prepare features/target."""
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print(f"Total samples: {len(df)}")
    print(f"Unique species (primary_label): {df['primary_label'].nunique()}")

    # Check for missing values in lat/lon
    missing_coords = df[['latitude', 'longitude']].isnull().sum()
    print(f"Missing coordinates:\n{missing_coords}")

    # Drop rows with missing coordinates
    df_clean = df.dropna(subset=['latitude', 'longitude'])
    print(f"Samples after dropping missing coords: {len(df_clean)}")

    # Show class distribution summary
    class_counts = df_clean['primary_label'].value_counts()
    print(f"\nClass distribution summary:")
    print(f"  Min samples per class: {class_counts.min()}")
    print(f"  Max samples per class: {class_counts.max()}")
    print(f"  Mean samples per class: {class_counts.mean():.1f}")
    print(f"  Median samples per class: {class_counts.median():.1f}")
    print(f"  Classes with 1 sample: {(class_counts == 1).sum()}")
    print(f"  Classes with 2-4 samples: {((class_counts >= 2) & (class_counts <= 4)).sum()}")
    print(f"  Classes with 5+ samples: {(class_counts >= 5).sum()}")

    # Features and target
    X = df_clean[['latitude', 'longitude']].values
    y = df_clean['primary_label'].values

    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"\nNumber of classes: {len(label_encoder.classes_)}")

    return X, y_encoded, label_encoder, df_clean


def adaptive_stratified_split(X, y, test_size=0.2, random_state=42):
    """
    Stratified split with adaptive strategy for small classes.

    Strategy:
    - Classes with 1 sample: duplicated in both train and test sets
    - Classes with 2-4 samples: 50/50 split (at least 1 in each set)
    - Classes with 5+ samples: standard test_size split (e.g., 80/20)

    Returns:
        X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)

    # Get class counts
    unique_classes, class_counts = np.unique(y, return_counts=True)
    class_count_dict = dict(zip(unique_classes, class_counts))

    train_indices = []
    test_indices = []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        n_samples = len(cls_indices)

        if n_samples == 1:
            # Only 1 sample: put in BOTH training and test sets
            train_indices.extend(cls_indices)
            test_indices.extend(cls_indices)
        elif n_samples <= 4:
            # 2-4 samples: 50/50 split
            split_point = n_samples // 2
            train_indices.extend(cls_indices[:split_point])
            test_indices.extend(cls_indices[split_point:])
        else:
            # 5+ samples: standard split
            n_test = max(1, int(n_samples * test_size))
            train_indices.extend(cls_indices[:-n_test])
            test_indices.extend(cls_indices[-n_test:])

    # Convert to arrays
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    # Shuffle indices
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]

    # Report split statistics
    print(f"\nAdaptive Stratified Split:")
    print(f"  Classes with 1 sample (in both sets): {sum(1 for c in class_counts if c == 1)}")
    print(f"  Classes with 2-4 samples (50/50 split): {sum(1 for c in class_counts if 2 <= c <= 4)}")
    print(f"  Classes with 5+ samples ({int((1-test_size)*100)}/{int(test_size*100)} split): {sum(1 for c in class_counts if c >= 5)}")

    # Verification
    train_classes = set(np.unique(y_train))
    test_classes = set(np.unique(y_test))
    all_classes = set(unique_classes)

    classes_only_in_train = train_classes - test_classes
    classes_only_in_test = test_classes - train_classes
    classes_in_both = train_classes & test_classes

    print(f"\nSplit Verification:")
    print(f"  Total classes: {len(all_classes)}")
    print(f"  Classes in train: {len(train_classes)}")
    print(f"  Classes in test: {len(test_classes)}")
    print(f"  Classes in both train & test: {len(classes_in_both)}")
    print(f"  Classes only in train: {len(classes_only_in_train)}")
    print(f"  Classes only in test: {len(classes_only_in_test)}")

    if classes_only_in_test:
        print(f"  WARNING: {len(classes_only_in_test)} classes appear only in test set!")

    # Verify all original classes are accounted for
    missing_classes = all_classes - (train_classes | test_classes)
    if missing_classes:
        print(f"  ERROR: {len(missing_classes)} classes missing from both sets!")
    else:
        print(f"  ✓ All classes accounted for in train/test split")

    # Verify sample counts (accounting for duplicated 1-sample classes)
    n_duplicated = sum(1 for c in class_counts if c == 1)
    print(f"\nSample counts:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    print(f"  Duplicated samples (1-sample classes): {n_duplicated}")
    print(f"  Total: {len(X_train) + len(X_test)} (original + duplicates: {len(X) + n_duplicated})")

    assert len(X_train) + len(X_test) == len(X) + n_duplicated, "Sample count mismatch!"

    return X_train, X_test, y_train, y_test

def train_and_evaluate_models(X_train, X_test, y_train, y_test, label_encoder):
    """Train all three models and return performance metrics."""

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_classes = len(label_encoder.classes_)
    results = {}
    models = {}

    # 1. Logistic Regression
    print("\n" + "="*50)
    print("Training Logistic Regression...")
    print("="*50)
    lr_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        n_jobs=-1,
        random_state=42
    )
    lr_model.fit(X_train_scaled, y_train)

    lr_pred = lr_model.predict(X_test_scaled)
    lr_proba = lr_model.predict_proba(X_test_scaled)

    results['Logistic Regression'] = {
        'accuracy': accuracy_score(y_test, lr_pred),
        'top_3_accuracy': top_k_accuracy_score(y_test, lr_proba, k=3),
        'top_5_accuracy': top_k_accuracy_score(y_test, lr_proba, k=5),
        'log_loss': log_loss(y_test, lr_proba)
    }
    models['Logistic Regression'] = (lr_model, lr_proba, lr_pred)
    print(f"Accuracy: {results['Logistic Regression']['accuracy']:.4f}")

    # 2. Decision Tree
    print("\n" + "="*50)
    print("Training Decision Tree...")
    print("="*50)
    dt_model = DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    dt_model.fit(X_train, y_train)

    dt_pred = dt_model.predict(X_test)
    dt_proba = dt_model.predict_proba(X_test)

    results['Decision Tree'] = {
        'accuracy': accuracy_score(y_test, dt_pred),
        'top_3_accuracy': top_k_accuracy_score(y_test, dt_proba, k=3),
        'top_5_accuracy': top_k_accuracy_score(y_test, dt_proba, k=5),
        'log_loss': log_loss(y_test, dt_proba)
    }
    models['Decision Tree'] = (dt_model, dt_proba, dt_pred)
    print(f"Accuracy: {results['Decision Tree']['accuracy']:.4f}")

    # 3. XGBoost
    print("\n" + "="*50)
    print("Training XGBoost...")
    print("="*50)
    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=n_classes,
        n_jobs=-1,
        random_state=42,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)

    xgb_pred = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)

    results['XGBoost'] = {
        'accuracy': accuracy_score(y_test, xgb_pred),
        'top_3_accuracy': top_k_accuracy_score(y_test, xgb_proba, k=3),
        'top_5_accuracy': top_k_accuracy_score(y_test, xgb_proba, k=5),
        'log_loss': log_loss(y_test, xgb_proba)
    }
    models['XGBoost'] = (xgb_model, xgb_proba, xgb_pred)
    print(f"Accuracy: {results['XGBoost']['accuracy']:.4f}")

    return results, models

def create_performance_table(results):
    """Create and save performance comparison table."""
    df_results = pd.DataFrame(results).T
    df_results = df_results.round(4)
    df_results.index.name = 'Model'

    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON TABLE")
    print("="*60)
    print(df_results.to_string())

    # Save to CSV
    df_results.to_csv(OUTPUT_DIR / "performance_table.csv")
    print(f"\nTable saved to: {OUTPUT_DIR / 'performance_table.csv'}")

    return df_results

def plot_metrics(results, models, y_test, label_encoder):
    """Generate and save metric visualizations."""

    # 1. Bar chart comparing all metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    model_names = list(results.keys())
    metrics = ['accuracy', 'top_3_accuracy', 'top_5_accuracy', 'log_loss']
    titles = ['Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy', 'Log Loss']
    colors = ['#2ecc71', '#3498db', '#9b59b6']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        values = [results[m][metric] for m in model_names]
        bars = ax.bar(model_names, values, color=colors)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(title)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "metrics_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'metrics_comparison.png'}")

    # 2. Accuracy comparison (single focused plot)
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(model_names))
    width = 0.25

    acc_vals = [results[m]['accuracy'] for m in model_names]
    top3_vals = [results[m]['top_3_accuracy'] for m in model_names]
    top5_vals = [results[m]['top_5_accuracy'] for m in model_names]

    bars1 = ax.bar(x - width, acc_vals, width, label='Top-1 Accuracy', color='#e74c3c')
    bars2 = ax.bar(x, top3_vals, width, label='Top-3 Accuracy', color='#3498db')
    bars3 = ax.bar(x + width, top5_vals, width, label='Top-5 Accuracy', color='#2ecc71')

    ax.set_xlabel('Model')
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Accuracy Comparison (Top-K)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'accuracy_comparison.png'}")

    # 3. Log Loss comparison
    fig, ax = plt.subplots(figsize=(8, 5))

    log_loss_vals = [results[m]['log_loss'] for m in model_names]
    bars = ax.bar(model_names, log_loss_vals, color=['#e74c3c', '#3498db', '#2ecc71'])

    ax.set_xlabel('Model')
    ax.set_ylabel('Log Loss')
    ax.set_title('Log Loss Comparison (Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for bar, val in zip(bars, log_loss_vals):
        height = bar.get_height()
        ax.annotate(f'{val:.4f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "log_loss_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'log_loss_comparison.png'}")

def main():
    print("="*60)
    print("APPROACH 1: GEOLOCATION-BASED SPECIES CLASSIFICATION")
    print("="*60)

    # Load data
    X, y, label_encoder, df = load_and_prepare_data()

    # Adaptive stratified split (handles small classes)
    X_train, X_test, y_train, y_test = adaptive_stratified_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train and evaluate models
    results, models = train_and_evaluate_models(
        X_train, X_test, y_train, y_test, label_encoder
    )

    # Create performance table
    df_results = create_performance_table(results)

    # Generate plots
    print("\nGenerating plots...")
    plot_metrics(results, models, y_test, label_encoder)

    print("\n" + "="*60)
    print("APPROACH 1 COMPLETE")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("="*60)

    return results, models

if __name__ == "__main__":
    results, models = main()
