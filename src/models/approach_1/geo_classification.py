"""
Approach 1: Geographic Multi-Classification Models

Predict species (primary_label) from latitude and longitude coordinates.
Models: Logistic Regression, Decision Tree, XGBoost

Author: Claude Code
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import json

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    top_k_accuracy_score,
    log_loss,
    f1_score,
    classification_report,
    confusion_matrix
)
from scipy.stats import uniform, randint

# Scoring metric for imbalanced data
SCORING_METRIC = 'balanced_accuracy'

warnings.filterwarnings('ignore')

# Try to import xgboost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Will skip XGBoost model.")


def custom_stratified_split(X, y, test_size=0.2, random_state=42):
    """
    Custom stratified split that handles small classes.
    For classes with few samples, splits 50/50 to ensure representation in both sets.
    """
    np.random.seed(random_state)

    unique_classes = np.unique(y)

    train_indices = []
    test_indices = []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        n_samples = len(cls_indices)

        # Shuffle indices for this class
        np.random.shuffle(cls_indices)

        if n_samples == 1:
            # Only 1 sample: put in train (can't split)
            train_indices.extend(cls_indices)
        elif n_samples < 5:
            # Small class: split 50/50 to ensure at least 1 in each set
            n_test = max(1, n_samples // 2)
            test_indices.extend(cls_indices[:n_test])
            train_indices.extend(cls_indices[n_test:])
        else:
            # Normal stratified split
            n_test = max(1, int(n_samples * test_size))
            test_indices.extend(cls_indices[:n_test])
            train_indices.extend(cls_indices[n_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    # Shuffle the final indices
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return train_indices, test_indices


def load_and_prepare_data(data_path: str, test_size: float = 0.2, random_state: int = 42):
    """Load train.csv and prepare for modeling with stratified split."""

    df = pd.read_csv(data_path)

    # Features and target
    X = df[['latitude', 'longitude']].values
    y = df['primary_label'].values

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Check class distribution
    unique, counts = np.unique(y_encoded, return_counts=True)
    min_count = counts.min()
    small_classes = (counts < 5).sum()

    print(f"Class distribution: min={min_count}, max={counts.max()}, mean={counts.mean():.1f}")
    print(f"Classes with <5 samples: {small_classes}")

    # Use custom stratified split to handle small classes
    train_indices, test_indices = custom_stratified_split(
        X, y_encoded,
        test_size=test_size,
        random_state=random_state
    )

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y_encoded[train_indices], y_encoded[test_indices]

    # Verify split
    train_classes = np.unique(y_train)
    test_classes = np.unique(y_test)

    print(f"Stratified split completed:")
    print(f"  Train samples: {len(y_train)}, classes: {len(train_classes)}")
    print(f"  Test samples: {len(y_test)}, classes: {len(test_classes)}")

    # Check for classes only in train (not in test)
    train_only = set(train_classes) - set(test_classes)
    if train_only:
        print(f"  Warning: {len(train_only)} classes only in train set (single-sample classes)")

    # Scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'label_encoder': label_encoder,
        'scaler': scaler,
        'n_classes': len(label_encoder.classes_),
        'class_names': label_encoder.classes_
    }


def get_safe_cv(y_train, max_splits=3):
    """
    Get a safe cross-validation strategy based on minimum class size.
    Adjusts number of splits to ensure all classes have enough samples.
    """
    _, counts = np.unique(y_train, return_counts=True)
    min_class_size = counts.min()

    # Number of splits cannot exceed minimum class size
    n_splits = min(max_splits, min_class_size)

    if n_splits < 2:
        # If we can't do stratified CV, fall back to simple 2-fold
        print(f"  Warning: min class size={min_class_size}, using 2-fold CV without stratification")
        return 2  # Simple integer CV
    else:
        print(f"  Using {n_splits}-fold StratifiedKFold (min class size={min_class_size})")
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)


def train_logistic_regression(X_train, y_train, n_classes, n_iter=20):
    """Train Logistic Regression with hyperparameter tuning."""
    print("Training Logistic Regression with hyperparameter tuning...", flush=True)

    base_model = LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        n_jobs=-1,
        random_state=42
    )

    param_dist = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l2'],
        'class_weight': [None, 'balanced'],
        'tol': [1e-4, 1e-3, 1e-2]
    }

    cv = get_safe_cv(y_train, max_splits=3)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=SCORING_METRIC,  # balanced_accuracy for imbalanced data
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(X_train, y_train)

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV {SCORING_METRIC}: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.cv_results_


def train_decision_tree(X_train, y_train, n_classes, n_iter=20):
    """Train Decision Tree with hyperparameter tuning."""
    print("Training Decision Tree with hyperparameter tuning...", flush=True)

    base_model = DecisionTreeClassifier(random_state=42)

    param_dist = {
        'max_depth': [5, 10, 15, 20, 30, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    }

    cv = get_safe_cv(y_train, max_splits=3)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=SCORING_METRIC,  # balanced_accuracy for imbalanced data
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(X_train, y_train)

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV {SCORING_METRIC}: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.cv_results_


def train_xgboost(X_train, y_train, n_classes, n_iter=20):
    """Train XGBoost with hyperparameter tuning."""
    print("Training XGBoost with hyperparameter tuning...", flush=True)

    base_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        n_jobs=-1,
        random_state=42,
        verbosity=0,
        use_label_encoder=False
    )

    param_dist = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5]
    }

    cv = get_safe_cv(y_train, max_splits=3)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=SCORING_METRIC,  # balanced_accuracy for imbalanced data
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(X_train, y_train)

    print(f"  Best params: {search.best_params_}")
    print(f"  Best CV {SCORING_METRIC}: {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.cv_results_


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics suitable for imbalanced data."""

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    # Standard accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Balanced accuracy (better for imbalanced data)
    bal_accuracy = balanced_accuracy_score(y_test, y_pred)

    # F1 scores (macro and weighted for imbalanced data)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # Top-k accuracy
    top3_acc = top_k_accuracy_score(y_test, y_prob, k=3)
    top5_acc = top_k_accuracy_score(y_test, y_prob, k=5)
    top10_acc = top_k_accuracy_score(y_test, y_prob, k=10)

    # Log loss
    logloss = log_loss(y_test, y_prob)

    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'balanced_accuracy': bal_accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'top3_accuracy': top3_acc,
        'top5_accuracy': top5_acc,
        'top10_accuracy': top10_acc,
        'log_loss': logloss
    }

    return metrics, y_pred, y_prob


def create_performance_table(results: list, output_path: str):
    """Create and save performance comparison table."""

    df = pd.DataFrame(results)
    df = df.set_index('model')

    # Format percentages
    df_display = df.copy()
    pct_cols = ['accuracy', 'balanced_accuracy', 'f1_macro', 'f1_weighted',
                'top3_accuracy', 'top5_accuracy', 'top10_accuracy']
    for col in pct_cols:
        df_display[col] = df_display[col].apply(lambda x: f'{x*100:.2f}%')
    df_display['log_loss'] = df_display['log_loss'].apply(lambda x: f'{x:.4f}')

    # Save as CSV
    df.to_csv(output_path)

    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON TABLE (Imbalanced Metrics)")
    print("="*80)
    print(df_display.to_string())
    print("="*80)

    return df


def plot_accuracy_comparison(results: list, output_dir: str):
    """Plot accuracy comparison bar chart including balanced accuracy."""

    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Standard vs Balanced Accuracy + F1 scores
    ax1 = axes[0]
    x = np.arange(len(df))
    width = 0.2

    bars1 = ax1.bar(x - 1.5*width, df['accuracy'], width, label='Accuracy', color='#3498db')
    bars2 = ax1.bar(x - 0.5*width, df['balanced_accuracy'], width, label='Balanced Acc', color='#2ecc71')
    bars3 = ax1.bar(x + 0.5*width, df['f1_macro'], width, label='F1 Macro', color='#e74c3c')
    bars4 = ax1.bar(x + 1.5*width, df['f1_weighted'], width, label='F1 Weighted', color='#9b59b6')

    ax1.set_xlabel('Model')
    ax1.set_ylabel('Score')
    ax1.set_title('Imbalanced Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['model'], rotation=15)
    ax1.legend(loc='upper right')
    ax1.set_ylim(0, 1)

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.1%}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7, rotation=90)

    # Plot 2: Top-K Accuracy
    ax2 = axes[1]
    bars1 = ax2.bar(x - width, df['top3_accuracy'], width, label='Top-3', color='#3498db')
    bars2 = ax2.bar(x, df['top5_accuracy'], width, label='Top-5', color='#2ecc71')
    bars3 = ax2.bar(x + width, df['top10_accuracy'], width, label='Top-10', color='#e74c3c')

    ax2.set_xlabel('Model')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Top-K Accuracy Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['model'], rotation=15)
    ax2.legend(loc='lower right')
    ax2.set_ylim(0, 1)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.1%}',
                        xy=(bar.get_x() + bar.get_width()/2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, rotation=90)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_comparison.png', dpi=150)
    plt.close()

    print(f"Saved: {output_dir}/accuracy_comparison.png")


def plot_log_loss_comparison(results: list, output_dir: str):
    """Plot log loss comparison."""

    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#2ecc71', '#3498db', '#e74c3c']
    bars = ax.bar(df['model'], df['log_loss'], color=colors[:len(df)])

    ax.set_xlabel('Model')
    ax.set_ylabel('Log Loss')
    ax.set_title('Model Log Loss Comparison (Lower is Better)')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                   xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/log_loss_comparison.png', dpi=150)
    plt.close()

    print(f"Saved: {output_dir}/log_loss_comparison.png")


def plot_metrics_radar(results: list, output_dir: str):
    """Plot radar chart of metrics including imbalanced metrics."""

    df = pd.DataFrame(results)

    # Normalize log_loss (invert so higher is better)
    max_logloss = df['log_loss'].max()
    df['log_loss_inv'] = 1 - (df['log_loss'] / max_logloss)

    categories = ['Balanced Acc', 'F1 Macro', 'Top-3 Acc', 'Top-5 Acc', 'Top-10 Acc', 'Log Loss (inv)']

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    colors = ['#2ecc71', '#3498db', '#e74c3c']

    for idx, row in df.iterrows():
        values = [
            row['balanced_accuracy'],
            row['f1_macro'],
            row['top3_accuracy'],
            row['top5_accuracy'],
            row['top10_accuracy'],
            row['log_loss_inv']
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Model Performance Radar Chart (Imbalanced Metrics)')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/metrics_radar.png', dpi=150)
    plt.close()

    print(f"Saved: {output_dir}/metrics_radar.png")


def plot_prediction_map(X_test, y_test, y_pred, model_name, output_dir: str):
    """Plot geographic distribution of predictions."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Correct predictions
    correct = y_pred == y_test

    # Plot 1: All predictions colored by correctness
    ax1 = axes[0]
    scatter = ax1.scatter(X_test[correct, 1], X_test[correct, 0],
                         c='green', alpha=0.5, s=10, label='Correct')
    ax1.scatter(X_test[~correct, 1], X_test[~correct, 0],
               c='red', alpha=0.5, s=10, label='Incorrect')
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title(f'{model_name}: Prediction Correctness')
    ax1.legend()

    # Plot 2: Pantanal region zoom
    ax2 = axes[1]
    pantanal_mask = (
        (X_test[:, 0] >= -22) & (X_test[:, 0] <= -16) &
        (X_test[:, 1] >= -58) & (X_test[:, 1] <= -55)
    )

    if pantanal_mask.sum() > 0:
        ax2.scatter(X_test[pantanal_mask & correct, 1], X_test[pantanal_mask & correct, 0],
                   c='green', alpha=0.7, s=20, label='Correct')
        ax2.scatter(X_test[pantanal_mask & ~correct, 1], X_test[pantanal_mask & ~correct, 0],
                   c='red', alpha=0.7, s=20, label='Incorrect')
        ax2.set_xlabel('Longitude')
        ax2.set_ylabel('Latitude')
        ax2.set_title(f'{model_name}: Pantanal Region')
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, 'No test samples\nin Pantanal region',
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title(f'{model_name}: Pantanal Region')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_map_{model_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.close()

    print(f"Saved: {output_dir}/prediction_map_{model_name.lower().replace(' ', '_')}.png")


def main():
    # Paths
    base_dir = Path(__file__).parent.parent.parent.parent
    data_path = base_dir / "data" / "raw" / "train.csv"
    output_dir = Path(__file__).parent / "plots"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("APPROACH 1: Geographic Multi-Classification")
    print("Target: primary_label | Features: latitude, longitude")
    print("="*70)
    print()

    # Load data
    print("Loading and preparing data...", flush=True)
    data = load_and_prepare_data(data_path)

    print(f"Training samples: {len(data['y_train'])}")
    print(f"Testing samples: {len(data['y_test'])}")
    print(f"Number of classes: {data['n_classes']}")
    print()

    # Train models
    models = {}
    best_params = {}
    results = []
    predictions = {}

    # 1. Logistic Regression (uses scaled features)
    model, params, cv_results = train_logistic_regression(
        data['X_train_scaled'],
        data['y_train'],
        data['n_classes']
    )
    models['Logistic Regression'] = model
    best_params['Logistic Regression'] = params
    metrics, y_pred, y_prob = evaluate_model(
        model,
        data['X_test_scaled'],
        data['y_test'],
        'Logistic Regression'
    )
    results.append(metrics)
    predictions['Logistic Regression'] = (y_pred, y_prob)
    print(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f} | F1 Macro: {metrics['f1_macro']:.4f} | Top-5: {metrics['top5_accuracy']:.4f}")

    # 2. Decision Tree (uses raw features)
    model, params, cv_results = train_decision_tree(
        data['X_train'],
        data['y_train'],
        data['n_classes']
    )
    models['Decision Tree'] = model
    best_params['Decision Tree'] = params
    metrics, y_pred, y_prob = evaluate_model(
        model,
        data['X_test'],
        data['y_test'],
        'Decision Tree'
    )
    results.append(metrics)
    predictions['Decision Tree'] = (y_pred, y_prob)
    print(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f} | F1 Macro: {metrics['f1_macro']:.4f} | Top-5: {metrics['top5_accuracy']:.4f}")

    # 3. XGBoost (uses raw features)
    if HAS_XGBOOST:
        model, params, cv_results = train_xgboost(
            data['X_train'],
            data['y_train'],
            data['n_classes']
        )
        models['XGBoost'] = model
        best_params['XGBoost'] = params
        metrics, y_pred, y_prob = evaluate_model(
            model,
            data['X_test'],
            data['y_test'],
            'XGBoost'
        )
        results.append(metrics)
        predictions['XGBoost'] = (y_pred, y_prob)
        print(f"  Balanced Acc: {metrics['balanced_accuracy']:.4f} | F1 Macro: {metrics['f1_macro']:.4f} | Top-5: {metrics['top5_accuracy']:.4f}")

    print()

    # Create performance table
    perf_table = create_performance_table(results, output_dir / "performance_table.csv")

    # Create plots
    print("\nGenerating plots...", flush=True)
    plot_accuracy_comparison(results, output_dir)
    plot_log_loss_comparison(results, output_dir)
    plot_metrics_radar(results, output_dir)

    # Prediction maps for each model
    for model_name, (y_pred, y_prob) in predictions.items():
        X_test = data['X_test_scaled'] if model_name == 'Logistic Regression' else data['X_test']
        # Use unscaled for plotting
        plot_prediction_map(data['X_test'], data['y_test'], y_pred, model_name, output_dir)

    # Save results summary
    summary = {
        'approach': 'approach_1',
        'description': 'Geographic classification using latitude/longitude',
        'features': ['latitude', 'longitude'],
        'target': 'primary_label',
        'n_classes': data['n_classes'],
        'train_samples': len(data['y_train']),
        'test_samples': len(data['y_test']),
        'scoring_metric': SCORING_METRIC,
        'best_hyperparameters': {k: str(v) for k, v in best_params.items()},
        'results': results
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {output_dir}/summary.json")
    print("\nApproach 1 completed!")


if __name__ == "__main__":
    main()
