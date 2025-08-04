import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

# Paths
DATA_PATH = "data/iris.csv"
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Poisoning function
def poison_data_features(df, poison_percent, seed=42):
    np.random.seed(seed)
    df_poisoned = df.copy()
    num_samples = int(len(df_poisoned) * poison_percent)
    poison_indices = np.random.choice(df_poisoned.index, size=num_samples, replace=False)

    noise = np.random.normal(loc=0, scale=0.5, size=(num_samples, 4))  # Gaussian noise
    df_poisoned.loc[poison_indices, ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']] += noise

    return df_poisoned, poison_indices

# Logging summary
summary = []

# Run for multiple poison levels
for level in [0.0, 0.05, 0.10, 0.50]:
    # Poison data
    raw_data = pd.read_csv(DATA_PATH)
    poisoned_data, poison_indices = poison_data_features(raw_data, poison_percent=level)

    # Save poisoned indices
    poison_filename = f"poisoned_indices_{int(level*100)}.txt"
    with open(os.path.join(ARTIFACTS_DIR, poison_filename), "w") as f:
        for idx in poison_indices:
            f.write(f"{idx}\n")

    # Split and prepare
    train, test = train_test_split(poisoned_data, test_size=0.4, stratify=poisoned_data['species'], random_state=42)
    X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train.species
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test.species

    # Train model
    clf = DecisionTreeClassifier(max_depth=3, random_state=1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=clf.classes_)

    # Save model
    model_name = f"model_{int(level*100)}.pkl"
    joblib.dump(clf, os.path.join(ARTIFACTS_DIR, model_name))

    # Save metrics
    with open(os.path.join(ARTIFACTS_DIR, f"metrics_{int(level*100)}.txt"), "w") as f:
        f.write(f"accuracy: {accuracy:.3f}\n")

    # Save feature importances
    with open(os.path.join(ARTIFACTS_DIR, f"feature_importances_{int(level*100)}.txt"), "w") as f:
        for feature, importance in zip(X_train.columns, clf.feature_importances_):
            f.write(f"{feature}: {importance:.4f}\n")

    # Save tree structure
    with open(os.path.join(ARTIFACTS_DIR, f"tree_structure_{int(level*100)}.txt"), "w") as f:
        f.write(export_text(clf, feature_names=list(X_train.columns)))

    # Save confusion matrix plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=clf.classes_, yticklabels=clf.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {int(level*100)}% Poisoned')
    plt.tight_layout()
    cm_filename = f"confusion_matrix_{int(level*100)}.png"
    plt.savefig(os.path.join(ARTIFACTS_DIR, cm_filename))
    plt.close()

    # Save row to summary
    summary.append((f"{int(level*100)}%", accuracy, cm))

# Print summary table
print("\n📊 Model Performance Summary:")
print("Poison Level | Accuracy | Confusion Matrix")
print("---------------------------------------------")
for level, acc, cm in summary:
    print(f"{level:>12} |   {acc:.3f}  | {cm.tolist()}")

# (Optional) Save mitigation tips
with open(os.path.join(ARTIFACTS_DIR, "mitigation.txt"), "w") as f:
    f.write("""
# 🛡️ Mitigation Strategies Against Data Poisoning

1. **Data Sanitization**: Use statistical techniques like z-score or isolation forest to remove feature outliers.
2. **Robust Models**: Train with models that are less sensitive to noise, like RANSAC or tree ensembles.
3. **Adversarial Training**: Inject synthetic poisoned data during training to improve robustness.
4. **Data Provenance**: Track dataset versions with DVC or Git-LFS to detect unauthorized changes.
5. **Ensemble Learning**: Voting systems reduce the influence of any single poisoned sample.
    """)

