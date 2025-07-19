import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import metrics
import joblib

# Unified artifacts directory
ARTIFACTS_DIR = "artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

DATA_PATH = "data/iris.csv"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.txt")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_importances.txt")
TREE_PATH = os.path.join(ARTIFACTS_DIR, "tree_structure.txt")

# Load data
data = pd.read_csv(DATA_PATH)
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train.species
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test.species

# Train
clf = DecisionTreeClassifier(max_depth=3, random_state=1)
clf.fit(X_train, y_train)

# Evaluate
predictions = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, predictions)
print(f"The accuracy of the Decision Tree is: {accuracy:.3f}")

# Save model
joblib.dump(clf, MODEL_PATH)

# Save metrics
with open(METRICS_PATH, "w") as f:
    f.write(f"accuracy: {accuracy:.3f}\n")

# Save feature importances
with open(FEATURES_PATH, "w") as f:
    for feature, importance in zip(X_train.columns, clf.feature_importances_):
        f.write(f"{feature}: {importance:.4f}\n")

# Save tree structure
with open(TREE_PATH, "w") as f:
    f.write(export_text(clf, feature_names=list(X_train.columns)))
