import os
import unittest
import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Paths and constants
ARTIFACTS_DIR = "artifacts"
DATA_PATH = "data/iris.csv"
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "model.pkl")
METRICS_PATH = os.path.join(ARTIFACTS_DIR, "metrics.txt")
FEATURES_PATH = os.path.join(ARTIFACTS_DIR, "feature_importances.txt")
TREE_PATH = os.path.join(ARTIFACTS_DIR, "tree_structure.txt")

class TestIrisPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PATH)
        cls.model = joblib.load(MODEL_PATH)

    def test_data_shape(self):
        self.assertGreater(self.df.shape[0], 0, "Dataset is empty")
        self.assertEqual(self.df.shape[1], 5, "Dataset should have 5 columns")

    def test_no_nulls(self):
        self.assertFalse(self.df.isnull().any().any(), "Dataset contains null values")

    def test_column_types(self):
        expected_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
        for col in expected_columns:
            self.assertIn(col, self.df.columns, f"Missing column: {col}")
        self.assertTrue(self.df['species'].dtype == object or str(self.df['species'].dtype).startswith('category'))

    def test_species_balance(self):
        counts = self.df['species'].value_counts()
        for species, count in counts.items():
            self.assertGreater(count, 0, f"Species {species} has no samples")

    def test_model_file_exists(self):
        self.assertTrue(os.path.exists(MODEL_PATH), "Model file not found")

    def test_model_type(self):
        self.assertIsInstance(self.model, DecisionTreeClassifier, "Model is not a DecisionTreeClassifier")

    def test_prediction_accuracy(self):
        X = self.df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        y = self.df['species']
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        self.assertGreater(acc, 0.7, f"Model accuracy too low: {acc:.3f}")

    def test_metrics_file(self):
        self.assertTrue(os.path.exists(METRICS_PATH), "Metrics file not found")
        with open(METRICS_PATH) as f:
            content = f.read()
        self.assertIn("accuracy", content.lower(), "Accuracy not found in metrics file")

    def test_feature_importances_file(self):
        self.assertTrue(os.path.exists(FEATURES_PATH), "Feature importances file not found")
        with open(FEATURES_PATH) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 4, "Should have 4 feature importances")

    def test_tree_structure_file(self):
        self.assertTrue(os.path.exists(TREE_PATH), "Tree structure file not found")
        with open(TREE_PATH) as f:
            content = f.read()
        self.assertIn("class:", content.lower(), "Tree structure seems invalid")

if __name__ == '__main__':
    unittest.main()
