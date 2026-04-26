import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

os.makedirs("../results", exist_ok=True)
os.makedirs("../results/shap", exist_ok=True)

datasets = ["dataset1.csv", "dataset2.csv", "dataset3.csv"]

results = []

for dataset_name in datasets:
    print(f"\nRunning {dataset_name}")

    df = pd.read_csv(f"../data/{dataset_name}")
    df = df.dropna()

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    if y.dtype == "object":
        y = y.map({"negative": 0, "positive": 1})

    if y.dtype != "object" and y.nunique() > 2:
        y = (y >= y.median()).astype(int)

    X = pd.get_dummies(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        print(dataset_name, model_name, accuracy)

        results.append([
            dataset_name,
            model_name,
            accuracy,
            precision,
            recall,
            f1
        ])

        if model_name == "Random Forest":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_scaled_df)

            if isinstance(shap_values, list):
                shap_values_for_plot = shap_values[1]
            else:
                shap_values_for_plot = shap_values

            if len(shap_values_for_plot.shape) == 3:
                shap_values_for_plot = shap_values_for_plot[:, :, 1]

            mean_shap = np.abs(shap_values_for_plot).mean(axis=0)

            shap_importance = pd.DataFrame({
                "Feature": X_test_scaled_df.columns,
                "Importance": mean_shap
            }).sort_values("Importance", ascending=False)

            plt.figure(figsize=(10, 6))
            plt.bar(shap_importance["Feature"], shap_importance["Importance"])

            plt.title(f"SHAP Feature Importance - {dataset_name}", fontsize=16)
            plt.xlabel("Feature", fontsize=12)
            plt.ylabel("Mean Absolute SHAP Value", fontsize=12)

            plt.xticks(rotation=30, ha="right")
            plt.grid(axis="y", linestyle="--", alpha=0.5)

            plt.tight_layout()

            safe_name = dataset_name.replace(".csv", "")
            plt.savefig(
                f"../results/shap/{safe_name}_rf_shap_bar.png",
                bbox_inches="tight"
            )
            plt.close()

results_df = pd.DataFrame(
    results,
    columns=["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1-score"]
)

results_df.to_csv("../results/results_table.csv", index=False)

print("\nDone. Results saved in results/results_table.csv")
print("SHAP clean bar plots saved in results/shap/")
