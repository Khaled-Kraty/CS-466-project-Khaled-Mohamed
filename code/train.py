import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

datasets = ["dataset1.csv", "dataset2.csv", "dataset3.csv"]

results = []

for dataset_name in datasets:
    print(f"\nRunning {dataset_name}")

    df = pd.read_csv(f"../data/{dataset_name}")

    # Remove empty rows
    df = df.dropna()

    # Assume target is the last column
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # If target is continuous, convert it to binary classes
    if y.dtype != "object" and y.nunique() > 2:
        y = (y >= y.median()).astype(int)

    # Convert text/categorical columns to numbers
    X = pd.get_dummies(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

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

results_df = pd.DataFrame(
    results,
    columns=["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1-score"]
)

results_df.to_csv("../results/results_table.csv", index=False)

print("\nDone. Results saved in results/results_table.csv")