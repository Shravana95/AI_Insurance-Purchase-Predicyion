

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


DATA_FILE = "Social_Network_Ads.csv"

data = pd.read_csv(DATA_FILE)

print("First five rows:")
print(data.head())

print("\nDataset information:")
print(data.info())



X = data[["Age", "EstimatedSalary"]]
y = data["Purchased"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "SVM": SVC(kernel="rbf", random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42
    )
}


results = []
trained_models = {}

for name, model in models.items():

    
    if name in ["Logistic Regression", "KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        predictions = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

    trained_models[name] = model

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    results.append([
        name, accuracy, precision, recall, f1
    ])

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)
    print(classification_report(y_test, predictions, zero_division=0))


results_df = pd.DataFrame(
    results,
    columns=["Algorithm", "Accuracy", "Precision", "Recall", "F1-Score"]
)

print("\nModel Comparison:")
print(results_df.to_string(index=False))

results_df.to_csv("model_comparison.csv", index=False)


plt.figure(figsize=(10, 6))
plt.bar(results_df["Algorithm"], results_df["Accuracy"])
plt.xlabel("Classification Algorithm")
plt.ylabel("Accuracy")
plt.title("Comparison of Classification Algorithms")
plt.xticks(rotation=25)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("algorithm_accuracy_comparison.png", dpi=300)
plt.show()


plt.figure(figsize=(9, 6))

for value in [0, 1]:
    subset = data[data["Purchased"] == value]
    label = "Not Purchased" if value == 0 else "Purchased"
    plt.scatter(
        subset["Age"],
        subset["EstimatedSalary"],
        label=label,
        alpha=0.7
    )

plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.title("Age vs Estimated Salary and Insurance Purchase")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig("age_salary_purchase.png", dpi=300)
plt.show()


test_cases = pd.DataFrame({
    "Age": [30, 40, 40, 50, 18, 22, 35, 60],
    "EstimatedSalary": [
        87000, 0, 100000, 0,
        0, 600000, 2500000, 100000000
    ]
})


best_name = results_df.loc[results_df["Accuracy"].idxmax(), "Algorithm"]
best_model = trained_models[best_name]

if best_name in ["Logistic Regression", "KNN", "SVM"]:
    test_cases_scaled = scaler.transform(test_cases)
    test_predictions = best_model.predict(test_cases_scaled)
else:
    test_predictions = best_model.predict(test_cases)

test_cases["Prediction"] = test_predictions
test_cases["Prediction_Label"] = test_cases["Prediction"].map({
    0: "Not Purchased",
    1: "Purchased"
})

print("\nPredictions for required cases:")
print(test_cases.to_string(index=False))

test_cases.to_csv("required_test_case_predictions.csv", index=False)


if best_name in ["Logistic Regression", "KNN", "SVM"]:
    best_predictions = best_model.predict(X_test_scaled)
else:
    best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - " + best_name)
plt.colorbar()
plt.xticks([0, 1], ["Not Purchased", "Purchased"])
plt.yticks([0, 1], ["Not Purchased", "Purchased"])
plt.xlabel("Predicted")
plt.ylabel("Actual")

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("confusion_matrix_best_model.png", dpi=300)
plt.show()

print("\nBest model based on test accuracy:", best_name)
print("Confusion Matrix:")
print(cm)

print("\nProject completed.")
