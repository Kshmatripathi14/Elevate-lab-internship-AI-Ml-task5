import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

def main():
    print("=== Decision Trees & Random Forest Classifier ===")
    file_path = input("Enter dataset CSV path (e.g., heart.csv):")

    try:
        data = pd.read_csv('C:\\Users\\lenovo\\Downloads\\Elevate labs internship\\AI ML\\task5\\heart.csv')
        print("\nDataset loaded successfully ✅")
        print("First 5 rows:\n", data.head())
    except Exception as e:
        print("Error loading dataset:", e)
        return

    target_col = input("\nEnter target column name (e.g., target): ")

    if target_col not in data.columns:
        print(f"❌ Target column '{target_col}' not found in dataset.")
        return

    # Features & target
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ==========================
    # Decision Tree Classifier
    # ==========================
    print("\nTraining Decision Tree Classifier...")
    dt_clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt_clf.fit(X_train, y_train)

    y_pred_dt = dt_clf.predict(X_test)
    acc_dt = accuracy_score(y_test, y_pred_dt)

    print(f"Decision Tree Accuracy: {acc_dt:.4f}")
    print("\nClassification Report (Decision Tree):\n", classification_report(y_test, y_pred_dt))

    # Visualize tree
    plt.figure(figsize=(14, 8))
    plot_tree(dt_clf, filled=True, feature_names=X.columns, class_names=True, rounded=True)
    plt.title("Decision Tree Visualization")
    plt.savefig("decision_tree.png")
    plt.show()

    # ==========================
    # Random Forest Classifier
    # ==========================
    print("\nTraining Random Forest Classifier...")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)

    y_pred_rf = rf_clf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)

    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    print("\nClassification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))

    # ==========================
    # Feature Importances
    # ==========================
    importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=importances.index, palette="viridis")
    plt.title("Feature Importances (Random Forest)")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.savefig("feature_importances.png")
    plt.show()

    # ==========================
    # Cross-validation
    # ==========================
    print("\nCross-validation Results:")
    dt_scores = cross_val_score(dt_clf, X, y, cv=5)
    rf_scores = cross_val_score(rf_clf, X, y, cv=5)

    print(f"Decision Tree CV Accuracy: {dt_scores.mean():.4f} ± {dt_scores.std():.4f}")
    print(f"Random Forest CV Accuracy: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")

    # ==========================
    # Save Models
    # ==========================
    joblib.dump(dt_clf, "decision_tree_model.pkl")
    joblib.dump(rf_clf, "random_forest_model.pkl")
    print("\nModels saved: decision_tree_model.pkl & random_forest_model.pkl")
    

if __name__ == "__main__":
    main()
