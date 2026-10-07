from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


def main():

    # 1. Load the Iris dataset
    X, y = load_iris(return_X_y=True)

    # 2. Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3. Create Decision Tree model
    model = DecisionTreeClassifier(
        max_depth=3,
        random_state=42
    )

    # 4. Train the model
    model.fit(X_train, y_train)

    # 5. Make predictions
    y_pred = model.predict(X_test)

    # 6. Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        average="macro"
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro"
    )

    # 7. Display results
    print("Accuracy :", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall   :", round(recall, 2))


if __name__ == "__main__":
    main()