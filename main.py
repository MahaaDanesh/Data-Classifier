from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Loaded Successfully!")
print("Feature Names:", iris.feature_names)
print("Target Names:", iris.target_names)
print("Total Samples:", len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("\nSample Data:", sample)
print("Predicted Flower:", iris.target_names[prediction[0]])
