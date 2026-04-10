import pickle
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict and check accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Model trained successfully!")
print("Accuracy:", acc)

# Save model to pickle file
with open("iris_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved as iris_model.pkl")