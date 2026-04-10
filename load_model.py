import pickle

# Load saved model
with open("iris_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

print("Model loaded successfully!")

# Test prediction (sample input)
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = loaded_model.predict(sample)

print("Prediction for sample input:", prediction)