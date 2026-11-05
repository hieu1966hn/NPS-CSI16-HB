import pickle

print("Start")

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("Loaded:", type(model))