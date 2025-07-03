import joblib
import pickle

# Load your original pickle file
with open("similarity.pkl", "rb") as f:
    similarity = pickle.load(f)

# Save compressed version
joblib.dump(similarity, "similarity_compressed.pkl", compress=3)
