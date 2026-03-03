# output FINAL.csv (ID, Class)

# import required things
import numpy as np
import pandas as pd
import pickle
import os

TEST_PATH  = "TEST.csv"
MODEL_PATH = os.path.join("models", "stacking_model.pkl")
OUTPUT_PATH = "FINAL.csv"

# load model
print("Loading model...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. Run train.py first."
    )

with open(MODEL_PATH, 'rb') as f:
    stack_model = pickle.load(f)

print("Model loaded successfully.")

# load test data
test_df = pd.read_csv(TEST_PATH)
print(f"Test shape: {test_df.shape}")

ids = test_df['ID']
X_test = test_df.drop('Class',axis=1)

#prediction
prediction = stack_model.predict(X_test)

final = pd.DataFrame(
    {
        "ID" : ids,
        "Class" : prediction
    }
)

final.to_csv(OUTPUT_PATH,index=False)
print(f"\nFINAL.csv saved to: {OUTPUT_PATH}")
