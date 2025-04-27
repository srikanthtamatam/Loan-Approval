import pickle
import numpy as np
import pandas as pd

# Load the best model and scaler
with open("best_model.pkl", "rb") as f:
    model, scaler = pickle.load(f)

# Define the features in order
features = [
    "no_of_dependents",
    "education (Graduate/Not Graduate)",
    "self_employed (Yes/No)",
    "income_annum (numeric, e.g., 500000)",
    "loan_amount (numeric, e.g., 200000)",
    "loan_term (in years, e.g., 10)",
    "cibil_score (numeric, e.g., 750)",
    "residential_assets_value (numeric)",
    "commercial_assets_value (numeric)",
    "luxury_assets_value (numeric)",
    "bank_asset_value (numeric)"
]

# Input prompt for each feature
print("Enter the following details:")

user_input = []
for feature in features:
    while True:
        try:
            if "education" in feature.lower():
                val = input(f"{feature}: ").strip().lower()
                if val not in ['graduate', 'not graduate']:
                    raise ValueError("Enter either 'Graduate' or 'Not Graduate'")
                user_input.append(1 if val == 'graduate' else 0)

            elif "self_employed" in feature.lower():
                val = input(f"{feature}: ").strip().lower()
                if val not in ['yes', 'no']:
                    raise ValueError("Enter either 'Yes' or 'No'")
                user_input.append(1 if val == 'yes' else 0)

            else:
                val = float(input(f"{feature}: "))
                user_input.append(val)
            break
        except ValueError as e:
            print(f"Invalid input: {e}")


# Column names used during training
columns = [
    "no_of_dependents", "education", "self_employed", "income_annum", "loan_amount",
    "loan_term", "cibil_score", "residential_assets_value", "commercial_assets_value",
    "luxury_assets_value", "bank_asset_value"
]

# Create a DataFrame with the same structure
input_df = pd.DataFrame([user_input], columns=columns)

# Scale the input using the same scaler
input_array = scaler.transform(input_df)


# Predict
prediction = model.predict(input_array)[0]

# Output result
print("\n Loan Status:", "Approved" if prediction == 0 else "Rejected")
