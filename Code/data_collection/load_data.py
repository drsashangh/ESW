import pandas as pd

# Set the path to your synthea csv output folder
# Make sure to change this to your actual path!
# For example: 'C:/Users/YourUser/Documents/synthea/output/csv/'
path_to_csvs = './synthea/output/csv/'

# --- 1. Explore patients.csv ---
try:
    patients_df = pd.read_csv(path_to_csvs + 'patients.csv')
    print("--- Contents of patients.csv ---")
    print(patients_df.head())
    print("\n")
except FileNotFoundError:
    print("patients.csv not found in the specified path.")


# --- 2. Explore conditions.csv ---
try:
    conditions_df = pd.read_csv(path_to_csvs + 'conditions.csv')
    print("--- Contents of conditions.csv ---")
    print(conditions_df.head())
    print("\n")
except FileNotFoundError:
    print("conditions.csv not found in the specified path.")


# --- 3. Explore observations.csv ---
try:
    observations_df = pd.read_csv(path_to_csvs + 'observations.csv')
    print("--- Contents of observations.csv ---")
    print(observations_df.head())
    print("\n")
except FileNotFoundError:
    print("observations.csv not found in the specified path.")