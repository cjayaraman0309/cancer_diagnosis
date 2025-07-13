import pandas as pd

# Load data
df = pd.read_excel('data/raw/cancer-data.xlsx', sheet_name='in')

# Encode diagnosis
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Drop ID column
df = df.drop(columns=['id'])

# Save preprocessed data
df.to_csv('data/preprocess/cancer_data_preprocessed.csv', index=False)