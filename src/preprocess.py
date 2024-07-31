import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('housing.csv')

#drop ocean_proximity
df.drop('ocean_proximity', axis=1, inplace=True)

# Handle missing values
data.fillna(method='ffill', inplace=True)

# Normalize/standardize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Save preprocessed data
pd.DataFrame(data_scaled, columns=data.columns).to_csv('data/preprocessed_data.csv', index=False)
