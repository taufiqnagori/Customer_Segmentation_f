import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle

# Load dataset
file_path = "OnlineRetail.csv"  # Ensure the dataset is in the same directory
retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)

# Drop missing CustomerIDs
retail.dropna(subset=['CustomerID'], inplace=True)
retail['CustomerID'] = retail['CustomerID'].astype(str)

# Compute Amount column
retail['Amount'] = retail['Quantity'] * retail['UnitPrice']

# Aggregate RFM metrics
rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()
rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()
rfm_f.columns = ['CustomerID', 'Frequency']

# Convert InvoiceDate to datetime
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format='%d-%m-%Y %H:%M', dayfirst=True)
max_date = max(retail['InvoiceDate'])

# Calculate Recency
retail['Diff'] = max_date - retail['InvoiceDate']
rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()
rfm_p['Diff'] = rfm_p['Diff'].dt.days  # Convert timedelta to days

# Merge RFM metrics
rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

# Convert to numeric and drop NaN values
rfm[['Amount', 'Frequency', 'Recency']] = rfm[['Amount', 'Frequency', 'Recency']].apply(pd.to_numeric, errors='coerce')
rfm.dropna(subset=['Amount', 'Frequency', 'Recency'], inplace=True)

# Scale the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Amount', 'Frequency', 'Recency']])

# Save the scaler
scaler_path = "scaler.pkl"  # Save it in the same directory as app.py
with open(scaler_path, 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print(f"Scaler saved successfully at: {scaler_path}")
