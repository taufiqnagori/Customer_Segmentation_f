import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset (modify path accordingly)
df = pd.read_csv("OnlineRetail.csv", encoding="ISO-8859-1")

# Drop missing values
df.dropna(inplace=True)

# Ensure correct data types
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%d-%m-%Y %H:%M", errors="coerce")

# Calculate TotalPrice (Amount)
df["Amount"] = df["Quantity"] * df["UnitPrice"]

# Reference date for recency calculation
REFERENCE_DATE = df["InvoiceDate"].max()

# Group by CustomerID
df_grouped = df.groupby("CustomerID").agg({
    "InvoiceDate": "max",  # Last purchase date
    "InvoiceNo": "count",  # Number of transactions (Frequency)
    "Amount": "sum"        # Total spending
})

# Calculate Recency
df_grouped["Recency"] = (REFERENCE_DATE - df_grouped["InvoiceDate"]).dt.days

# Rename columns
df_grouped.rename(columns={"InvoiceNo": "Frequency"}, inplace=True)

# Drop InvoiceDate
df_grouped.drop(columns=["InvoiceDate"], inplace=True)

# Ensure correct column order
expected_features = ["Recency", "Frequency", "Amount"]
df_grouped = df_grouped[expected_features]

# Apply Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_grouped)

# Train K-Means model
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(df_scaled)

# Save Scaler & Model
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(kmeans, open("kmeans_model.pkl", "wb"))

print("Model training complete. Scaler and model saved.")
