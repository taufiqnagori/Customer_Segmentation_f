import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle

def load_and_clean_data(file_path):
    # Load the data
    retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)

    # Drop rows with missing CustomerID
    retail.dropna(subset=['CustomerID'], inplace=True)

    # Convert CustomerID to string
    retail['CustomerID'] = retail['CustomerID'].astype(str)

    # Create a new column 'Amount'
    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']

    # Drop rows with missing values in critical columns
    retail.dropna(subset=['InvoiceDate', 'Amount', 'Quantity', 'UnitPrice'], inplace=True)

    # Convert InvoiceDate to datetime
    retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format='%d-%m-%Y %H:%M', dayfirst=True)

    # Group by CustomerID to get RFM values
    rfm_m = retail.groupby('CustomerID')['Amount'].sum().reset_index()
    rfm_f = retail.groupby('CustomerID')['InvoiceNo'].count().reset_index()
    rfm_f.columns = ['CustomerID', 'Frequency']
    max_date = max(retail['InvoiceDate'])
    retail['Diff'] = max_date - retail['InvoiceDate']
    rfm_p = retail.groupby('CustomerID')['Diff'].min().reset_index()
    rfm_p['Diff'] = rfm_p['Diff'].dt.days
    rfm = pd.merge(rfm_m, rfm_f, on='CustomerID', how='inner')
    rfm = pd.merge(rfm, rfm_p, on='CustomerID', how='inner')
    rfm.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency']

    # Ensure columns are numeric and drop rows that cannot be converted
    rfm['Amount'] = pd.to_numeric(rfm['Amount'], errors='coerce')
    rfm['Frequency'] = pd.to_numeric(rfm['Frequency'], errors='coerce')
    rfm['Recency'] = pd.to_numeric(rfm['Recency'], errors='coerce')

    # Drop rows with NaN values after conversion
    rfm.dropna(subset=['Amount', 'Frequency', 'Recency'], inplace=True)

    # Debug: Print the DataFrame to inspect it before quantile calculation
    print("DataFrame before quantile calculation:")
    print(rfm.head())
    print(rfm.dtypes)

    # Quantile calculation
    try:
        Q1 = rfm.quantile(0.25)
        Q3 = rfm.quantile(0.75)
        IQR = Q3 - Q1
        rfm = rfm[~((rfm < (Q1 - 1.5 * IQR)) | (rfm > (Q3 + 1.5 * IQR))).any(axis=1)]
        print("DataFrame after removing outliers:")
        print(rfm)
    except Exception as e:
        print("Error during quantile calculation:", e)

    return rfm

def train_and_save_model(file_path, model_path):
    rfm = load_and_clean_data(file_path)
    rfm_df = rfm[['Amount', 'Frequency', 'Recency']]

    scaler = StandardScaler()
    rfm_df_scaled = scaler.fit_transform(rfm_df)

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(rfm_df_scaled)

    with open(model_path, 'wb') as model_file:
        pickle.dump(kmeans, model_file)

if __name__ == '__main__':
    data_file_path = 'OnlineRetail.csv'  # Path to the dataset
    model_file_path = 'kmeans_model.pkl'  # Path to save the trained model
    train_and_save_model(data_file_path, model_file_path)
    print("Model training and saving completed successfully.")