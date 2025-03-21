import pandas as pd
import numpy as np

def load_and_clean_data(file_path):
    # Load the data
    retail = pd.read_csv(file_path, sep=",", encoding="ISO-8859-1", header=0)

    # Convert CustomerID to string
    retail['CustomerID'] = retail['CustomerID'].astype(str)

    # Create a new column 'Amount'
    retail['Amount'] = retail['Quantity'] * retail['UnitPrice']

    # Drop rows with missing values
    retail.dropna(subset=['CustomerID', 'Amount', 'Quantity', 'UnitPrice', 'InvoiceDate'], inplace=True)

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

    # Ensure columns are numeric
    rfm['Amount'] = pd.to_numeric(rfm['Amount'], errors='coerce')
    rfm['Frequency'] = pd.to_numeric(rfm['Frequency'], errors='coerce')
    rfm['Recency'] = pd.to_numeric(rfm['Recency'], errors='coerce')

    # Drop rows with NaN values after conversion
    rfm.dropna(subset=['Amount', 'Frequency', 'Recency'], inplace=True)

    # Handle outliers using IQR method
    Q1 = rfm.quantile(0.25)
    Q3 = rfm.quantile(0.75)
    IQR = Q3 - Q1
    rfm = rfm[~((rfm < (Q1 - 1.5 * IQR)) | (rfm > (Q3 + 1.5 * IQR))).any(axis=1)]

    return rfm

# Example usage
if __name__ == '__main__':
    data_file_path = r'C:\Users\USER\PycharmProjects\Custumer_Segmentaion_f\OnlineRetail.csv'
    cleaned_data = load_and_clean_data(data_file_path)
    print(cleaned_data.head())
    print(cleaned_data.info())