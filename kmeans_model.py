import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os


def perform_kmeans(df, image_folder):
    # Data cleaning and RFM calculation
    df = df.dropna(subset=['CustomerID'])
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Amount'] = df['Quantity'] * df['UnitPrice']

    # Calculate RFM values
    max_date = df['InvoiceDate'].max()
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,
        'InvoiceNo': 'count',
        'Amount': 'sum'
    }).reset_index()
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']

    # Scaling RFM values
    rfm_scaled = rfm[['Recency', 'Frequency', 'Monetary']].apply(lambda x: (x - x.mean()) / x.std())

    # K-means clustering
    kmeans = KMeans(n_clusters=3)
    rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

    # Create and save the cluster plot
    cluster_plot_path = create_cluster_plot(rfm, image_folder)

    return rfm.to_dict(orient='records'), cluster_plot_path


def create_cluster_plot(df, image_folder):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Recency', y='Monetary', hue='Cluster', palette='viridis')
    plt.title('K-means Clustering (Recency vs Monetary)')
    plt.xlabel('Recency')
    plt.ylabel('Monetary')

    plot_path = os.path.join(image_folder, 'cluster_plot.png')
    plt.savefig(plot_path)
    plt.close()

    return plot_path