from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

app = Flask(__name__)

# Load trained model and scaler
try:
    model = pickle.load(open("kmeans_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    print("Model and scaler loaded successfully!")
except Exception as e:
    print("Error loading model/scaler:", e)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/result", methods=["POST"])
def result():
    if "file" not in request.files:
        return "No file uploaded!"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file!"

    # Read uploaded CSV
    df = pd.read_csv(file, encoding="ISO-8859-1")

    # Ensure necessary columns exist
    required_columns = ["CustomerID", "InvoiceDate", "InvoiceNo", "Quantity", "UnitPrice"]
    for col in required_columns:
        if col not in df.columns:
            return f"Error: Missing column '{col}' in CSV file."

    # Data Preprocessing
    df.dropna(subset=["CustomerID"], inplace=True)
    df["CustomerID"] = df["CustomerID"].astype(str)
    df["Amount"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], format="%d-%m-%Y %H:%M", errors="coerce")

    # RFM Calculation
    max_date = df["InvoiceDate"].max()
    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (max_date - x.max()).days,  # Recency
        "InvoiceNo": "count",  # Frequency
        "Amount": "sum"  # Amount instead of Monetary
    }).reset_index()

    rfm.columns = ["CustomerID", "Recency", "Frequency", "Amount"]

    # Apply Scaling
    rfm_scaled = scaler.transform(rfm[["Recency", "Frequency", "Amount"]])

    # Predict Clusters
    rfm["Cluster"] = model.predict(rfm_scaled)

    # Ensure `static/` directory exists
    if not os.path.exists("static"):
        os.makedirs("static")

    # Plot 1: Cluster vs Amount
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=rfm["Cluster"], y=rfm["Amount"], hue=rfm["Cluster"], palette="coolwarm")
    plt.title("Cluster vs Amount")
    plt.xlabel("Cluster ID")
    plt.ylabel("Amount")
    plt.savefig("static/cluster_vs_amount.png")
    plt.close()

    # Plot 2: Cluster vs Frequency
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=rfm["Cluster"], y=rfm["Frequency"], hue=rfm["Cluster"], palette="coolwarm")
    plt.title("Cluster vs Frequency")
    plt.xlabel("Cluster ID")
    plt.ylabel("Frequency")
    plt.savefig("static/cluster_vs_frequency.png")
    plt.close()

    # Plot 3: Cluster vs Recency
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=rfm["Cluster"], y=rfm["Recency"], hue=rfm["Cluster"], palette="coolwarm")
    plt.title("Cluster vs Recency")
    plt.xlabel("Cluster ID")
    plt.ylabel("Recency")
    plt.savefig("static/cluster_vs_recency.png")
    plt.close()

    return render_template("result.html")

if __name__ == "__main__":
    app.run(debug=True)