# Customer Segmentation

This project is a university-level project focused on customer segmentation using machine learning techniques. The project utilizes the K-Means clustering algorithm to segment customers based on their purchasing behavior.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)

## Introduction

Customer segmentation is the practice of dividing a customer base into groups of individuals that are similar in specific ways relevant to marketing, such as age, gender, interests, and spending habits. This project aims to segment customers based on their purchasing behavior using the K-Means clustering algorithm.

## Features

- Load and preprocess customer data
- Calculate Recency, Frequency, and Monetary (RFM) values
- Scale the RFM values
- Apply K-Means clustering to segment customers
- Visualize the clusters

## Installation

1. Clone the repository:
   sh
   git clone https://github.com/your-username/Customer_Segmentation_f.git

   cd Customer_Segmentation_f

2. Create a virtual environment and activate it:

    python -m venv venv

     .\venv\Scripts\activate  # On Windows

     source venv/bin/activate  # On macOS/Linux


3. Install the required packages:

    pip install -r requirements.txt


## Usage

1. Ensure you have the dataset (OnlineRetail.csv) in the project directory.

2. Run the customer_segmentation_debug.py script to preprocess the data and train the model:

    python customer_segmentation_debug.py


3. Start the Flask application:

    python app.py


4. Open your web browser and go to http://127.0.0.1:5000/ to use the application.



## Project Structure


Customer_Segmentation_f/
│
├── [app.py]

├── [customer_segmentation_debug.py]

├── [requirements.txt]

├── [kmeans_model.pkl]

├── [scaler.pkl]

├── static/

│   ├── cluster_vs_amount.png

│   ├── cluster_vs_frequency.png

│   ├── cluster_vs_recency.png

├── templates/

│   ├── index.html

│   ├── result.html

└── [README.md]



## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.