# Online Fraud Transaction Detection - ML Capstone
 
## Project Overview
This project is an end-to-end machine learning system for detecting fraudulent online transactions. It features a Random Forest classification model handling extreme class imbalance using SMOTE, exposed via a REST API using FastAPI.
 
## Folder Structure
* `api/`: FastAPI application and test scripts
* `data/`: Raw and processed datasets (Ignored in Git)
* `models/`: Serialized trained models (Ignored in Git)
* `src/`: Model training and data preprocessing scripts
* `.github/workflows/`: CI pipeline configuration
 
## Setup Instructions
1. Clone the repository.
2. Ensure you have Python 3.12 installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
 