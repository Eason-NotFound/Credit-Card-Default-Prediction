# Credit Default Prediction with Machine Learning

## Overview

This project focuses on **credit default prediction** using machine learning techniques, based on the **American Express Default Prediction** Kaggle competition.  
The objective is to predict whether a customer will default in the future using large-scale, highly imbalanced time-series credit data.

The project is implemented in **Python** and emphasizes data sampling, feature engineering for time-series data, and model comparison.

---

## Research Objective

The main goals of this project are to:
- Handle large-scale and imbalanced credit default data
- Perform effective feature engineering on customer-level time-series records
- Compare multiple machine learning models for default prediction
- Evaluate model performance using appropriate classification metrics (F1-score)

---

## Dataset

- Source: **American Express – Default Prediction (Kaggle)**
- Raw training data size: ~16.39 GB
- Highly imbalanced default vs non-default classes
- Multiple records per customer over time

Competition link:  
https://www.kaggle.com/competitions/amex-default-prediction/

---

## Methodology

The notebook `Default Prediction.ipynb` follows a structured 9-step pipeline:

### Step 1: Data Sampling
- Due to data size and imbalance, customers who defaulted and did not default were **equally sampled**
- Dataset size reduced to **2 million rows**
- Final class distribution: ~51% non-default, ~48% default

### Step 2: Data Loading
- Sampled data loaded and converted into DataFrame format

### Step 3: Data Cleaning
- Columns with more than **40% missing values** removed
- Important features (e.g. time-series feature `S_2`) properly formatted

### Step 4: Feature Engineering
- Sorted records by **customer ID and time**
- Aggregated time-series records into **customer-level features**
- Generated summary statistics for numerical features to reduce noise and redundancy

### Step 5: Exploratory Data Analysis (EDA)
- Visualized missing value patterns
- Inspected target variable distribution
- Ensured data quality before modeling

### Step 6: Test Data Loading
- Test data loaded and converted into parquet format for efficiency

### Step 7: Test Data Preparation
- Applied the same cleaning and feature engineering pipeline to test data

### Step 8: Model Training and Evaluation
Four models were trained and compared:
- Logistic Regression
- LightGBM
- Tuned LightGBM
- XGBoost

**Best model:** Tuned LightGBM  
- **F1 Score:** ~0.6747

### Step 9: Final Prediction
- Used the tuned LightGBM model to generate default predictions on test data
- Produced final submission scores

---

## Results

- Best achieved **F1 score ≈ 0.67**
- Tuned LightGBM outperformed baseline models
- Results indicate strong predictive capability despite data complexity and imbalance

---

## Technologies Used

- Python
- Pandas / NumPy
- LightGBM
- XGBoost
- Scikit-learn
- Parquet data format
- Jupyter Notebook

## Future Improvements

- Train models on larger subsets or full dataset
- Further hyperparameter tuning for LightGBM
- Explore advanced time-series aggregation techniques
- Experiment with ensemble stacking methods

---

## Author

**Yichen Qu**

This project was completed as part of the **SC4000** course and Kaggle competition.

---

## Data Reference
https://www.kaggle.com/competitions/amex-default-prediction/
