# 🛒 Supermarket Sales Analysis and Prediction

## 📌 Project Overview

This is a Machine Learning minor project for analyzing and predicting supermarket sales.

The project uses a Supermarket Sales dataset to perform data analysis, visualization, preprocessing, machine learning model training, and sales prediction.

A Streamlit web application is created to provide an easy-to-use interface for viewing the analysis and making sales predictions.

---

## 🎯 Objectives

- Analyze supermarket sales data
- Perform data preprocessing
- Create useful features from Date and Time
- Visualize sales information
- Train multiple Machine Learning models
- Compare model performance
- Select the best-performing model
- Predict sales using new transaction details
- Create an interactive Streamlit application

---

## 📊 Dataset

The dataset contains supermarket transaction information such as:

- Invoice ID
- Branch
- City
- Customer Type
- Gender
- Product Line
- Unit Price
- Quantity
- Tax
- Sales
- Date
- Time
- Payment
- COGS
- Gross Income
- Rating

---

## 🤖 Machine Learning Algorithms

Three regression algorithms are used in this project:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

The models are evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

## 🏆 Model Result

The models were compared based on their R² Score.

### Results

| Model | R² Score |
|---|---:|
| Linear Regression | 0.5014 |
| Decision Tree | 0.0725 |
| Random Forest | 0.4491 |

### Best Model

**Linear Regression**

R² Score: **0.5014**

The best model is saved as:

```text
model.pkl
