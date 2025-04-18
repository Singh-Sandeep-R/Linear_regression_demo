# Linear Regression Modeling Pipeline

This repository contains a blueprint for building a robust **Linear Regression Model** from scratch using Python. It includes all essential preprocessing steps such as missing value treatment, variable reduction, feature transformation, and finally model training and evaluation.

---

## 🔍 Project Objective

To create a modular and reusable pipeline for training a linear regression model that handles:

- Categorical and numerical variables
- Missing values
- High-cardinality or low-importance variables
- Feature transformations
- Multicollinearity via VIF
- Final regression with coefficient interpretation and contribution analysis

---

## 🛠️ Pipeline Steps

1. **Missing Value Treatment**
   - Numerical variables: imputed using mean/median
   - Categorical variables: imputed using mode or predefined rules

2. **Variable Reduction**
   - For categorical variables: assessed importance using R² from one-variable models
   - For numerical variables: correlation, low variance, and VIF checks

3. **Feature Transformation**
   - One-hot encoding for categorical variables (with drop='first')
   - Standardization where applicable
   - Target encoding (optional, based on business case)

4. **Model Building**
   - Fit using `statsmodels.OLS` for access to p-values and summary
   - Outputs include:
     - Coefficients
     - P-values
     - VIF for multicollinearity
     - Contribution of each variable (standardized coefficient weights)

5. **Model Evaluation**
   - R² and adjusted R²
   - Actual vs Predicted plots
   - Summary table with:  
     - Variable  
     - Coefficient  
     - Std. Dev  
     - P-Value  
     - Contribution %  
     - VIF
6. **Run the model fit with Normal Equation**
7. **show how gradient descent work in Linear Regression**

---
