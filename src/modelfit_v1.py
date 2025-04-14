from sklearn.metrics import r2_score
import scipy
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

from IPython.display import display

def char_after_model(y,y_pred) :
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, color='royalblue', alpha=0.7)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')  # 45-degree line
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def modelfit(data,var_trans, target):
    X = data[var_trans]
    y = data[target]
    X_const = sm.add_constant(X)

    # Fit model
    model = sm.OLS(y, X_const).fit()
    y_pred = model.predict(X_const)
    # Summary dataframe
    summary_df = model.summary2().tables[1].reset_index()
    summary_df.rename(columns={'index': 'Variable', 'Coef.': 'Coefficient', 'P>|t|': 'p_value'}, inplace=True)

    # Calculate std dev for each variable
    std_devs = X_const.std().to_dict()

    # Contribution = coef * std_dev / total sum
    summary_df['StdDev'] = summary_df['Variable'].map(std_devs)
    summary_df['Raw_Impact'] = summary_df['Coefficient'] * summary_df['StdDev']
    total_impact = summary_df['Raw_Impact'].sum()
    summary_df['Contribution'] = abs((summary_df['Raw_Impact'] / total_impact)*100)
    
    
    # VIF calculation
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i+1) for i in range(len(X.columns))]
    summary_df['VIF'] = vif_data["VIF"].round(2)
    
    
    
    print(model.summary2().tables[0])
#     print(summary_df)
    char_after_model(y,y_pred)
    return model.summary2().tables[0], summary_df




