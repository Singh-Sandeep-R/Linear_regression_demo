
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import scipy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


def missing_value_treatment(data,target):

    impute_mv_cont = {}
    impute_mv_cat = {}
    var_list = list(data.columns)
    var_list.remove(target)
    var_list = [var for var in var_list if data[var].isna().sum()!=0]
    print(var_list)
    for var in var_list :
        if pd.api.types.is_numeric_dtype(data[var]):
            data['bin'] = pd.qcut(data[var],q=10,duplicates = 'drop')
            bin_mean = data.groupby(['bin']).agg({var:'mean',target:'mean'}).reset_index()
            null_mean = data[data[var].isna()][[var,target]].mean()
            mv_target = null_mean[target]
            null_mean = pd.DataFrame({'bin':'Missing',var :[None], target:[null_mean[target]] })

            bins = bin_mean.copy()
            bins['close_bin'] = abs(bins[target]-mv_target)
            impute_mv_cont[var] = bins.loc[bins['close_bin'].idxmin(),var]
            bins.drop(columns = ['close_bin'],inplace= True)
            bins = pd.concat([bin_mean,null_mean],ignore_index = True)
        else :
            bin_mean = data.groupby([var]).agg({target:'mean'}).reset_index()
            null_mean = data[data[var].isna()][[var,target]].mean()
            mv_target = null_mean[target]
            null_mean = pd.DataFrame({'bin':'Missing',var :[None], target:[null_mean[target]] })

            bins = bin_mean.copy()
            bins['close_bin'] = abs(bins[target]-mv_target)
            impute_mv_cat[var] = bins.loc[bins['close_bin'].idxmin(),var]
            bins.drop(columns = ['close_bin'],inplace= True)
            bins = pd.concat([bin_mean,null_mean],ignore_index = True)
    return impute_mv_cont,impute_mv_cat



def r2 (target, df,num_var,impute_mv_cont):
    r2_data = pd.DataFrame(columns = ['var','r2'])
    for var in num_var:        
        if var in impute_mv_cont:
            df[var] = df[var].fillna(impute_mv_cont[var])
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(df[var],df[target])
        var_r2 = pd.Series({'var':var,'r2':r_value})
        r2_data.loc[len(r2_data)] = var_r2

    return r2_data.sort_values(by= 'r2',ascending= False)



def transformation_iterator(df,var,target,floor_list,cap_list,transformation_list,impute_mv_cont):
    data = df.copy()
    if var in impute_mv_cont:
        print(f"{var} null value is imputed with :",impute_mv_cont[var])
        data[var] = data[var].fillna(impute_mv_cont[var])
    r2_data_all_trans = pd.DataFrame(columns = [var,'floor','cap','trans','R2'])
#         slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(data[var],data[target])
#         print(r_value)
    for trans in transformation_list:
        for cap in cap_list:
            for floor in floor_list :
                floor_value = np.percentile(data[var],floor)
                cap_value = np.percentile(data[var],cap)
                data[var+'_'] = data[var].clip(lower = floor_value,upper=  cap_value)



                if trans=='None':
                    data[var+'__'] = data[var+'_'].copy()
                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(data[var+'__'],data[target])
#                         print(r_value)
                elif trans=='sqaure':
                    data[var+'__'] = data[var+'_']**2
                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(data[var+'__'],data[target])
                elif trans=='cube':
                    data[var+'__'] = data[var+'_']**3
                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(data[var+'__'],data[target])
                elif trans=='sqrt':
                    data[var+'__'] = data[var+'_']**0.5
                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(data[var+'__'],data[target])
                else :
                    data[var+'__'] = np.log(data[var+'_']+0.00001)
                    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(data[var+'__'],data[target])
                results = pd.Series({
                    var :var,
                    'floor':floor,
                    'cap':cap,
                    'trans' :trans,
                    'R2':r_value})
                r2_data_all_trans.loc[len(r2_data_all_trans)] = results

    return r2_data_all_trans.sort_values(by=['R2'],ascending= False).reset_index(drop = True)
def chart (bin_mean,var,target):    
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression


    x = bin_mean[var].values.reshape(-1,1)
    y = bin_mean[target].values
    model = LinearRegression()
    model.fit(x,y)
    y_pred = model.predict(x)

    plt.figure(figsize=(8,5))
    plt.scatter(bin_mean[var], bin_mean[target], color = 'blue', label= 'Data points')
    plt.plot(bin_mean[var], y_pred , color = 'red', linewidth = 2, label = 'Trend Line')
    plt.xlabel(var)
    plt.ylabel(target)
    plt.title('Scatter Plot with Trend Line')
    plt.legend()
    # plt.grid(True)
    plt.tight_layout()
    plt.show()

def trasnformation_apply(df,impute_mv_cont,var,r2_data_all_trans,nth_trans,target):
    data = df.copy()
    if var in impute_mv_cont :
        data[var] = data[var].fillna(impute_mv_cont[var])
    unique_value = data[var].nunique()
    if unique_value<=20 :
        data['bin'] = pd.cut(data[var], bins =unique_value )
    else :
        data['bin'] = pd.qcut(data[var],q=20,duplicates = 'drop')

    bin_mean = pd.DataFrame(data.groupby(['bin']).agg({var:'mean',target:'mean'})).reset_index()
    bin_mean = bin_mean[bin_mean[var].notna()]
    print("-------Before Any Transformation------------")
    print(bin_mean)
    chart(bin_mean,var,target)


    print("--------After Applying Transformation-----------\n")
    floor = r2_data_all_trans.loc[nth_trans]['floor'] 
    floor_value = np.percentile(data[var],floor)
    cap = r2_data_all_trans.loc[nth_trans]['cap']
    cap_value = np.percentile(data[var],cap)

    data[var+'_'] = data[var].clip(lower = floor_value,upper=  cap_value)
    print(f"{var} floored at {floor} : {floor_value}")
    print(f"{var} cap at {cap} : {cap_value}")
#         print('Transforamtion is as below:\n')
#         print(f"x = df[{var}].copy()\n")
#         print(f"x x= x.clip(lower = f{floor_value}, upper = {cap_value}")
#         print(f"df[{var}_]= x.copy()")

    trans = r2_data_all_trans.loc[nth_trans]['trans']
    if trans=='None':
        data[var+'__T'] = data[var+'_'].copy()
        print(f"Transformation after cap and floor is {trans}")

#         slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(data[var+'__'],data[target])
#         #                         print(r_value)
    elif trans=='sqaure':
        data[var+'__T'] = data[var+'_']**2

#         slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(data[var+'__'],data[target])
    elif trans=='cube':
        data[var+'__T'] = data[var+'_']**3
#         slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(data[var+'__'],data[target])
    elif trans=='sqrt':
        data[var+'__T'] = data[var+'_']**0.5
#         slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(data[var+'__'],data[target])
    else :
        data[var+'__T'] = np.log(data[var+'_']+0.000001)
#         slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(data[var+'__'],data[target])
    print(f"Transformation after cap and floor is {trans}")    
    print(f"Transformed variable is {var}__T")
    var = var+'__T'
    data['bin'] = pd.qcut(data[var],q=20,duplicates = 'drop')
    bin_mean = pd.DataFrame(data.groupby(['bin']).agg({var:'mean',target:'mean'})).reset_index()
    chart(bin_mean,var,target)
    dataout = data.copy()
    return dataout





def cat_var_importance(df,categorical_vars,target,impute_mv_cat):

    y = df[target]

    r2_scores = pd.DataFrame(columns = ['var','R2'])

    for var in categorical_vars:
        if var in impute_mv_cat:
#             print(var)
            df[var] = df[var].fillna(impute_mv_cat[var])
        # Select this variable only
        X = df[[var]]

        # Pipeline: OneHotEncode + Linear Regression
        pipeline = Pipeline([
            ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore')),
            ('model', LinearRegression())
        ])

        # Use cross_val_score or just fit and score
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)
        r2 = r2_score(y, y_pred)
        var_r2 = pd.Series({'var':var,'R2':r2})
        r2_scores.loc[len(r2_scores)] = var_r2
    return r2_scores.sort_values(by=['R2'],ascending= False)

def cat_transformation(data_4,cat_var_list,impute_mv_cat):
    data = data_4.copy()
    for var in cat_var_list:
        if var in impute_mv_cat:
#             print(var)
            data[var] = data[var].fillna(impute_mv_cat[var])
        encoder = OneHotEncoder(drop='first', sparse=False, dtype=int)

        # Fit and transform
        encoded_array = encoder.fit_transform(data[[var]])

        encoded_cols = encoder.get_feature_names_out([var])


        # Create DataFrame from encoded array
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols)

        data = data.join(encoded_df)
    return data