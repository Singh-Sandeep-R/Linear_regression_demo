import pandas as pd
def vartransformation(df,impute_mv_cont,impute_mv_cat):
    data = df.copy()
    for var in data.columns:
        if var in impute_mv_cont:
            data[var] = data[var].fillna(impute_mv_cont[var])
        if var in impute_mv_cat:
            data[var] = data[var].fillna(impute_mv_cat[var])
    ##### ------LotArea----------####
    var = 'LotArea'
    data[var] = data[var].clip(lower= 5000.0, upper =14381.70000000001 )
    data[var+'__T'] = data[var]**2

    #######--------OverallQual__T------------###
    var = 'OverallQual'
    data[var] = data[var].clip(lower= 3, upper =10 )
    data[var+'__T'] = data[var].copy()
  
    #####--------BsmtFinSF1__T------------####
    var = 'BsmtFinSF1'
    data[var] = data[var].clip(lower= 0, upper =1572.41 )
    data[var+'__T'] = data[var].copy()
    
    
    
    ##### GrLivArea--------#######
    var = 'GrLivArea'
    data[var] = data[var].clip(lower= 692.18, upper =3123.4800000000023 )
    data[var+'__T'] = data[var].copy()

    ##### TotalBsmtSF--------#######
    var = 'TotalBsmtSF'
    data[var] = data[var].clip(lower= 519.3000000000001, upper =2155.05 )
    data[var+'__T'] = data[var].copy()

    ##### Fullbath ####
    var = 'FullBath'
    df_encoded = pd.get_dummies(data[var],prefix = var, dtype=int)
    data = data.join(df_encoded)        
    ##### ExterQual ####
    var = 'ExterQual'
    df_encoded = pd.get_dummies(data[var],prefix = var, dtype=int)
    data = data.join(df_encoded)
    
        ##### KitchenQual ####
    var = 'KitchenQual'
    df_encoded = pd.get_dummies(data[var],prefix = var, dtype=int)
    data = data.join(df_encoded)
      ##### BsmtQual ####
    var = 'BsmtQual'
    df_encoded = pd.get_dummies(data[var],prefix = var, dtype=int)
    data = data.join(df_encoded)
  ##### GarageType ####
    var = 'GarageType'
    df_encoded = pd.get_dummies(data[var],prefix = var, dtype=int)
    data = data.join(df_encoded)
    
    return data