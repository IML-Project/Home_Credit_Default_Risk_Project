import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def fill_missing_values(data_frame, mean = True):  
    if mean == True:
        for column in data_frame:
            if type(data_frame[column]) is not object:
                data_frame[column].fillna(data_frame[column].mean(), inplace = True)
                
    else: 
        for column in data_frame:
            if type(data_frame[column]) is not object:
                data_frame[column].fillna(0, inplace = True)
                
    return data_frame

def drop_missing_columns(data_frame, threshold = 70, print_info = True):
        df_missing = pd.DataFrame(data_frame.isnull().sum())
        df_missing['percent'] = 100 * df_missing[0] / len(df_missing)
        
        missing_cols = list(df_missing.index[df_missing['percent'] > threshold])
        
        if print_info:
            print(f'There are {len(missing_cols)} with greater than {threshold} missing values')
            if len(missing_cols) > 10:
                print('10 exemplary incomplete columns to be deleted: ')
                print(missing_cols[0:10])
            elif len(missing_cols) == 0:
                print("No columns will be deleted")
            else:
                print('Incomplete columns: ')
                print(missing_cols)
                
        data_frame = data_frame.drop(columns = missing_cols)
        
        return data_frame
    
