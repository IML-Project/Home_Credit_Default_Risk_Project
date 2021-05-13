import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# General functions, which should always work for pretty much any given data frame
# However, as it was tested in case of filling missing values, the function could not process extremely big data sets, probably due to their size

# Function used for filling NaN values in tha data frame
# When the parameter mean is True, we fill the misisng values with the column mean, otherwise we fill those with zeros
def fill_missing_values(data_frame, mean = True):  
    if mean == True:
        for column in data_frame:
            if type(data_frame[column]) is not object:
                data_frame[column].fillna(value = data_frame[column].mean(), inplace = True)
                
    else: 
        for column in data_frame:
            if type(data_frame[column]) is not object:
                data_frame[column].fillna(value = 0, inplace = True)
                
    return data_frame

# Function used for dropping missing columns and printing the information of (some of) the columns deleted
def drop_missing_columns(data_frame, threshold = 70, print_info = True):
        data_frame_missing_values = pd.DataFrame(data_frame.isnull().sum())
        data_frame_missing_values['percent'] = 100 * data_frame_missing_values[0] / len(data_frame_missing_values)
        
        missing_cols = list(data_frame_missing_values.index[data_frame_missing_values['percent'] > threshold])
        
        if print_info == True:
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

