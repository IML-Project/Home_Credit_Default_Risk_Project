import pandas as pd 
import numpy as np
import math

# Generic functions, that will universally work for any data frame

# Function used for grouping numerical values by passed statistics
def group_numeric_values(data_frame,
                         data_frame_name,
                         groupby_id = 'SK_ID_CURR', 
                         grouping_statistics = ['count', 'min', 'max', 'mean', 'median', 'sum']):
    
    isTarget = False
    for column in data_frame:
        if column == 'TARGET':
            isTarget = True
            target_cols = data_frame[[groupby_id, 'TARGET']]
            data_frame = data_frame.drop(columns = 'TARGET')
        if column != groupby_id and 'SK_ID' in column:
            data_frame = data_frame.drop(columns = column)
        
    tmp_data_frame = data_frame.select_dtypes(exclude='object')
    
    grouped_data_frame = tmp_data_frame.groupby(groupby_id).agg(grouping_statistics)
    
    new_columns = []
    
    for variable_type in grouped_data_frame.columns.levels[0]:
        if variable_type != groupby_id:
            for stat_type in grouped_data_frame.columns.levels[1]:
                col = f'{data_frame_name}_{variable_type}_{stat_type}'
                new_columns.append(col)
                
    grouped_data_frame.columns = new_columns
    if isTarget:
        grouped_data_frame = grouped_data_frame.merge(target_cols, on = groupby_id, how = 'left')
    return grouped_data_frame


# Function used for grouping object values by passed statistics
def group_object_values(data_frame,
                        data_frame_name,
                        groupby_id = 'SK_ID_CURR',
                        grouping_statistics = ['count', 'mean', 'sum']):

    grouped_data_frame = pd.get_dummies(data_frame.select_dtypes(include='object'))
    
    grouped_data_frame[groupby_id] = data_frame[groupby_id]
    
    grouped_data_frame = grouped_data_frame.groupby(groupby_id).agg(grouping_statistics)
    
    new_columns = []
    
    for variable_type in grouped_data_frame.columns.levels[0]:
        if variable_type != groupby_id:
            for stat_type in grouped_data_frame.columns.levels[1]:
                col = f'{data_frame_name}_{variable_type}_{stat_type}'
                new_columns.append(col)
                
    grouped_data_frame.columns = new_columns
    return grouped_data_frame


# Function used for finding correlations values between features in the given data frame and train set
def data_correlation(train_frame, analysed_data_frame):
    analysed_data_frame['TARGET'] = train_frame['TARGET']
    analysed_data_frame = analysed_data_frame[analysed_data_frame['TARGET'].notna()]
    correlations = analysed_data_frame.corr()['TARGET'].sort_values()
    return correlations


# Function used for convinient column deletion in the data frame
def drop_columns(data_frame, columns):
    for col in data_frame:
        if col in columns:
            data_frame = data_frame.drop(columns = col)
    return data_frame
    
# Function used for deleting columns with a certain correlation (in regards to Target) below certain threshold:
def remove_target_correlated_cols(data_frame, 
                                  special_id = 'SK_ID_CURR', 
                                  threshold = 0.01):
    if 'TARGET' not in data_frame:
        return data_frame
    
    corrs = data_frame.corr()
    corrs = corrs.sort_values('TARGET', ascending = False)
    
    table = pd.DataFrame(corrs['TARGET'])
    cols_to_delete = []
    
    for row in table.index:
        if (abs(table.at[row, 'TARGET']) < threshold or math.isnan(table.at[row, 'TARGET'])) and row != special_id:
            cols_to_delete.append(row)
            
    data_frame = data_frame.drop(columns = cols_to_delete)
    return data_frame
   

# Funtion applying log transformation to given data frame
def log_transform(data_frame, min_mean_value = 1000, groupby_id = 'SK_ID_CURR'):
    for column in data_frame:
        if column == groupby_id:
            continue
        if data_frame[column].mean() > min_mean_value:
            data_frame[f'{column}_log'] = (data_frame[column] + 1).transform(np.log)
            
    return data_frame;

def normalization(data_frame, min_mean_value = 10, groupby_id = 'SK_ID_CURR'):
    for column in data_frame:
        if column == groupby_id:
            continue
        if data_frame[column].mean() > min_mean_value:
            data_frame[f'{column}_norm'] = (data_frame[column] - data_frame[column].min()) / (data_frame[column].max() - data_frame[column].min())
            
    return data_frame;