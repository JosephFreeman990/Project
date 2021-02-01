# -*- coding: utf-8 -*-
# System
from datetime import timedelta, date

# Data Manipulation
import numpy as np
import pandas as pd

# Visualisation
from matplotlib import pyplot as plt
import seaborn as sns

# Plotting Numerical and Categorical Features
def plot_distribution(dataset, columns, cols=5, rows=2, width=20 , height=10, hspace=0.4, wspace=0.1):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    for i, column in enumerate(columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(column)
        #if feature is categorical, to plot countplot
        if dataset.dtypes[column] == np.object:
            sns.countplot(y=dataset[column])
            plt.xticks(rotation=25)
        #if feature is numerical, to plot boxplot
        else:
            sns.boxplot(dataset[column])
            plt.xticks(rotation=25)

def return_index(dataset, column, value, criteria):
    if criteria == 'equal':
        dataset_index = dataset.loc[dataset[column] == value].index
    elif criteria == 'more':
        dataset_index = dataset.loc[dataset[column] > value].index
    elif criteria == 'less':
        dataset_index = dataset.loc[dataset[column] < value].index
    return dataset_index

def firstlast_datehour(dataset, datecolumn='date', hrcolumn='hr'):
    first_date = dataset.loc[0, datecolumn].date()
    first_hour = dataset.loc[0, hrcolumn]
    last_date = dataset.loc[dataset.shape[0] - 1, datecolumn].date()
    last_hour =dataset.loc[dataset.shape[0] - 1, hrcolumn]
    return first_date, first_hour, last_date, last_hour

def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def full_datehour(first_date, first_hour, last_date, last_hour, datecolumn='date', hrcolumn='hr', dateformat='%Y-%m-%d'):
    complete_datehour = []
    for single_date in daterange(first_date, last_date+timedelta(days=1)):
        for single_hour in range (0, 24):
            single_datehour = [single_date, single_hour]
            complete_datehour.append(single_datehour)
    complete_datehour = pd.DataFrame(complete_datehour, columns=[datecolumn, hrcolumn])
    # Ensure that 'date' column is in the correct format
    complete_datehour[datecolumn] =  pd.to_datetime(complete_datehour[datecolumn], format=dateformat)
    complete_datehour = complete_datehour.drop(complete_datehour.loc[(complete_datehour[datecolumn]==pd.Timestamp(first_date)) & (complete_datehour[hrcolumn]<first_hour)].index)
    complete_datehour = complete_datehour.drop(complete_datehour.loc[(complete_datehour[datecolumn]==pd.Timestamp(last_date)) & (complete_datehour[hrcolumn]>last_hour)].index)

    return complete_datehour

def return_missing_datehour(full_datehour, dataset, datecolumn='date', hrcolumn='hr'):
    for index, row in dataset[[datecolumn, hrcolumn]].iterrows():
        full_datehour = full_datehour.drop(full_datehour.loc[(full_datehour[datecolumn]==row[datecolumn]) & (full_datehour[hrcolumn]==row[hrcolumn])].index)
    return full_datehour

def add_features_datetime_YMD (dataset, column='date', feature_name=['year', 'month', 'day']):
    # Create numpy arrays of zeros/empty string, we will replace the values subsequently
    dt_year = np.ones(len(dataset[column]))
    dt_month = np.ones(len(dataset[column]))
    dt_day = []
    
    # Extract the relevant feature from column and update the features to dataset
    for feature in feature_name:
        if feature == 'year':
            for index, row in dataset[column].to_frame().iterrows():
                dt_year[index] = row[column].year
            dt_year = pd.DataFrame(data=dt_year, columns=['year'], dtype=np.int64)
            dataset =  pd.concat([dataset, dt_year], axis=1, sort=False)
        elif feature == 'month':
            for index, row in dataset[column].to_frame().iterrows():
                dt_month[index] = row[column].month
            dt_month = pd.DataFrame(data=dt_month, columns=['month'], dtype=np.int64)
            dataset =  pd.concat([dataset, dt_month], axis=1, sort=False)
        elif feature == 'day':
            for index, row in dataset[column].to_frame().iterrows():
                dt_day.append(row[column].strftime('%w')) #%A
            dt_day = pd.DataFrame(data=dt_day, columns=['day'], dtype=np.int64) #str
            dataset =  pd.concat([dataset, dt_day], axis=1, sort=False)
            #dataset['day_of_the_week'] = np.where(dataset['day_of_the_week'] == ('Saturday' or 'Sunday'), 'Weekend', 'Weekday')
    
    # Drop column as relevant features were already extracted
    dataset = dataset.drop([column], axis = 1)
            
    return dataset

def cyclical_features(dataset, columnheaders=['hr', 'day','month']):
    for header in columnheaders:
        if header == 'hr':
            dataset['hr_sin'] = np.sin(dataset.hr*(2.*np.pi/24))
            dataset['hr_cos'] = np.cos(dataset.hr*(2.*np.pi/24))
            dataset = dataset.drop(['hr'], axis=1)
        elif header == 'month':
            dataset['month_sin'] = np.sin((dataset.month-1)*(2.*np.pi/12))
            dataset['month_cos'] = np.cos((dataset.month-1)*(2.*np.pi/12))
            dataset = dataset.drop(['month'], axis=1)
        elif header == 'day':
            dataset['day_sin'] = np.sin(dataset.day*(2.*np.pi/7))
            dataset['day_cos'] = np.cos(dataset.day*(2.*np.pi/7))
            dataset = dataset.drop(['day'], axis=1)
        
        else:
            print('column headers not recognised, please use either "hr" or "month"')
    return dataset

def plot_correlation(X, y, X_columns, y_columns, rows=2, cols=2, width=40 , height=40, hspace=0.4, wspace=0.9):
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width,height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    
    num_plots = 0
    for j, xcolumn in enumerate(X_columns):
        for i, ycolumn in enumerate(y_columns):
            ax = fig.add_subplot(rows, cols, num_plots+1)
            #ax.set_title(xcolumn)
            #if feature is numerical, to plot scatterplot
            if X.dtypes[xcolumn] == 'float' or 'int':
                sns.scatterplot(x=X[xcolumn], y=y[ycolumn], ax=ax);
                plt.xticks(rotation=25)
                '''
                sns.countplot(y=dataset[column])
                plt.xticks(rotation=25)
                '''
            #if feature is numerical, to plot boxplot
            else:
                sns.boxplot(x=xcolumn, y=ycolumn, data=pd.concat([X,y], axis=1)); 
                plt.xticks(rotation=25)
            num_plots += 1
            
def eda_preprocess():
    print('Importing data...')
    # Importing the dataset
    data_url = 'https://aisgaiap.blob.core.windows.net/aiap6-assessment-data/scooter_rental_data.csv'
    dataset = pd.read_csv(data_url)
    print('Done')
       
    print('Conducting data preprocessing...')
    print('- Working on numerical features...')
    # Replace data that has the value zero for 'relative-humidity' with mean
    median_relativehumidity = dataset['relative-humidity'].median(skipna=True)
    dataset = dataset.replace({'relative-humidity': {0: median_relativehumidity}})
    # Drop the 'feels-like-temperature' feature
    dataset = dataset.drop(['windspeed'], axis = 1)
    # Replace data that has the value zero for 'psi' with mean
    median_psi = dataset['psi'].median(skipna=True)
    dataset = dataset.replace({'psi': {0: median_psi}})
    # Replace data that has negative values for 'guest-users' with positive values
    dataset['guest-users'] = dataset['guest-users'].abs()
    # Replace data that has the value zero for "relative-humidity" with mean
    dataset['registered-users'] = dataset['registered-users'].abs()
    # Drop the 'feels-like-temperature' feature
    dataset = dataset.drop(['feels-like-temperature'], axis = 1)
    # Drop the 'psi' feature
    dataset = dataset.drop(['psi'], axis = 1)
    
    print('- Working on categorical features...')
    # Convert uppercase strings to lowercase
    dataset['weather'] = dataset['weather'].str.lower()
    dataset.loc[dataset['weather'].str.contains('lear'), 'weather'] = 'clear'
    dataset.loc[dataset['weather'].str.contains('loudy'), 'weather'] = 'cloudy'
    
    print('- Working on datetime features...')
    # Convert 'date' to datetime format='%Y-%m-%d'
    dataset['date'] =  pd.to_datetime(dataset['date'], format='%Y-%m-%d')
    # Drop the duplicated entries
    dataset = dataset.drop_duplicates(keep='first')
    # Create 3 new features, 'year', 'month' and 'day_of_the_week' to replace 'date'. 
    # These features seperately will be more informative in predicting total number of active users.
    dataset = add_features_datetime_YMD (dataset, column='date', feature_name=['year', 'month', 'day'])
    # Create cyclical features for 'hr' and 'month'
    dataset = cyclical_features(dataset, columnheaders=['hr', 'month', 'day'])
    print('Done')
    return dataset
    