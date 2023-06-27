import sys
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print(time.strftime('%Y/%m/%d %H:%M'))
print('OS:', sys.platform)
print('CPU Cores:', os.cpu_count())
print('Python:', sys.version)
print('NumPy:', np.__version__)
print('Pandas:', pd.__version__)

# Formatting for seaborn plots
sns.set_context('notebook', font_scale=1.1)
sns.set_style('ticks')

# Displays all dataframe columns
pd.set_option('display.max_columns', None)

#################################################################################################################
##### Exploratory Data Analysis

# Quick EDA report on dataframe
import pandas_profiling
profile = pandas_profiling.ProfileReport(df)
profile.get_rejected_variables(threshold=0.9)  # Rejected variables w/ high correlation
profile.to_file(outputfile='/tmp/myoutputfile.html')  # Saving report as a file

#################################################################################################################
##### Missing Values

# Returning a dict with the percentage of missing values per column
def percent_missing(dataframe: pd.DataFrame) -> dict:
    '''
    Returns a dict with the percentage of missing values for each column in a dataframe
    '''
    # Summing the number of missing values per column and then dividing by the total number of rows
    sumMissing = dataframe.isnull().values.sum(axis=0)
    pctMissing = sumMissing / dataframe.shape[0]
    pct_missing_by_col = {}
    
    # Adding any columns w/ missing vlaues to the dictionary
    if sumMissing.sum() != 0:
        for idx, col in enumerate(dataframe.columns):
            if sumMissing[idx] > 0:
                pct_missing_by_col[col] = pctMissing[idx]
    
    return pct_missing_by_col
        

# Plotting missing values
import missingno as msno  # Visualizes missing values
msno.matrix(df)
msno.heatmap(df)  # Co-occurrence of missing values

# Drop missing values
df.dropna(how='any', thresh=None, inplace=True)  # Also 'all' for how, and thresh is an int

# Filling missing values with columnar means
df.fillna(value=df.mean(), inplace=True)

# Filling missing values with interpolation
df.fillna(method='ffill', inplace=True)  #'backfill' for interpolating the other direction

# Filling missing values with a predictive model
def predict_missing_values(data, column, correlationThresh: float = 0.5, cross_validations: int = 3):
    '''
    Fills missing values using a random forest model on highly correlated columns
    Returns a series of the column with missing values filled
    
    TODO: - Add the option to specify columns to use for predictions
           - Look into other options for handling missing predictors
    '''
    from sklearn.model_selection import cross_val_score
    from sklearn import ensemble
    
    # Printing number of percentage values missing
    pctMissing = data[column].isnull().values.sum() / data.shape[0]
    print('Predicting missing values for {0}\n'.format(column))
    print('Percentage missing: {0:.2f}%'.format(pctMissing*100))
    
    # Multi-threading if the dataset is a size where doing so is beneficial
    if data.shape[0] < 100000:
        num_cores = 1  # Single-thread
    else:
        num_cores = -1  # All available cores
    
    # Instantiating the model
    # Picking a classification model if the number of unique labels are 25 or under
    num_unique_values = len(np.unique(data[column]))
    if num_unique_values > 25 or data[column].dtype != 'category':
        print('Variable is continuous')
        rfImputer = ensemble.RandomForestRegressor(n_estimators=100, n_jobs=num_cores)
    else:
        print('Variable is categorical with {0} classes').format(num_unique_values)
        rfImputer = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=num_cores)
    
    # Calculating the highly correlated columns to use for the model
    highlyCorrelated = abs(data.corr()[column]) >= correlationThresh
    
    # Exiting the function if there are not any highly correlated columns found
    if highlyCorrelated.sum() < 2:  # Will always be 1 because of correlation with self
        print('Error: No correlated variables found. Re-try with less a lower correlation threshold')
        return  # Exits the function
    highlyCorrelated = data[data.columns[highlyCorrelated]]
    highlyCorrelated = highlyCorrelated.dropna(how='any')  # Drops any missing records
    print('Using {0} highly correlated features for predictions\n'.format(highlyCorrelated.shape[1]))
    
    # Creating the X/y objects to use for the
    y = highlyCorrelated[column]
    X = highlyCorrelated.drop(column, axis=1)
    
    # Evaluating the effectiveness of the model
    cvScore = np.mean(cross_val_score(rfImputer, X, y, cv=cross_validations, n_jobs=num_cores))
    print('Cross Validation Score:', cvScore)

    # Fitting the model for predictions and displaying initial results
    rfImputer.fit(X, y)
    if num_unique_values > 25 or data[column].dtype.name != 'category':
        print('R^2:', rfImputer.score(X, y))
    else:
        print('Accuracy:', rfImputer.score(X, y))
    
    # Re-filtering the dataset down to highly correlated columns
    # Filling NA predictors w/ columnar mean instead of removing
    X_missing = data[highlyCorrelated.columns]
    X_missing = X_missing.drop(column, axis=1)
    X_missing = X_missing.fillna(X_missing.mean())
    
    # Filtering to rows with missing values before generating predictions
    missingIndexes = data[data[column].isnull()].index
    X_missing = X_missing.iloc[missingIndexes]
    
    # Predicting the missing values
    predictions = rfImputer.predict(X_missing)
    
    # Preventing overwriting of original dataframe
    data = data.copy()

    # Looping through the missing values and replacing with predictions
    for i, idx in enumerate(missingIndexes):
        data.set_value(idx, column, predictions[i])
    
    return data[column]

    
df[colName] = predict_missing_values(df, colName)

#################################################################################################################
##### Outliers

# TODO: - Add docstrings to functions
#       - Add other functions (GESD, local outlier factor, isolation forests, etc.)

# Detecting outliers with Interquartile Range (IQR)
# Note: The function in its current form is taken from Chris Albon's Machine Learning with Python Cookbook
def iqr_indices_of_outliers(X: np.ndarray) -> np.ndarray:
    '''
    Detects outliers using the interquartile range (IQR) method
    
    Input: An array of a variable to detect outliers for
    Output: An array with indices of detected outliers
    '''
    q1, q3 = np.percentile(X, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    outlier_indices = np.where((X > upper_bound) | (X < lower_bound))
    return outlier_indices


# Detecting outliers with Z scores
def z_score_indices_of_outliers(X: np.ndarray, threshold: float = 3) -> np.ndarray:
    '''
    Detects outliers using the Z score method method
    
    Input: - X: An array of a variable to detect outliers for
           - threshold: The number of standard deviations from the mean
                        to be considered an outlier
                        
    Output: An array with indices of detected outliers
    '''
    X_mean = np.mean(X)
    X_stdev = np.std(X)
    z_scores = [(y - X_mean) / X_stdev for y in X]
    outlier_indices = np.where(np.abs(z_scores) > threshold)
    return outlier_indices


# Detecting outliers with the Elliptical Envelope method
def ellipses_indices_of_outliers(X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
    '''
    Detects outliers using the elliptical envelope method
    
    Input: An array of all variables to detect outliers for
    Output: An array with indices of detected outliers
    '''
    from sklearn.covariance import EllipticEnvelope
    
    # Copying to prevent changes to the input array
    X = X.copy()
    
    # Dropping categorical columns
    non_categorical = []
    for feature in range(X.shape[1]):
        num_unique_values = len(np.unique(X[:, feature]))
        if num_unique_values > 30:
            non_categorical.append(feature)
    X = X[:, non_categorical]  # Subsetting to columns without categorical indexes

    # Testing if there are an adequate number of features
    if X.shape[0] < X.shape[1] ** 2.:
        print('Will not perform well. Reduce the dimensionality and try again.')
        return
    
    # Creating and fitting the detector
    outlier_detector = EllipticEnvelope(contamination=contamination)
    outlier_detector.fit(X)
    
    # Predicting outliers and outputting an array with 1 if it is an outlier
    outliers = outlier_detector.predict(X)
    outlier_indices = np.where(outliers == -1)
    return outlier_indices


# Detecting outliers with the Isolation Forest method
def isolation_forest_indices_of_outliers(X, contamination='auto', n_estimators=100):
    '''
    Detects outliers using the isolation forest method
    
    Inputs:
        - X (array or data frame): Non-categorical variables to detect outliers for
        - Contamination (float or 'auto'): The percentage of outliers
        - n_estimators (int): The number of treess to use in the isolation forest
    Output: An array with indices of detected outliers
    '''
    from sklearn.ensemble import IsolationForest
    
    # Copying to prevent changes to the input array
    X = X.copy()

    # Creating and fitting the detector
    outlier_detector = IsolationForest(contamination=contamination,
                                       n_estimators=n_estimators,
                                       behaviour='new',
                                       n_jobs=-1)
    outlier_detector.fit(X)
    
    # Predicting outliers and outputting an array with 1 if it is an outlier
    outliers = outlier_detector.predict(X)
    outlier_indices = np.where(outliers == -1)
    return outlier_indices

outlier_indexes_forest = helper.isolation_forest_indices_of_outliers(X.select_dtypes(exclude='category'),
                                                              contamination='auto')
print('Outliers detected: {0}'.format(len(outlier_indexes_forest[0])))


# Detecting outliers with the One Class SVM method
def one_class_svm_indices_of_outliers(X):
    '''
    Detects outliers using the one class SVM method
    
    Input: An array of all variables to detect outliers for
    Output: An array with indices of detected outliers
    '''
    from sklearn.svm import OneClassSVM
    
    # Copying to prevent changes to the input array
    X = X.copy()
    
    # Dropping categorical columns
    non_categorical = []
    for feature in range(X.shape[1]):
        num_unique_values = len(np.unique(X[:, feature]))
        if num_unique_values > 30:
            non_categorical.append(feature)
    X = X[:, non_categorical]  # Subsetting to columns without categorical indexes

    # Testing if there are an adequate number of features
    if X.shape[0] < X.shape[1] ** 2.:
        print('Will not perform well. Reduce the dimensionality and try again.')
        return
    
    # Creating and fitting the detector
    outlier_detector = OneClassSVM()
    outlier_detector.fit(X)
    
    # Predicting outliers and outputting an array with 1 if it is an outlier
    outliers = outlier_detector.predict(X)
    outlier_indices = np.where(outliers == -1)
    return outlier_indices

       
outlier_report(df)['feature']['Outlier type']  # Returns array of indices for outliers
# or
outlier_report(df)['Multiple feature outlier type']  # Returns array of indices for outliers
