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
    """
    Calculates the percentage of missing values for each column in a pandas DataFrame.

    Args:
        dataframe: pandas DataFrame
            The input DataFrame containing the data.

    Returns:
        dict:
            A dictionary where keys represent column names and values represent the percentage
            of missing values in each respective column.

    Example:
        >>> data = pd.DataFrame({'A': [1, 2, np.nan], 'B': [np.nan, 4, 5], 'C': [6, 7, 8]})
        >>> percent_missing(data)
        {'A': 0.3333, 'B': 0.6666}

    Notes:
        - The function requires the pandas library to be installed. You can install it via pip: `pip install pandas`.
        - The percentage of missing values is calculated by summing the number of missing values in each column
          and dividing it by the total number of rows in the DataFrame.
        - The resulting dictionary provides the column names as keys and the corresponding percentage of missing
          values as values.
        - If a column has no missing values, it will not be included in the dictionary.
    """
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
# TODO: Make this shorter and less messy
def predict_missing_values(data, column, correlationThresh: float = 0.5, cross_validations: int = 3):
    """
    Fills missing values using a random forest model on highly correlated columns
    Returns a series of the column with missing values filled
    
    Args:
        data: pandas DataFrame
            The input DataFrame containing the data.
        column: str
            The name of the column with missing values to be predicted.
        correlationThresh: float, optional (default=0.5)
            The threshold for selecting highly correlated columns. Columns with an absolute correlation
            coefficient greater than or equal to this threshold will be used in the prediction model.
        cross_validations: int, optional (default=3)
            The number of cross-validations to perform during model evaluation.

    Returns:
        pandas Series:
            A series containing the predicted missing values for the specified column.
            
    Notes:
        - The function requires the pandas library and scikit-learn library to be installed.
          You can install them via pip: `pip install pandas scikit-learn`.
        - The function fills missing values in the specified column using a random forest model.
        - The highly correlated columns are selected based on their absolute correlation coefficient
          with the target column.
        - The model used for prediction depends on the number of unique values in the column.
          For continuous variables or categorical variables with more than 25 unique classes,
          a Random Forest Regressor is used. For categorical variables with 25 or fewer unique classes,
          a Random Forest Classifier is used.
        - The cross_validations parameter determines the number of cross-validations performed
          during model evaluation.
        - The function prints information about the percentage of missing values, the model's
          cross-validation score, and, if applicable, the R^2 or accuracy score of the fitted model.
        - The original DataFrame is not modified; a copy is made to prevent overwriting.
        - The function returns a pandas Series containing the predicted missing values for the specified column.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn import ensemble
    
    # Printing number of percentage values missing
    pctMissing = data[column].isnull().values.sum() / data.shape[0]
    print('Predicting missing values for {0}\n'.format(column))
    print('Percentage missing: {0:.2f}%'.format(pctMissing*100))
    
    # Instantiating the model
    # Picking a classification model if the number of unique labels are 25 or under
    num_unique_values = len(np.unique(data[column]))
    if num_unique_values > 25 or data[column].dtype != 'category':
        print('Variable is continuous')
        rfImputer = ensemble.RandomForestRegressor(n_estimators=100, n_jobs=-1)
    else:
        print('Variable is categorical with {0} classes').format(num_unique_values)
        rfImputer = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=-1)
    
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
    cvScore = np.mean(cross_val_score(rfImputer, X, y, cv=cross_validations, n_jobs=-1))
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
    

#################################################################################################################
##### Outliers

# Detecting outliers with Interquartile Range (IQR)
# Note: The function in its current form is taken from Chris Albon's Machine Learning with Python Cookbook
def iqr_indices_of_outliers(X: np.ndarray) -> np.ndarray:
    """
    Detects outliers in an array using the interquartile range (IQR) method.
    
    Args:
        X: numpy ndarray
            An array of a variable to detect outliers for.
    
    Returns:
        numpy ndarray:
            An array with indices of the detected outliers.
    
    Example:
        >>> data = np.array([1, 2, 3, 10, 15, 20])
        >>> iqr_indices_of_outliers(data)
        array([3, 4, 5])

    Notes:
        - The function requires the numpy library to be installed.
          You can install it via pip: `pip install numpy`.
        - Outliers are values that fall below the lower bound or above the upper bound, where the
          bounds are defined as Q1 - 1.5 * IQR and Q3 + 1.5 * IQR, respectively.
        - Q1 is the first quartile (25th percentile), Q3 is the third quartile (75th percentile),
          and IQR is the interquartile range (Q3 - Q1).
        - The function returns an array containing the indices of the detected outliers.
        - If no outliers are detected, an empty array is returned.
    """
    q1, q3 = np.percentile(X, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    outlier_indices = np.where((X > upper_bound) | (X < lower_bound))
    return outlier_indices


# Detecting outliers with Z scores
def z_score_indices_of_outliers(X: np.ndarray, threshold: float = 3) -> np.ndarray:
    """
    Detects outliers in an array using the Z-score method.
    
    Args:
        X: numpy ndarray
            An array of a variable to detect outliers for.
        threshold: float, optional (default=3)
            The number of standard deviations from the mean to be considered an outlier.
    
    Returns:
        numpy ndarray:
            An array with indices of the detected outliers.
    
    Example:
        >>> data = np.array([1, 2, 3, 10, 15, 20])
        >>> z_score_indices_of_outliers(data)
        array([3, 4, 5])

    Notes:
        - The function requires the numpy library to be installed.
          You can install it via pip: `pip install numpy`.
        - Outliers are values that fall outside a certain number of standard deviations from the mean.
          The number of standard deviations is determined by the `threshold` argument.
        - The function calculates the mean and standard deviation of the input array using the numpy
          `np.mean` and `np.std` functions, respectively.
        - The Z-score for each element in the array is computed by subtracting the mean and dividing
          by the standard deviation.
        - The function identifies the outlier indices by comparing the absolute values of the Z-scores
          with the specified threshold using the `np.where` function.
        - The function returns an array containing the indices of the detected outliers.
        - If no outliers are detected, an empty array is returned.
    """
    X_mean = np.mean(X)
    X_stdev = np.std(X)
    z_scores = [(y - X_mean) / X_stdev for y in X]
    outlier_indices = np.where(np.abs(z_scores) > threshold)
    return outlier_indices


# Detecting outliers with the Elliptical Envelope method
def ellipses_indices_of_outliers(X: np.ndarray, contamination: float = 0.1) -> np.ndarray:
    """
    Detects outliers in an array using the elliptical envelope method.
    
    Args:
        X: numpy ndarray
            An array of all variables to detect outliers for.
        contamination: float, optional (default=0.1)
            The expected proportion of outliers in the data.
    
    Returns:
        numpy ndarray:
            An array with indices of the detected outliers.
    
    Example:
        >>> data = np.array([[1, 2], [2, 4], [10, 12], [15, 18], [20, 24]])
        >>> ellipses_indices_of_outliers(data)
        array([2, 3, 4])

    Notes:
        - The function requires the numpy library and the EllipticEnvelope class from the
          sklearn.covariance module to be installed. You can install numpy via pip: `pip install numpy`.
          For scikit-learn, you can use: `pip install scikit-learn`.
        - The function detects outliers in the given array using the elliptical envelope method.
          This method fits a multivariate Gaussian distribution to the data and identifies points
          that are unlikely to belong to the estimated distribution.
        - The function drops categorical columns from the input array to ensure compatibility
          with the elliptical envelope method. Categorical columns are identified based on the
          number of unique values in each feature. Features with more than 30 unique values are
          considered non-categorical and included in the outlier detection process.
        - The function checks if there are enough features (columns) compared to the number of
          observations (rows) to perform the outlier detection adequately. If the number of rows
          is less than the square of the number of columns, the function warns that the results
          may not be reliable and suggests reducing the dimensionality of the data.
        - The function creates an instance of the EllipticEnvelope class with the specified
          contamination parameter, which determines the expected proportion of outliers in the data.
        - The outlier detector is fitted to the preprocessed array.
        - The function predicts outliers based on the fitted detector and returns an array
          containing the indices of the detected outliers.
    """
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
    """
    Detects outliers in an array using the isolation forest method.
    
    Args:
        X: array or dataframe
            Non-categorical variables to detect outliers for.
        contamination: float or 'auto', optional (default='auto')
            The expected proportion of outliers in the data. If set to 'auto',
            the contamination parameter is automatically determined based on
            the size of the input data.
        n_estimators: int, optional (default=100)
            The number of trees to use in the isolation forest.
    
    Returns:
        numpy ndarray:
            An array with indices of the detected outliers.
    
    Example:
        >>> data = np.array([[1, 2], [2, 4], [10, 12], [15, 18], [20, 24]])
        >>> isolation_forest_indices_of_outliers(data)
        array([2, 3, 4])

    Notes:
        - The function requires the numpy library and the IsolationForest class from the
          sklearn.ensemble module to be installed. You can install numpy via pip: `pip install numpy`.
          For scikit-learn, you can use: `pip install scikit-learn`.
        - The function detects outliers in the given array using the isolation forest method.
          The isolation forest algorithm isolates observations by randomly selecting a feature
          and then randomly selecting a split value between the maximum and minimum values of
          the selected feature. The number of splittings required to isolate an observation
          serves as a measure of the abnormality of that observation.
        - The function creates an instance of the IsolationForest class with the specified
          contamination parameter and number of estimators.
        - The outlier detector is fitted to the input array.
        - The function predicts outliers based on the fitted detector and returns an array
          containing the indices of the detected outliers.
    """
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

outlier_indexes_forest = isolation_forest_indices_of_outliers(X.select_dtypes(exclude='category'),
                                                              contamination='auto')
print('Outliers detected: {0}'.format(len(outlier_indexes_forest[0])))


# Detecting outliers with the One Class SVM method
def one_class_svm_indices_of_outliers(X: np.ndarray) -> np.ndarray:
    """
    Detects outliers in an array using the one-class SVM (Support Vector Machine) method.
    
    Args:
        X: array or dataframe
            An array of all variables to detect outliers for.
    
    Returns:
        numpy ndarray:
            An array with indices of the detected outliers.
    
    Example:
        >>> data = np.array([[1, 2], [2, 4], [10, 12], [15, 18], [20, 24]])
        >>> one_class_svm_indices_of_outliers(data)
        array([2, 3, 4])

    Notes:
        - The function requires the numpy library and the OneClassSVM class from the
          sklearn.svm module to be installed. You can install numpy via pip: `pip install numpy`.
          For scikit-learn, you can use: `pip install scikit-learn`.
        - The function detects outliers in the given array using the one-class SVM method,
          which is an unsupervised algorithm that learns a decision function representing
          the support of a high-dimensional distribution. It is commonly used for novelty
          detection or outlier detection.
        - The function creates an instance of the OneClassSVM class.
        - The input array is preprocessed by dropping categorical columns. Non-categorical
          columns are selected by checking the number of unique values in each column.
        - The function checks if there are an adequate number of features based on the
          dimensions of the preprocessed array. If not, it prints a warning message and returns.
        - The outlier detector is fitted to the preprocessed array.
        - The function predicts outliers based on the fitted detector and returns an array
          containing the indices of the detected outliers.
    """
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
