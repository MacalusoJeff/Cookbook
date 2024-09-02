import sys
import os
import time
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

print(time.strftime('%Y/%m/%d %H:%M'))
print('OS:', sys.platform)
print('CPU Cores:', os.cpu_count())
print('Python:', sys.version)
print('NumPy:', np.__version__)
print('Pandas:', pd.__version__)
print('Scikit-Learn:', sklearn.__version__)

# Formatting for seaborn plots
sns.set_context('notebook', font_scale=1.1)
sns.set_style('ticks')

# Displays all dataframe columns
pd.set_option('display.max_columns', None)

# Force SettingWithCopyWarning to raise an exception instead of a warning
# This avoids cases where there is ambiguity around if a value was actually assigned
pd.set_option('mode.chained_assignment', 'raise')

#################################################################################################################
##### Cross Validation

# Holdout method
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=46)

# Train/validation/test split
test_size = 0.1  # Pct of overall data
val_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=46)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(val_size / (1 - test_size)), random_state=46)

# Train/val/test split by index, to be used when there is a datetime
def train_val_test_split_by_index(X: np.ndarray, y: np.ndarray, test_size: float=0.1, val_size: float=0.1):
    """
    Split the data into train, validation, and test sets by index. Can be used
    for time series data, assumes the data is already sorted.

    Parameters:
    - X (np.ndarray): Features array.
    - y (np.ndarray): Target array.
    - test_size (float): Proportion of the data to include in the test split.
    - val_size (float): Proportion of the data to include in the validation split.

    Returns:
    - X_train (np.ndarray): Features for the training set.
    - X_val (np.ndarray): Features for the validation set.
    - X_test (np.ndarray): Features for the test set.
    - y_train (np.ndarray): Target for the training set.
    - y_val (np.ndarray): Target for the validation set.
    - y_test (np.ndarray): Target for the test set.
    """
    # Calculate the number of samples for each set
    num_samples = X.shape[0]
    num_test = int(num_samples * test_size)
    num_val = int(num_samples * val_size)
    num_train = num_samples - num_test - num_val

    # Split the data
    X_train = X[:num_train]
    X_val = X[num_train:(num_train + num_val)]
    X_test = X[(num_train + num_val):]

    y_train = y[:num_train]
    y_val = y[num_train:(num_train + num_val)]
    y_test = y[(num_train + num_val):]

    return X_train, X_val, X_test, y_train, y_val, y_test


# K-fold cross validation
from sklearn.model_selection import KFold, cross_val_score
k_fold = KFold(n_splits=10, shuffle=True, random_state=46)
cross_val_score(model, X, y, cv=k_fold, n_jobs=-1)

#################################################################################################################
##### Hyperparameter tuning

# Random Search

def hyperparameter_random_search(X: np.ndarray, y: np.ndarray, model, parameters: dict, num_iterations: int = 50, num_folds: int = 5):
    """
    Perform random search for hyperparameter tuning using RandomizedSearchCV.

    Args:
        X (np.ndarray): The features used to train the model
        y (np.ndarray): Target values.
        model (Estimator object): The model to be tuned.
        param_distributions (dict):
            Dictionary with parameters names (string) as keys and distributions 
            or lists of parameters to try as values.
        num_iterations (int, optional): Number of parameter settings that are sampled.
        num_folds (int, optional): Number of cross-validation folds.

    Returns:
        BaseEstimator: Fitted estimator with best hyperparameters.
    """
    # Randomized Search
    randomized_search = RandomizedSearchCV(model, param_distributions=param_distributions,
                                           n_iter=num_iterations, cv=num_folds, n_jobs=-1, verbose=2)
    randomized_search.fit(X, y)

    # Reporting the results
    print('Best Estimator:', randomized_search.best_estimator_)
    print('Best Parameters:', randomized_search.best_params_)
    print('Best Score:', randomized_search.best_score_)

    return randomized_search.best_estimator_


# Grid search
from sklearn.model_selection import GridSearchCV

# Specifying the model and parameters to use
parameters = {'kernel': ('linear', 'rbf'), 'C': [1, 10]}
svc = svm.SVC()

# Performing grid search
model = GridSearchCV(svc, parameters)
model.fit(X, y)

print('Best Estimator:', model.best_estimator_, '\n', 
      'Best Parameters:', model.best_params_, '\n', 
      'Best Score:', model.best_score_)

# Iteratively training ensemble models
# Note: this function needs some more work :)
def iteratively_train_ensemble_model(model, num_trees_to_try: int, X_train, y_train, X_test, y_test) -> list:
    """
    Iteratively trains an ensemble model with different numbers of trees, collects the error for plotting, and returns the testing errors.

    Args:
        model (Ensemble model object): The ensemble model to be trained.
        num_trees_to_try (int): The different numbers of trees to try.
        X_train (np.ndarray): The training features.
        y_train (np.ndarray): The training labels.
        X_test (np.ndarray): The test data.
        y_test (np.ndarray): The test labels.

    Returns:
        list: A list of testing errors for each number of trees.

    # TODO: Allow different metrics, adjust for regression vs. classification, add total number of trees
    """
    # Enforcing the model has a warm start for iterative training
    if model.warm_start == False:
        model.set_params(warm_start=True)
    
    # Adding a seed if it does not exist
    if model.random_state == None:
        model.set_params(random_state=46)
        
    # Collecting the error for plotting
    testing_error = []
    
    # Iteratively training the model
    for num_trees in num_trees_to_try:
        model.set_params(n_estimators=num_trees)
        print('Fitting with {0} trees'.format(num_trees))
        model.fit(X_train, y_train)
        testing_error.append(metrics.log_loss(y_test, model.predict_proba(X_test)))
    
    return testing_errors


#################################################################################################################
##### Class Probability Cutoffs

# Probability Threshold Search - xgboost
cv = cross_validation.KFold(len(X), n_folds=5, shuffle=True, random_state=46)

# Making a dataframe to store results of various iterations
xgbResults = pd.DataFrame(columns=['probabilityThreshold', 'f1'])
accuracy, precision, recall, f1 = [], [], [], []

# Parameters for the model
num_rounds = 8000
params = {'booster': 'gbtree', 'max_depth': 4, 'eta': 0.001, 'objective': 'binary:logistic'}

for traincv, testcv in cv:
    
    # Converting the data frames/series to DMatrix objects for xgboost
    Dtrain = xgb.DMatrix(X.ix[traincv], label=y[traincv])
    Dtest = xgb.DMatrix(X.ix[testcv])
    
    # Building the model and outputting class probability estimations
    model = xgb.train(params, Dtrain, num_rounds)
    predictions = model.predict(Dtest)
    temporaryResults = pd.DataFrame(columns=['probabilityThreshold', 'f1'])
    
    # Looping through probability thresholds to gather the f1 score at each threshold
    for probabilityThreshold in np.linspace(0,0.1,100):
        predBin = pd.Series(predictions).apply(lambda x: 1 if x > probabilityThreshold else 0)
        threshF1 = {'probabilityThreshold': probabilityThreshold, 'f1': f1_score(y[testcv], predBin)}
        temporaryResults = temporaryResults.append(threshF1, ignore_index=True)
    
    # Retrieving the f1 score and probability thresholds at the highest f1 score
    bestIndex = list(temporaryResults['f1']).index(max(temporaryResults['f1']))
    bestTempResults = {'probabilityThreshold': temporaryResults.ix[bestIndex][0], 'f1': temporaryResults.ix[bestIndex][1]}
    xgbResults = xgbResults.append(bestTempResults, ignore_index=True)    

print('The Model performace is:')
print(xgbResults.mean())


# Probability Threshold Search - scikit-learn
def optimal_probability_cutoff(model, test_dataset: np.ndarray, test_labels: np.ndarray, max_thresh: float = 0.3, step_size: float = 0.01):
    '''
    Finds the optimal probability cutoff to maximize the F1 score.

    Args:
        model: The trained model used for prediction.
        test_dataset (np.ndarray): The test dataset used for prediction.
        test_labels (np.ndarray): The true labels of the test dataset.
        max_thresh (float, optional): The maximum probability threshold to consider. Defaults to 0.3.
        step_size (float, optional): The step size between probability thresholds. Defaults to 0.01.

    Returns:
        pd.Series: A pandas Series containing the optimal probability cutoff, F1 score.
                   The Series index represents the threshold and score.
    '''
    from sklearn import metrics

    # Prediction probabilities of the test dataset
    predicted = model.predict_proba(test_dataset)[:, 1]

    # Creating an empty dataframe to fill with probability cutoff thresholds and f1 scores
    results = pd.DataFrame(columns=['Threshold', 'F1 Score'])

    # Setting f1 score average metric based on binary or multi-class classification
    if len(np.unique(test_labels)) == 2:
        avg = 'binary'
    else:
        avg = 'micro'

    # Looping trhough different probability thresholds
    for thresh in np.arange(0, (max_thresh+step_size), step_size):
        pred_bin = pd.Series(predicted).apply(lambda x: 1 if x > thresh else 0)
        f1 = metrics.f1_score(test_labels, pred_bin, average=avg)
        tempResults = {'Threshold': thresh, 'F1 Score': f1}
        results = results.append(tempResults, ignore_index=True)
        
    # Plotting the F1 score throughout different probability thresholds
    results.plot(x='Threshold', y='F1 Score')
    plt.title('F1 Score by Probability Cutoff Threshold')
    
    best_index = list(results['F1 Score']).index(max(results['F1 Score']))
    print('Threshold for Optimal F1 Score:')
    return results.iloc[best_index]

#################################################################################################################
##### Prediction Intervals

# Prediction Intervals - Ensemble Scikit-Learn Models
# This is also a messy function that needs work
def ensemble_prediction_intervals(model, X: np.ndarray, X_train=None, y_train=None, percentile: float = 0.95) -> pd.DataFrame:
    """
    Calculates the specified prediction intervals for each prediction
    from an ensemble scikit-learn model.
    
    Args:
        model: 
            The scikit-learn model to create prediction intervals for. This must be
            either a RandomForestRegressor or GradientBoostingRegressor
        X (np.ndarray): The input array to create predictions & prediction intervals for
        X_train (np.ndarray, optional): The training features for the gradient boosted trees
        y_train (np.ndarray, optional): The training label for the gradient boosted trees
        percentile (float): The prediction interval percentile. Default of 0.95 is 0.025 - 0.975
    
    Note: Use X_train and y_train when using a gradient boosted regressor because a copy of
          the model will be re-trained with quantile loss.
          These are not needed for a random forest regressor
    
    Returns: 
        pd.DataFrame: The predictions and prediction intervals for X
    
    TODO: 
      - Try to optimize by removing loops where possible
      - Fix upper prediction intervals for gradient boosted regressors
      - Make work with xgboost and lightgbm
    """
    # Checking if the model has the estimators_ attribute
    if 'estimators_' not in dir(model):
        print('Not an ensemble model - exiting function')
        return

    # Accumulating lower and upper prediction intervals
    lower_PI = []
    upper_PI = []
    
    # Generating predictions to be returned with prediction intervals
    print('Generating predictions with the model')
    predictions = model.predict(X)
    
    # Prediction intervals for a random forest regressor
    # Taken from https://blog.datadive.net/prediction-intervals-for-random-forests/
    if str(type(model)) == "<class 'sklearn.ensemble.forest.RandomForestRegressor'>":
        print('Generating upper and lower prediction intervals')
        
        # Looping through individual records for predictions
        for record in range(len(X)):
            estimator_predictions = []
        
            # Looping through estimators and gathering predictions
            for estimator in model.estimators_:
                estimator_predictions.append(estimator.predict(X[record].reshape(1, -1))[0])
            
            # Adding prediction intervals
            lower_PI.append(np.percentile(estimator_predictions, (1 - percentile) / 2.))
            upper_PI.append(np.percentile(estimator_predictions, 100 - (1 - percentile) / 2.))
    
    # Prediction intervals for gradient boosted trees
    # Taken from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html
    if str(type(model)) == "<class 'sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'>":
        # Cloning the model so the original version isn't overwritten
        from sklearn.base import clone
        quantile_model = clone(model)
        
        # Calculating buffer for upper/lower alpha to get the Xth percentile
        alpha_buffer = ((1 - x) / 2)
        alpha = percentile + alpha_buffer
        
        # Setting the loss function to quantile before re-fitting
        quantile_model.set_params(loss='quantile')
        
        # Upper prediction interval
        print('Generating upper prediction intervals')
        quantile_model.set_params(alpha=alpha)
        quantile_model.fit(X_train, y_train)
        upper_PI = quantile_model.predict(X)
        
        # Lower prediction interval
        print('Generating lower prediction intervals')
        quantile_model.set_params(alpha=(1 - alpha))
        quantile_model.fit(X_train, y_train)
        lower_PI = quantile_model.predict(X)
    
    # Compiling results of prediction intervals and the actual predictions
    results = pd.DataFrame({'lower_PI': lower_PI,
                            'prediction': predictions,
                            'upper_PI': upper_PI})
    
    return results


#################################################################################################################
##### Ensemble Predictions
  
# Blending predictions - xgboost
def blend_xgboost_predictions(train_features: np.ndarray, train_labels: np.ndarray, prediction_features: np.ndarray, num_models: int = 3) -> np.ndarray:
    """
    Trains the number of specified xgboost models and averages the predictions.
    
    Args: 
        train_features (np.ndarray): The features for the training dataset
        train_labels (np.ndarray): The labels for the training dataset
        prediction_features (np.ndarray): The features to create predictions for
        num_models (int): The number of models to train
        
    Returns:
        np.ndarray: Point or class probability predictions
    """
    
    # Auto-detecting if it's a classification problem and setting the objective for the model
    # Adjust the num_classes cutoff if dealing with a high number of classes
    num_classes = len(np.unique(train_labels))
    if num_classes < 50:
        is_classification = 1
        if num_classes == 2:
            objective = 'binary:logistic'
        else:
            objective = 'multi:softprob'
    else:
        is_classification = 0
        objective = 'reg:linear'
        
    # Creating the prediction object to append results to
    predictions = []
    
    # Parameters for the model - http://xgboost.readthedocs.io/en/latest/parameter.html
    num_rounds = 100
    params = {'booster': 'gbtree',
              'max_depth': 6,  # Default is 6
              'eta': 0.3,  # Step size shrinkage. Default is 0.3
              'alpha': 0,  # L1 regularization. Default is 0.
              'lambda': 1,  # L2 regularization. Default is 1.
              
              # Use reg:linear for regression
              # Use binary:logistic, or multi:softprob for classification
              # Add gpu: to the beginning if training with a GPU. Ex. 'gpu:'+objective
              'objective': objective
             }
    
    # Adding the required parameter for num_classes if performing multiclass classificaiton
    if is_classification == 1 and num_classes != 2:
        params['num_class'] = num_classes
    
    # Creating DMatrix objects from X/y
    D_train = xgb.DMatrix(train_features, label=train_labels)
    D_test = xgb.DMatrix(prediction_features)
    
    # Training each model and gathering the predictions
    for num_model in range(num_models):
        
        # Progress printing for every 10% of completion
        if (num_model+1) % (round(num_models) / 10) == 0:
            print('Training model number', num_model+1)
        
        # Training the model and gathering predictions
        model = xgb.train(params, D_train, num_rounds)
        model_prediction = model.predict(D_test)
        predictions.append(model_prediction)
    
    # Averaging the predictions for output
    predictions = np.asarray(predictions).mean(axis=0)
    
    return predictions


# Blending predictions - Scikit-Learn & LightGBM
def blend_predictions(model, train_features: np.ndarray, train_labels: np.ndarray, prediction_features: np.ndarray,
                      num_models: int = 3, average_results: bool = False) -> np.ndarray:
    """
    Trains the number of specified scikit-learn or LightGBM models and averages the predictions.
    
    Args: 
        train_features (np.ndarray): The features for the training dataset
        train_labels (np.ndarray): The labels for the training dataset
        prediction_features (np.ndarray): The features to create predictions for
        num_models (int): The number of models to train
        average_results (bool): Whether or not to return the raw results or the averaged results
        
    Returns:
        np.ndarray: A numpy array of point or class probability predictions
    """
    from sklearn.base import clone
    
    # Auto-detecting if it's a classification problem
    # Adjust the num_classes cutoff if dealing with a high number of classes
    num_classes = len(np.unique(train_labels))
    if num_classes < 50:
        is_classification = 1
    else:
        is_classification = 0
        
    # Creating the prediction object to append results to
    predictions = []
        
    # Training each model and gathering the predictions
    for num_model in range(num_models):
        
        # Progress printing for every 10% of completion
        if (num_model+1) % (round(num_models) / 10) == 0:
            print('Training model number', num_model+1)
        
        # Cloning the original model
        model_iteration = clone(model)
        
        # Training the model
        model_iteration.fit(train_features, train_labels)
        
        # Gathering predictions
        if is_classification == 1:
            model_prediction = model_iteration.predict_proba(prediction_features)
        else:
            model_prediction = model_iteration.predict(prediction_features)
        predictions.append(model_prediction)
    
    # Averaging the predictions for output
    if average_results == True:
        predictions = np.asarray(predictions).mean(axis=0)
    
    return predictions

#################################################################################################################
##### Evaluating Clusters

def evaluate_k_means(data: np.ndarray, max_num_clusters: int = 10, is_data_scaled: bool = True) -> list:
    """
    Evaluates the K-means clustering algorithm by computing the inertia for different numbers of clusters.

    Args:
        data (np.ndarray): The input data to be clustered.
        max_num_clusters (int, optional): The maximum number of clusters to consider. Defaults to 10.
        is_data_scaled: (bool, optional): Specifies whether the input data is already scaled. Defaults to True.

    Returns:
        List: A list containing the inertia values for each number of clusters.
    """
    from sklearn.cluster import KMeans
    
    # Min max scaling the data if it isn't already scaled
    if is_data_scaled == False:
        from sklearn.preprocessing import MinMaxScaler
        data = MinMaxScaler().fit_transform(data)
    
    # For gathering the results and plotting
    inertia = []
    clusters_to_try = np.arange(2, max_num_clusters+1)
    
    # Iterating through the clusters and gathering the inertia
    for num_clusters in np.arange(2, max_num_clusters+1):
        print('Fitting with {0} clusters'.format(num_clusters))
        model = KMeans(n_clusters=num_clusters, n_jobs=-1)
        model.fit(data)
        inertia.append(model.inertia_)
    
    # Plotting the results
    plt.figure(figsize=(10, 7))
    plt.plot(clusters_to_try, inertia, marker='o')
    plt.xticks(clusters_to_try)
    plt.xlabel('# Clusters')
    plt.ylabel('Inertia')
    plt.title('Inertia by Number of Clusters')
    plt.show()
    
    return inertia


#################################################################################################################
##### Saving & Loading Models

def save_model(model, filepath: str, add_timestamp: bool = True) -> None:
    """
    Saves a machine learning model to a file.

    Args:
        model: The trained model object to be saved.
        filepath (str): The file path (including the file name and extension) where the model will be saved.
        add_timestamp (bool, optional): Specifies whether to add a timestamp to the file name. Defaults to True.

    Returns:
        None
    """
    import os
    
    # Creating the sub directory if it does not exist
    directory = filepath.split('/')[:-1]  # Gathering the components of the file path
    directory = '/'.join(directory) + '/'  # Formatting into the directory/subdirectory/ format
    if not os.path.exists(directory):
        print('Creating the directory')
        os.makedirs(directory)
        
    # Adding the date to the end of the file name if it doesn't exist
    # E.g. instead of model.pkl, model_yyyymmdd.pkl
    if add_timestamp == True:
        import datetime
        today = datetime.datetime.today().strftime('%Y%m%d')
        today = '_' + today + '.'
        filepath = filepath.split('.')
        filepath.insert(1, today)
        filepath = ''.join(filepath)
    
    print('Saving model')
    pickle.dump(model, open(filepath, 'wb'))
    print('Model saved')
