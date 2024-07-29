import xgboost as xgb

# Train a model with early stopping
model = xgb.XGBRegressor(early_stopping_rounds=5, n_jobs=-1, random_state=46)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("Stopped on iteration: ", early_stopping_model.best_iteration)


### Hyperparameter tune w/ early stopping - using sklearn RandomSearchCV()
# Note that the validation set is only used for early stopping and the cv param will create a separate validation set
import scipy
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error

# Large number of estimators since we will be using early stopping
# The early stopping rounds must be called in the model initialization to work with RandomizedSearchCV()
model = xgb.XGBRegressor(n_estimators=10000, early_stopping_rounds=20, n_jobs=-1, random_state=46)

# Define the parameter distributions for hyperparameter tuning
# Using this guide: https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
# Parameter documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
param_distributions = {
    "learning_rate": scipy.stats.uniform(loc=0.003, scale=0.19),  # Default is 0.3. Ranges from loc to loc+scale.
    "subsample": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    "colsample_bytree": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    "min_child_weight": [1, 3, 5, 7],  # Default is 1
    "max_depth": np.append(0, np.arange(3, 16)),  # Default is 6
    "alpha": [0, 0.01, 1, 2, 5, 7, 10, 50, 100],  # Default is 0. AKA reg_alpha.
    "lambda": [0, 0.01, 1, 5, 10, 20, 50, 100]  # Default is 0. AKA reg_lambda.
}

# Configure the randomized search
random_search = RandomizedSearchCV(model,
                                   param_distributions=param_distributions,
                                   n_iter=40,
                                   cv=3,  # k-folds
                                   #  cv=ShuffleSplit(n_splits=1, test_size=.2, random_state=46),  # Train/test split
                                   scoring="neg_mean_squared_error",
                                   n_jobs=-1)

# Perform the randomized search with early stopping
random_search.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

# Extract the tuned model from the random search
tuned_model = random_search.best_estimator_
print("Tuned model -")
print("R^2: ", tuned_model.score(X_test, y_test))
print("MSE: ", mean_squared_error(y_test, tuned_model.predict(X_test)), "\n")

# The XGBoost documentation recommends re-training the model with early stopping after the optimal hyperparameters are found with cross validation
# This is because the number of trees will likely change with each fold
# https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html#early-stopping
optimal_params = tuned_model.get_params()
optimal_params["n_estimators"] = tuned_model.best_iteration
tuned_retrained_model = xgb.XGBRegressor(**optimal_params)  # Inherits n_jobs and random_state from above
tuned_retrained_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print("Tuned re-trained model -")
print("R^2: ", tuned_retrained_model.score(X_test, y_test))
print("MSE: ", mean_squared_error(y_test, tuned_retrained_model.predict(X_test)))


### Hyperparameter tune w/ early stopping - using custom built random search
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from tqdm import tqdm

# Use train/val/test split or k-folds cross validation
tune_with_kfolds = True

# Define the parameter distributions for hyperparameter tuning
# Using this guide: https://machinelearningmastery.com/configure-gradient-boosting-algorithm/
# Parameter documentation: https://xgboost.readthedocs.io/en/stable/parameter.html
param_distributions = {
    "learning_rate": scipy.stats.uniform(loc=0.003, scale=0.19),  # Default is 0.3. Ranges from loc to loc+scale.
    "subsample": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    "colsample_bytree": scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1
    "min_child_weight": [1, 3, 5, 7],  # Default is 1
    "max_depth": np.append(0, np.arange(3, 16)),  # Default is 6
    "alpha": [0, 0.01, 1, 2, 5, 7, 10, 50, 100],  # Default is 0. AKA reg_alpha.
    "lambda": [0, 0.01, 1, 5, 10, 20, 50, 100]  # Default is 0. AKA reg_lambda.
}

def sample_from_param_distributions(param_distributions: dict) -> dict:
    """
    Sample a value from each parameter distribution defined in param_distributions.

    Parameters:
    - param_distributions (dict): Dictionary where keys are parameter names and values are either:
        - scipy.stats distribution objects for continuous distributions.
        - Lists or numpy arrays for discrete choices.

    Returns:
    - sampled_values (dict): Dictionary containing sampled values corresponding to each parameter.
    """
    sampled_values = {}
    for param, distribution in param_distributions.items():
        if isinstance(distribution, scipy.stats._distn_infrastructure.rv_frozen):
            sampled_values[param] = distribution.rvs()
        elif isinstance(distribution, list) or isinstance(distribution, np.ndarray):
            sampled_values[param] = np.random.choice(distribution)
        else:
            raise ValueError(f"Unsupported distribution type for parameter '{param}'")

    return sampled_values


num_iterations = 40
optimal_params = {}
best_score = -np.inf
for iteration in tqdm(range(num_iterations)):
    # Sample values from the distributions
    sampled_params = sample_from_param_distributions(param_distributions)

    # Train the model, get the performance on the validation set
    model = xgb.XGBRegressor(n_estimators=10000, early_stopping_rounds=20,
                             n_jobs=-1, random_state=46, **sampled_params)

    # Perform the tuning with either k-folds or train/test split
    if tune_with_kfolds == True:
        cv = KFold(n_splits=3, shuffle=True, random_state=46)  # Use StratifiedKFold for classification
        cv_results = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
            y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

            model_clone = clone(model)
            model_clone.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
            predictions = model_clone.predict(X_val_fold)
            fold_neg_mse = -mean_squared_error(y_true=y_val_fold, y_pred=predictions)
            cv_results.append(fold_neg_mse)

        neg_mean_squared_error = np.mean(cv_results)
    else:
        # Train/test split with the validation data set for early stopping
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        predictions = model.predict(X_val)
        neg_mean_squared_error = -mean_squared_error(y_true=y_val, y_pred=predictions)

    # Set the optimal parameters if the performance is better
    if neg_mean_squared_error > best_score:
        best_score = neg_mean_squared_error
        optimal_params = sampled_params
        # Need to re-train w/ early stopping to get optimal number of estimators if tuned with k-folds
        if tune_with_kfolds == False:
            optimal_params["n_estimators"] = model.best_iteration

# Re-train with the optimal hyperparams
# Re-perform early stopping if k-folds was used for tuning
if tune_with_kfolds == True:
    tuned_model = xgb.XGBRegressor(**optimal_params, n_jobs=-1, random_state=46,
                                   n_estimators=10000, early_stopping_rounds=20)
    tuned_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    optimal_params["n_estimators"] = tuned_model.best_iteration
else:
    tuned_model = xgb.XGBRegressor(**optimal_params, n_jobs=-1, random_state=46)
    tuned_model.fit(X_train, y_train)

# Report the results
print("Tuned model -")
print("R^2: ", tuned_model.score(X_test, y_test))
print("MSE: ", mean_squared_error(y_test, tuned_model.predict(X_test)))
