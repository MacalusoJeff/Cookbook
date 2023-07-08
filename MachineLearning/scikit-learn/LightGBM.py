import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# 60/20/20 train/val/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=46)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=46)

### Early stopping w/ sklearn API
# Set the hyperparameters for LightGBM
# params documentation: https://lightgbm.readthedocs.io/en/latest/Parameters.html
params = {
    'metric': 'mse',
    'learning_rate': 0.01,
    'n_estimators': 30000,  # Early stopping should catch this earlier and keep it from overfitting
    'num_leaves': 31,
    'feature_fraction': 0.9
}

# Train the model with early stopping
model = lgb.LGBMRegressor(n_jobs=-1, random_state=46, **params)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100)])
print(f'Done at {model.best_iteration_} iterations')

# Predict on the test set
y_pred = model.predict(X_test, num_iteration=model.best_iteration_)


### Early stopping with the LGBM API
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)

# Specifying the parameters for the model
# params documentation: https://lightgbm.readthedocs.io/en/latest/Parameters.html
params = {'objective': 'multiclass',  # regression, binary, or multiclass
          'num_class': 3,  # Number of classes for a multiclass problem
          'metric': ['multi_logloss'],  # rmse, mape, auc, multi_logloss, etc.
          'learning_rate': 0.001}
# Training the model
print('Starting training')
gbm = lgb.train(params,
                lgb_training_set,
                num_boost_round=10000,  # Early stopping should catch this earlier and keep it from overfitting
                valid_sets=lgb_evaluation_set,
                early_stopping_rounds=10,
                verbose_eval=10)
print('Done at {0} iterations'.format(gbm.best_iteration))


### Hyperparameter tuning with early stopping
# Defining a new model object with a large number of estimators since we will be using early stopping
model = lgb.LGBMRegressor(n_estimators=10000, n_jobs=-1, random_state=46)

# Define the parameter distributions for hyperparameter tuning
# Using this guide: https://towardsdatascience.com/beginners-guide-to-the-must-know-lightgbm-hyperparameters-a0005a812702
param_distributions = {
    'learning_rate': scipy.stats.uniform(loc=0.003, scale=0.19),  # Default is 0.1. Ranges from loc to loc+scale.
    'num_leaves': scipy.stats.randint(8, 256),  # Default is 31
    'max_depth': np.append(-1, np.arange(3, 16)),  # Default is -1
    'min_child_samples': scipy.stats.randint(5, 300),  # Default is 20. AKA min_data_in_leaf.
    'subsample': scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1. AKA bagging_fraction.
    'colsample_bytree': scipy.stats.uniform(loc=0.5, scale=0.5),  # Default is 1.0. AKA feature_fraction.
    'reg_alpha': [0, 0.01, 1, 2, 5, 7, 10, 50, 100],  # Default is 0. AKA lambda_l1.
    'reg_lambda': [0, 0.01, 1, 5, 10, 20, 50, 100]  # Default is 0. AKA lambda_l2.
}

# Configuring the randomized search
random_search = RandomizedSearchCV(model,
                                   param_distributions=param_distributions,
                                   n_iter=20, cv=3,
                                   scoring='neg_mean_squared_error',
                                   n_jobs=-1)

# Performing the randomized search with early stopping
random_search.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  callbacks=[lgb.early_stopping(20)])

# Extracting the parameters from the best model to re-train the model
# Updating the number of estimators to the best iteration from early stopping
best_model = random_search.best_estimator_
optimal_params = best_model.get_params()
optimal_params['n_estimators'] = best_model.best_iteration_

# Re-training the tuned model
model = lgb.LGBMRegressor(**optimal_params)  # Inherits n_jobs and random_state from above
model.fit(X_train, y_train)
print('Tuned model -')
print("R^2: ", model.score(X_test, y_test))
print('MSE: ', mean_squared_error(y_test, model.predict(X_test)))
