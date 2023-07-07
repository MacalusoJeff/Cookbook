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





