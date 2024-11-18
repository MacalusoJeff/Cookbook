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

def binary_eval_metrics(predicted_probabilities: np.ndarray, labels: np.ndarray, class_probability_cutoff: float=0.5) -> dict:
    """
    Computes various evaluation metrics for binary predictions and returns them as a dictionary.

    Args:
        predictions (numpy.ndarray): The predicted labels (binary) from a binary classification model.
        labels (numpy.ndarray): The true binary labels corresponding to the predictions.

    Returns:
        dict: A dictionary containing various evaluation metrics.

    """
    from sklearn import metrics

    # Extract the probability of the positive class if both classes are present
    if predicted_probabilities.ndim > 1:
        predicted_probabilities = predicted_probabilities[:, 1]

    # Generate the binary predictions based on the class_probability
    predictions = (predicted_probabilities > class_probability_cutoff).astype(int)

    # Calculate baseline accuracy with various methods
    zero_rule_accuracy = max(labels.mean(), 1 - labels.mean())  # Always predict the most common class
    uniform_random_predictions = np.random.binomial(n=1, p=0.5, size=len(labels))  # Randomly predict 0 or 1
    uniform_random_accuracy = metrics.accuracy_score(labels, uniform_random_predictions)
    label_distribution_predictions = np.random.binomial(n=1, p=labels.mean(), size=len(labels))  # Randomly predict based on the label distribution
    label_distribution_accuracy = metrics.accuracy_score(labels, label_distribution_predictions)

    # Calculate model metrics and fill the dict to return
    results = {}
    results['Zero Rule Baseline Accuracy'] = zero_rule_accuracy
    results['Uniform Random Baseline Accuracy'] = uniform_random_accuracy
    results['Label Distribution Baseline Accuracy'] = label_distribution_accuracy
    results['Accuracy'] = metrics.accuracy_score(labels, predictions)
    results['Precision'] = metrics.precision_score(labels, predictions)
    results['Recall'] = metrics.recall_score(labels, predictions)
    results['F1'] = metrics.f1_score(labels, predictions)
    results['AUC'] = metrics.roc_auc_score(labels, predicted_probabilities)
    results['Average Precision (PR AUC)'] = metrics.average_precision_score(labels, predicted_probabilities)

    return results


#################################################################################################################
##### Evaluation Plots

# Residuals
def plot_residuals(model, values: np.ndarray, labels: np.ndarray) -> None:
    """
    Creates two plots: Actual vs. Predicted and Residuals.

    Args:
        model: The trained regression model.
        values (np.ndarray): The feature values used for prediction.
        labels (np.ndarray): The true labels corresponding to the feature values.

    Returns:
        None.
    """
    # Calculating the predictions and residuals
    predictions = model.predict(values)
    df_results = pd.DataFrame({'Actual': labels, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])
    
    # Plotting the actual vs predicted
    sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, size=6)

    # Plotting the diagonal line
    line_coords = np.arange(df_results.min().min(), df_results.max().max())
    plt.plot(line_coords, line_coords,  # X and y points
             color='darkorange', linestyle='--')
    plt.title('Actual vs. Predicted')
    plt.show()
    
    # Plotting the residuals
    ax = plt.subplot(111)
    plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
    plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine
    plt.title('Residuals')
    plt.show()


# Learning Curve
def plot_learning_curve(model, data, labels):
    '''
    Plots the learning curve of a model using 3-fold Cross Validation
    '''
    from sklearn.model_selection import learning_curve
    learningCurve = learning_curve(model, X, y, cv=3, n_jobs=-1)
    trainScores = learningCurve[1].mean(axis=1)
    testScores = learningCurve[2].mean(axis=1)

    # Putting the results into a dataframe before plotting
    results = pd.DataFrame({'Training Set': trainScores, 'Testing Set': testScores},
                           index=learningCurve[0])  # Training size
    
    # Plotting the curve
    ax = results.plot(figsize=(10, 6), linestyle='-', marker='o')
    plt.title('Learning Curve')
    plt.xlabel('Training Size')
    plt.ylabel('Cross Validation Score')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine    
    plt.legend(loc=(1.04, 0.55))  # Moves the legend outside of the plot
    plt.show()
    

plot_learning_curve(model, X, y)


# Validation Curve
def plot_validation_curve(model, data: np.ndarray, labels: np.ndarray, param_name: str, param_values: list):
    """
    Plots the validation curve of a model using 3-fold Cross Validation.

    Args:
        model: The machine learning model to plot the validation curve for.
        data (np.ndarray): The input data used for training the model.
        labels (np.ndarray): The target labels corresponding to the input data.
        param_name (str): The name of the parameter that will be varied
        param_values (list): The values to use for the validation plot

    Returns:
        None.
    """
    from sklearn.model_selection import validation_curve
    validationCurve = validation_curve(model, X, y, cv=3, param_name=param_name,
                                       param_range=param_values, n_jobs=-1)

    trainScores = validationCurve[0].mean(axis=1)
    testScores = validationCurve[1].mean(axis=1)

    # Putting the results into a dataframe before plotting
    results = pd.DataFrame({'Training Set': trainScores, 'Testing Set': testScores}, 
                           index=param_values)
    
    # Plotting the curve
    ax = results.plot(figsize=(10, 6), linestyle='-', marker='o')
    plt.title('Validation Curve')
    plt.xlabel(param_name)
    plt.ylabel('Cross Validation Score')
    ax.spines['right'].set_visible(False)  # Removing the right spine
    ax.spines['top'].set_visible(False)  # Removing the top spine    
    plt.legend(loc=(1.04, 0.55))  # Moves the legend outside of the plot
    plt.show()
    

param_name = 'n_estimators'
param_range = [10, 30, 100, 300]
plot_validation_curve(model, X, y, param_name, param_range)


# Ensemble Model's Feature Importance
def plot_ensemble_feature_importance(model, features, top_n_features=None, plot_size=(15, 15), return_dataframe=False):
    """
    Plots the scaled feature importance for an ensemble model. Returns a data frame of these if requested.
    
    Args:
        model: The trained scikit-learn, LightGBM, or xgboost ensemble model
        features (pd.DataFrame): The training features
        top_n_features (int, optional): The number of features to plot
        plot_size (tuple): The x/y size of the output plot
        return_dataframe (bool): Whether or not to output a data frame in the results
    
    Returns:
        If return_dataframe is True, returns a pandas DataFrame of the feature importances.
        Otherwise, returns None.
    """
    # Putting the feature importances into a data frame
    feature_importances = pd.DataFrame(model.feature_importances_,
                                       index=features.columns,
                                       columns=['Feature Importance']).sort_values('Feature Importance',
                                                                                   ascending=False)
    # Calculating the scaled importances and adding it as a column
    scaled_feature_importances = 100.0 * (feature_importances['Feature Importance'] / feature_importances['Feature Importance'].max())
    feature_importances['Scaled Feature Importance'] = scaled_feature_importances
    
    # Plotting the feature importances
    plt.figure(figsize=plot_size)
    if top_n_features != None:
        feature_importances['Scaled Feature Importance'][:top_n_features].sort_values(ascending=True).plot.barh()
    else:
        feature_importances['Scaled Feature Importance'].sort_values(ascending=True).plot.barh()
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    
    # Returning the feature importance data frame if requested
    if return_dataframe == True:
        return feature_importances


# Visualizing the decision tree
def plot_decision_tree(model, feature_names=None) -> None:
    """
    Plots the decision tree from a scikit-learn DecisionTreeClassifier or DecisionTreeRegressor
    Requires graphviz: https://www.graphviz.org

    Args:
        model (ensemble model): The trained decision tree model.
        feature_names (list, optional): The names of the features. Defaults to None.

    Returns:
        None

    Notes:
        - The Gini score is the level of "impurity" of the node. 
            - Scores closer to 0.5 are more mixed, whereas scores closer to 0 are more homogenous
        - For classification, the colors correspond to different classes
            - The shades are determined by the Gini score. Nodes closer to 0.5 will be lighter.
        - Values contain the number of samples in each category
    """
    from sklearn.externals.six import StringIO  
    from IPython.display import Image  
    from sklearn.tree import export_graphviz
    import pydotplus

    dot_data = StringIO()
    
    export_graphviz(model, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,
                    feature_names=feature_names)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
    display(Image(graph.create_png()))
    
    
# Visualizing feature importance with SHAP (SHapely Additive exPlanations)
def explain_features_shap(model, features, max_features_to_show=15) -> None:
    """
    Visualizes feature importance using SHAP (SHapely Additive exPlanations).

    Args:
        model: The trained model for which feature importance is explained using SHAP values.
        features (pd.DataFrame or np.ndarray): The input features used for model predictions.
        max_features_to_show (int, optional): The maximum number of features to show individual SHAP values for.

    Returns:
        None

    Notes:
        - The function requires the SHAP library to be installed. You can install it via pip: `pip install shap`.
        - SHAP (SHapely Additive exPlanations) is a game-theoretic approach to explain the output of any machine learning model.
        - The function uses a TreeExplainer to compute SHAP values for the model's predictions.
        - The overall feature importance is visualized with a bar plot, and the detailed feature importance is shown with a structured scatter plot.
        - If the number of features is small (<= max_features_to_show), the function displays individual SHAP values for each feature.

    """
    import shap
    
    # Explaining the model's predictions using SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)
    
    # Summarizing the effect of all features
    print('Overall Feature Importance')
    shap.summary_plot(shap_values, features, plot_type="bar")  # Bar plot of feature importance
    
    print('-----------------------------------------------------------------')
    print('Detailed Feature Importance')
    shap.summary_plot(shap_values, features)  # Structured scatter plot
    
    # Showing the effect of each feature across the whole dataset if there are not too many features
    if features.shape[1] <= max_features_to_show:
        print('-----------------------------------------------------------------')
        print('SHAP values for individual features')
        for name in features.columns:
            shap.dependence_plot(name, shap_values, features, display_features=features)

            
# Confusion matrix
def plot_confusion_matrix(label: (np.ndarray, predictions: (np.ndarray, classes, normalize: bool = False) -> None:
    """
    Plots the confusion matrix to visualize the performance of a classification model.

    Args:
        label (np.ndarray): True labels of the data.
        predictions (np.ndarray): Predicted labels from the classification model.
        classes (list): List of class labels.
        normalize (bool, optional): Whether to normalize the confusion matrix. Defaults to False.

    Returns:
        None

    Notes:
        - The function requires the scikit-learn library to be installed. You can install it via pip: `pip install scikit-learn`.
        - The confusion matrix provides a tabular summary of the performance of a classification model.
        - The rows of the matrix correspond to the true labels, and the columns correspond to the predicted labels.
        - The diagonal elements represent the correctly classified instances, while off-diagonal elements represent misclassifications.
        - If `normalize` is set to True, the confusion matrix is normalized by dividing each row by the sum of its elements.
        - The resulting matrix shows the distribution of predicted classes relative to the true classes.
    """
    import itertools
    from sklearn import metrics
    
    # Calculating the confusion matrix
    cm = metrics.confusion_matrix(label, predictions)
    
    # Normalizing if specified
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix')

    print(cm)

    # Plotting the confusion matrix
    plt.figure(figsize=(7, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Adding the text properties to the graph
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment='center',
            color='white' if cm[i, j] > thresh else 'black',
        )

    # Additional graph properties
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Calibration curves and histograms of predicted probabilities
def plot_calibration_curves(models: dict, X: np.ndarray, y: np.ndarray, n_bins: int=10) -> None:
    """
    Plot combined calibration curves and separate shorter histograms of predicted probabilities for multiple classifiers.

    Parameters:
    models (dict): A dictionary where keys are classifier names and values are classifier instances.
    X (array-like): Feature matrix (test data).
    y (array-like): Target vector (test data).
    n_bins (int): Number of bins to discretize the [0, 1] interval.

    Returns:
    None
    """
    from matplotlib import gridspec
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(4, 3, height_ratios=[3, 1, 1, 1])  # Adjust height ratios to make the calibration curve taller
    model_probabilities = {}  # To avoid making predictions multiple times

    # Plot combined calibration curves
    ax0 = plt.subplot(gs[0, :])
    colors = plt.get_cmap('tab10')

    for idx, (name, model) in enumerate(models.items()):
        if hasattr(model, "predict_proba"):
            prob_pos = model.predict_proba(X)[:, 1]
        else:  # use decision function
            prob_pos = model.decision_function(X)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        model_probabilities[name] = prob_pos
        fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pos, n_bins=n_bins)

        ax0.plot(mean_predicted_value, fraction_of_positives, "s-", label=f"{name}", color=colors(idx))

    ax0.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    ax0.set_xlabel("Mean predicted value")
    ax0.set_ylabel("Fraction of positives")
    ax0.legend()
    ax0.set_title('Combined Calibration plots (reliability curve)')
    ax0.grid(True)

    # Plot separate shorter histograms
    for idx, (name, model) in enumerate(models.items()):
        ax = plt.subplot(gs[(idx // 3) + 1, idx % 3])
        ax.hist(model_probabilities[name], range=(0, 1), bins=n_bins, histtype="stepfilled", lw=2, color=colors(idx))
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Frequency")
        ax.set_title(f'Histogram of predicted probabilities ({name})')
        ax.grid(True)

    plt.tight_layout()
    plt.show()
