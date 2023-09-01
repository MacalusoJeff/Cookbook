import numpy as np
import pandas as pd

def reduce_memory_usage(df):
    """
    Reduces memory usage of a Pandas dataframe by dynamically choosing the lowest integer and floating-point types.

    Parameters:
        df (pd.DataFrame): The dataframe to optimize.

    Returns:
        pd.DataFrame: The optimized dataframe.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_data = df[col]
        col_type = col_data.dtype

        if col_type == 'object':
            col_data = col_data.astype('category')
        elif col_type.name == 'category':
            col_data = col_data.astype('category')
        elif issubclass(col_type.type, np.integer):
            col_data = pd.to_numeric(col_data, downcast='integer')
        elif issubclass(col_type.type, np.floating):
            col_data = pd.to_numeric(col_data, downcast='float')

        df[col] = col_data

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    print(f"Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%")

    return df

# One-hot encoding multiple columns
df_encoded = pd.get_dummies(df, columns=['a', 'b', 'c'], drop_first=False)

# Converting a categorical column to numbers
df['TargetVariable'].astype('category').cat.codes

# Scaling from 0 to 1
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)  # Scaling from 0 to 1 across columns

# Normalizing w/ standard scaling
from sklearn import preprocessing
normalizer = preprocessing.StandardScaler()
X_normalized = normalizer.fit_transform(X)  # Normalizing across columns

# Grouping by multiple levels and getting the percentage of the second level by the first level
df.groupby(['level_1', 'level_2']).size().groupby(level='level_1').apply(lambda x: x / x.sum())

# Principal Component Analysis (PCA)
def fit_PCA(X: np.ndarray, num_components: float or int = 0.99) -> np.ndarray:
    """
    Performs min-max normalization and PCA transformation on the input data array.

    Args:
        X (np.ndarray): Values to perform PCA on.
        num_components (float or int):
            If <1, the percentage of variance explained.
            If >1, the number of principal components.

    Returns:
        np.ndarray: The principal components.
    """
    from sklearn import preprocessing
    from sklearn.decomposition import PCA

    # Checking if the input is a numpy array and converting it if not
    if type(X) != np.ndarray:
        X = np.array(X)

    # Normalizing data before PCA if the data isn't already scaled or normalized
    if X.mean() != 0 and X.max() != 1:
        min_max_scaler = preprocessing.MinMaxScaler()
        X_norm = min_max_scaler.fit_transform(X)

    # Performing PCA
    pca = PCA(n_components=num_components)
    pca.fit(X_norm)

    # Reporting explained variance
    explained_variance = pca.explained_variance_ratio_ * 100
    print('Total variance % explained:', sum(explained_variance))
    print()
    print('Variance % explained by principal component:')
    for principal_component in range(len(explained_variance)):
        print(principal_component, ':', explained_variance[principal_component])

    # Transforming the data before returning
    principal_components = pca.transform(X_norm)
    return principal_components


# Oversampling
def oversample_binary_label(dataframe: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """
    Oversamples a dataframe with a binary label to have an equal proportion in classes. Dynamically
    determines the label with the lower proportion.

    Args:
        - dataframe (pd.DataFrame): A dataframe containing the label.
        - label_column (str): The column containing the label.

    Returns:
        pd.DataFrame: Dataframe with the lower proportion label oversampled.

    TODO: Update this to oversample the training set and return both the training and testing sets.
    """

    # Counting the classes
    class_0_count, class_1_count = dataframe[label_column].value_counts()

    # Creating two dataframes for each class
    dataframe_class_0 = dataframe[dataframe[label_column] == dataframe[label_column].unique()[0]]
    dataframe_class_1 = dataframe[dataframe[label_column] == dataframe[label_column].unique()[1]]

    # Determining the smaller class
    smaller_label = dataframe[label_column].value_counts().idxmin()

    # Oversampling
    if smaller_label == 0:
        dataframe_class_0_oversampled = dataframe_class_0.sample(class_1_count, replace=True)
        dataframe_oversampled = pd.concat([dataframe_class_1, dataframe_class_0_oversampled], axis=0)
    else:
        dataframe_class_1_oversampled = dataframe_class_1.sample(class_0_count, replace=True)
        dataframe_oversampled = pd.concat([dataframe_class_0, dataframe_class_1_oversampled], axis=0)

    # Printing results
    print('Initial number of observations in each class:')
    print(dataframe[label_column].value_counts())
    print()

    print('Oversampled number of observations in each class:')
    print(dataframe_oversampled[label_column].value_counts())

    return dataframe_oversampled

# Oversampling with SMOTE
def oversample_smote(training_features: pd.DataFrame or np.ndarray, training_labels: pd.DataFrame or np.ndarray) -> list:
    """
    Convenience function for oversampling with SMOTE. This generates synthetic samples via interpolation.
    Automatically encodes categorical columns if a dataframe is provided with categorical columns properly marked.

    Args:
        training_features (pd.DataFrame or np.ndarray): The features that will be used to predict the label.
        training_labels (pd.DataFrame or np.ndarray): The labels for the training set of the data.

    Returns:
        list: A list containing the oversampled training features and labels.
    """
    from imblearn import over_sampling

    if isinstance(training_features, pd.DataFrame):
        # Testing if there are any categorical columns
        # Note: These must have the "category" datatype
        categorical_variable_list = training_features.select_dtypes(exclude=['number', 'bool_', 'object_']).columns
        if categorical_variable_list.shape[0] > 0:
            categorical_variable_list = list(categorical_variable_list)
            categorical_variable_indexes = training_features.columns.get_indexer(categorical_variable_list)
            smote = over_sampling.SMOTENC(categorical_features=categorical_variable_indexes, random_state=46, n_jobs=-1)
        else:
            smote = over_sampling.SMOTE(random_state=46, n_jobs=-1)
    else:
        smote = over_sampling.SMOTE(random_state=46, n_jobs=-1)

    # Performing oversampling
    training_features_oversampled, training_labels_oversampled = smote.fit_sample(training_features, training_labels)

    # Rounding discrete variables for appropriate cutoffs
    # This is becuase SMOTE NC only deals with binary categorical variables, not discrete variables
    if isinstance(training_features, pd.DataFrame):
        discrete_variable_list = training_features.select_dtypes(include=['int', 'int32', 'int64']).columns
        if discrete_variable_list.shape[0] > 0:
            discrete_variable_indexes = training_features.columns.get_indexer(discrete_variable_list)
            for discrete_variable_index in discrete_variable_indexes:
                training_features_oversampled[:, discrete_variable_index] = np.round(training_features_oversampled[:, discrete_variable_index].astype(float)).astype(int)

    print('Previous training size:', len(training_labels))
    print('Oversampled training size', len(training_labels_oversampled), '\n')
    print('Previous label mean:', training_labels.astype(int).mean())
    print('Oversampled label mean:', training_labels_oversampled.mean())

    return training_features_oversampled, training_labels_oversampled

X_train_oversampled, y_train_oversampled = oversample_smote(X_train, y_train)


# Target mean encoding
def target_encode(train_variable: np.ndarray, test_variable: np.ndarray, train_label: np.ndarray,
                  smoothing: int = 1, min_samples_leaf: int = 1, noise_level: int = 0) -> list:
    """
    Mean target encoding using Daniele Micci-Barreca's technique from the following paper:
    http://helios.mm.di.uoa.gr/~rouvas/ssi/sigkdd/sigkdd.vol3.1/barreca.pdf

    This function heavily borrows code from Olivier's Kaggle post:
    https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features

    Args:
        - train_variable (Series): Variable in the training set to perform the encoding on.
        - test_variable (Series): Variable in the testing set to be transformed.
        - train_label (Series): The label in the training set to use for performing the encoding.
        - smoothing (int): Balances the categorical average vs. the prior.
        - min_samples_leaf (int): The minimum number of samples to take the category averages into account.
        - noise_level (int): Amount of Gaussian noise to add in order to help prevent overfitting.

    Returns:
        list: The target encoded training and test variable
    """
    assert len(train_variable) == len(train_label)
    assert train_variable.name == test_variable.name

    # Creating a data frame out of the training variable and label in order to get the averages of the label
    # for the training variable
    temp = pd.concat([train_variable, train_label], axis=1)

    # Computing the target mean
    averages = temp.groupby(train_variable.name)[train_label.name].agg(['mean', 'count'])

    # Computing the smoothing
    smoothing = 1 / (1 + np.exp(-(averages['count'] - min_samples_leaf) / smoothing))

    # Calculating the prior before adding the smoothing
    prior = train_label.mean()

    # Adding the smoothing to the prior to get the posterior
    # Larger samples will take the average into account less
    averages[train_label.name] = prior * (1 - smoothing) + averages['mean'] * smoothing

    # Applying the averages to the training variable
    fitted_train_variable = pd.merge(
        train_variable.to_frame(train_variable.name),
        averages.reset_index().rename(columns={'index': train_label.name, train_label.name: 'average'}),
        on=train_variable.name, how='left')
    fitted_train_variable = fitted_train_variable['average'].rename(train_variable.name + '_mean').fillna(prior)
    fitted_train_variable.index = train_variable.index  # Restoring the index lost in pd.merge

    # Applying the averages to the testing variable
    fitted_test_variable = pd.merge(
        test_variable.to_frame(test_variable.name),
        averages.reset_index().rename(columns={'index': train_label.name, train_label.name: 'average'}),
        on=test_variable.name, how='left')
    fitted_test_variable = fitted_test_variable['average'].rename(test_variable.name + '_mean').fillna(prior)
    fitted_test_variable.index = fitted_test_variable.index  # Restoring the index lost in pd.merge

    # Adding the noise if there is any
    if noise_level != 0:
        fitted_train_variable = fitted_train_variable * (1 + noise_level * np.random.randn(len(fitted_train_variable)))
        fitted_test_variable = fitted_test_variable * (1 + noise_level * np.random.randn(len(fitted_test_variable)))
    return fitted_train_variable, fitted_test_variable


# Percentage of total within group
df['numbers'] / df.groupby('group')['numbers'].transform('sum')

# Finding the max number of bins if using pd.qcut()
def find_max_qcut_bins(data: np.ndarray, max_bins: int = 25) -> int:
    """
    Returns the max number of bins for pd.qcut().

    Args:
        data (np.ndarray): What we want the bins for.
        max_bins (int): How many bins we will attempt to go up to.

    Returns:
        int: The maximum number of bins that pd.qcut() will accept for this data.
    """
    found_max_bins = False
    while found_max_bins == False:
        try:
            pd.qcut(data, q=max_bins)
            return max_bins
        except:
            max_bins -= 1
