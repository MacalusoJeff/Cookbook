from pyspark.sql import functions as F

# Num missing values by column
df.select([F.count(F.when(F.isnan(c), c)).alias(c) for c in df.columns]).show()

# Filling missing values for specified columns
df.fillna(0, subset=['a', 'b'])

# Converting all boolean columns to integers
df = df.select([df[column].cast("integer").alias(column) if column_type == "boolean" else df[column] for column, column_type in df.dtypes])

# String indexing multiple columns
from pyspark.ml.feature import StringIndexer
categorical_cols = ['col_1', 'col_2']
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in categorical_cols]
pipeline = Pipeline(stages=indexers)
df = pipeline.fit(df).transform(df)

# Vectorizing features, splitting between training and testing sets
from pyspark.ml.feature import VectorAssembler
cols_to_drop = ['col_1', 'col_2']
feature_cols = [feature for feature in df.columns if feature not in cols_to_drop]
featurizer = VectorAssembler(inputCols=feature_cols, outputCol="features")
train, test = df.randomSplit([0.8, 0.2], seed=46)
train_data = featurizer.transform(train)['label', 'features']
test_data = featurizer.transform(test)['label', 'features']

# Getting model evaluation metrics from a model
from synapse.ml.train import ComputeModelStatistics
predictions = model.transform(test_data)
metrics = ComputeModelStatistics(
    evaluationMetric="classification",
    labelCol="label",
    scoredLabelsCol="prediction",
).transform(predictions)
display(metrics)
