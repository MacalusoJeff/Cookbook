from pyspark.sql import functions as F

# Num missing values by column
df.select([F.count(F.when(F.isnan(c), c)).alias(c) for c in df.columns]).show()

# Filling missing values for specified
df.fillna(0, subset=['a', 'b'])

# String indexing multiple columns
from pyspark.ml.feature import StringIndexer
categorical_cols = ['col_1', 'col_2']
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df) for column in categorical_cols]
pipeline = Pipeline(stages=indexers)
df = pipeline.fit(df).transform(df)
