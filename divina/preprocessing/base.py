from pyspark.ml.feature import OneHotEncoder, StringIndexer
from pyspark.ml import Pipeline
from pyspark.ml.pipeline import Transformer
import pyspark.sql.functions as F


class CategoricalEncoder(Transformer):

    def __init__(self, input_cols):
        self.input_cols = input_cols
        super(CategoricalEncoder, self).__init__()

    def _transform(self, df):
        string_indexers = [StringIndexer(inputCol=c, outputCol="{}_index".format(c)) for c in
                           self.input_cols]
        one_hot_encoders = [
            OneHotEncoder(dropLast=False, inputCol="{}_index".format(c), outputCol="{}_ohe_vec".format(c)) for
            c in
            self.input_cols]
        preprocessing_pipeline = Pipeline(stages=string_indexers + one_hot_encoders)
        return preprocessing_pipeline.fit(df).transform(df)


class MeanFiller(Transformer):
    def __init__(self, input_col):
        self.input_col = input_col
        super(MeanFiller, self).__init__()

    def _transform(self, df):
        return df.na.fill(
            df.select(F.mean(df[self.input_col])).collect()[0][0],
            subset=[self.input_col])

