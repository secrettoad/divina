from ..ensembles.linear import GLASMA, GLASMAModel
from ..preprocessing.base import MeanFiller, CategoricalEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SQLContext, DataFrame
import pandas as pd
import os


def test_glasma_fit(spark_context, test_df_1):
    sql_context = SQLContext(spark_context)
    df = sql_context.createDataFrame(test_df_1)
    vector_assembler = VectorAssembler(
        inputCols=list(df.columns[:-1]), outputCol='features')
    model = GLASMA(feature_col='features', label_col=df.columns[-1])
    df = vector_assembler.transform(df)
    fit_model = model.fit(df)
    assert(type(fit_model) == GLASMAModel)


def test_glasma_predict(spark_context, test_df_1):
    sql_context = SQLContext(spark_context)
    df = sql_context.createDataFrame(test_df_1)
    vector_assembler = VectorAssembler(
        inputCols=list(df.columns[:-1]), outputCol='features')
    model = GLASMA(feature_col='features', label_col=df.columns[-1])
    df = vector_assembler.transform(df)
    df = model.fit(df).transform(df)
    assert(type(df) == DataFrame and df.columns == ['a', 'b', 'c', 'features', 'prediction'])


def test_glasma_output_1(spark_context, test_df_1):
    sql_context = SQLContext(spark_context)
    df = sql_context.createDataFrame(test_df_1)
    vector_assembler = VectorAssembler(
        inputCols=list(df.columns[:-1]), outputCol='features')
    model = GLASMA(feature_col='features', label_col=df.columns[-1])
    df = vector_assembler.transform(df)
    df = model.fit(df).transform(df)
    pd.testing.assert_frame_equal(df.toPandas(), pd.read_pickle(os.path.join('../mock', 'test_glasma_output_1.pickle')))


def test_mean_filler_transform(spark_context, test_df_1):
    sql_context = SQLContext(spark_context)
    df = sql_context.createDataFrame(test_df_1)
    mf = MeanFiller(input_col='a')
    df = mf.transform(df)
    assert (type(df) == DataFrame and df.columns == ['a', 'b', 'c'])


def test_mean_filler_output_1(spark_context, test_df_2):
    sql_context = SQLContext(spark_context)
    df = sql_context.createDataFrame(test_df_2)
    mf = MeanFiller(input_col='a')
    df = mf.transform(df)
    pd.testing.assert_frame_equal(df.toPandas(), pd.read_pickle(os.path.join('../mock', 'test_mean_filler_output_1.pickle')))


def test_categorical_encoder_transform(spark_context, test_df_3):
    sql_context = SQLContext(spark_context)
    df = sql_context.createDataFrame(test_df_3)
    ce = CategoricalEncoder(input_cols=['a', 'b', 'c'])
    df = ce.transform(df)
    assert (type(df) == DataFrame and df.columns == ['a', 'b', 'c', 'a_index', 'b_index', 'c_index', 'a_ohe_vec', 'b_ohe_vec', 'c_ohe_vec'])


def test_categorical_encoder_output_1(spark_context, test_df_3):
    sql_context = SQLContext(spark_context)
    df = sql_context.createDataFrame(test_df_3)
    ce = CategoricalEncoder(input_cols=['a', 'b', 'c'])
    df = ce.transform(df)
    pd.testing.assert_frame_equal(df.toPandas(), pd.read_pickle(os.path.join('../mock', 'test_category_encoder_output_1.pickle')))

