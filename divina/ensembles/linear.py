from pyspark.ml.regression import LinearRegression
from pyspark.ml.pipeline import Pipeline, Estimator, Transformer
from divina.preprocessing.base import MeanFiller


class GLASMA(Estimator):
    def __init__(self, feature_col, label_col):
        self.feature_col = feature_col
        self.label_col = label_col
        self.lr = LinearRegression(featuresCol=feature_col, labelCol=label_col)
        self.mf = MeanFiller(input_col=label_col)
        self.fit_pipeline = None
        self.coefficients = None
        super(GLASMA, self).__init__()

    def _fit(self, df):
        return GLASMAModel(feature_col=self.feature_col, label_col=self.label_col, lr=self.lr, mf=self.mf, df=df)

    def __repr__(self):
        return repr('GLASMA: {}'.format(','.join(['{}: {}'.format(key, self.__dict__[key]) for key in self.__dict__])))


class GLASMAModel(Transformer):
    def __init__(self, feature_col, label_col, lr, mf, df):
        self.feature_col = feature_col
        self.label_col = label_col
        self.lr = lr
        self.mf = mf
        self.pipeline = Pipeline(stages=[self.mf, self.lr]).fit(df)
        self.fit_pipeline = None
        super(GLASMAModel, self).__init__()

    def _transform(self, df):
        return self.pipeline.transform(df)

    def __repr__(self):
        return repr(
            'GLASMAModel: {}'.format(','.join(['{}: {}'.format(key, self.__dict__[key]) for key in self.__dict__])))
