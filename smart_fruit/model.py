from csv import DictReader

from pandas import DataFrame, concat

from sklearn import linear_model

from smart_fruit.feature_types import FeatureClassMeta
from smart_fruit.model_selection import train_test_split


class ModelMeta(type):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)

        for feature_type in ['Input', 'Output']:
            feature_class = getattr(cls, feature_type)

            setattr(cls, feature_type, FeatureClassMeta(feature_class.__name__, (feature_class,), {}))


class Model(metaclass=ModelMeta):
    model_class = linear_model.LinearRegression

    class Input:
        pass

    class Output:
        pass

    def __init__(self, *args, **kwargs):
        self.model = self.model_class(*args, **kwargs)

    @classmethod
    def input_features_from_list(cls, lists):
        for l in lists:
            yield cls.Input(*l).validate()

    @classmethod
    def input_features_from_json(cls, json):
        for feature in json:
            yield cls.Input.from_json(feature).validate()

    @classmethod
    def input_features_from_csv(cls, csv_path):
        with open(csv_path) as csv_file:
            yield from cls.input_features_from_json(DictReader(csv_file))

    @classmethod
    def features_from_list(cls, lists):
        for l in lists:
            yield cls.Input(*l[:len(cls.Input._fields)]).validate(), cls.Output(*l[len(cls.Input._fields):]).validate()

    @classmethod
    def features_from_json(cls, json):
        for feature in json:
            yield cls.Input.from_json(feature).validate(), cls.Output.from_json(feature).validate()

    @classmethod
    def features_from_csv(cls, csv_path):
        with open(csv_path) as csv_file:
            yield from cls.features_from_json(DictReader(csv_file))

    @staticmethod
    def _to_raw_features(dataframe, feature_class):
        return concat([
            column.apply(feature_type.to_series)
            for (i, column), feature_type in zip(dataframe.iteritems(), feature_class)
        ], axis=1)

    def _dataframes_from_features(self, features):
        dataframe = DataFrame(list(input_) + list(output) for input_, output in features)

        input_dataframe = self._to_raw_features(dataframe, self.Input)
        output_dataframe = self._to_raw_features(dataframe.loc[:, len(self.Input):], self.Output)

        return input_dataframe, output_dataframe

    @classmethod
    def train(cls, features, train_test_split_ratio=None, test_sample_count=None):
        if train_test_split_ratio is not None or test_sample_count is not None:
            train_features, test_features = train_test_split(
                features,
                train_test_split_ratio=train_test_split_ratio,
                test_sample_count=test_sample_count
            )

            model = cls.train(train_features)

            return model, model.score(test_features)

        model = cls()

        model.model.fit(*model._dataframes_from_features(features))

        return model

    def score(self, features):
        return self.model.score(*self._dataframes_from_features(features))

    @staticmethod
    def _chunk_dataframe(dataframe, feature_types):
        start = 0
        for feature_type in feature_types:
            yield dataframe.loc[:, start:start + feature_type.feature_count - 1], feature_type
            start += feature_type.feature_count

    def predict(self, input_features):
        raw_features = self._to_raw_features(DataFrame(input_features), self.Input)

        raw_prediction_dataframe = DataFrame(self.model.predict(raw_features))

        prediction_dataframe = concat([
            chunk.apply(feature_type.from_series, axis=1)
            for chunk, feature_type in self._chunk_dataframe(raw_prediction_dataframe, self.Output)
        ], axis=1)

        for _, output_series in prediction_dataframe.iterrows():
            yield self.Output(*output_series)
