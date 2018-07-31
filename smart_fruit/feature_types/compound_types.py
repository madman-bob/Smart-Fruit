from pandas import concat

from smart_fruit.feature_types.feature_type_base import FeatureType

__all__ = ["Vector"]


class Vector(FeatureType):
    def __init__(self, feature_types):
        self.feature_types = feature_types

    @property
    def feature_count(self):
        return sum(feature_type.feature_count for feature_type in self.feature_types)

    def validate(self, value):
        if len(value) != len(self.feature_types):
            raise ValueError(
                "Incorrect length vector (expected {}, got {!r})".format(len(self.feature_types), len(value))
            )

        return tuple(
            feature_type.validate(subvalue)
            for subvalue, feature_type in zip(value, self.feature_types)
        )

    def to_series(self, value):
        if len(value) != len(self.feature_types):
            raise ValueError(
                "Incorrect length vector (expected {}, got {!r})".format(len(self.feature_types), len(value))
            )

        return concat([
            feature_type.to_series(subvalue)
            for subvalue, feature_type in zip(value, self.feature_types)
        ], ignore_index=True)

    def from_series(self, features):
        return tuple(
            feature_type.from_series(chunk)
            for chunk, feature_type in self._chunk_series(features, self.feature_types)
        )

    @staticmethod
    def _chunk_series(series, feature_types):
        start = 0
        for feature_type in feature_types:
            yield series.iloc[start:start + feature_type.feature_count].reset_index(drop=True), feature_type
            start += feature_type.feature_count
