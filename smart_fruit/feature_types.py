from abc import ABCMeta
from collections import namedtuple

from pandas import Series


class FeatureClassMixin:
    @classmethod
    def from_json(cls, json):
        return cls(**{
            key: value
            for key, value in json.items()
            if key in cls._fields
        })

    def to_json(self):
        return self._asdict()


class FeatureClassMeta(type):
    def __new__(cls, name, bases, namespace):
        base_feature_type = bases[0]
        features = tuple(
            key
            for key, value in base_feature_type.__dict__.items()
            if isinstance(value, FeatureType)
        )

        return type.__new__(
            cls,
            name,
            tuple(bases) + (namedtuple(base_feature_type.__name__, features), FeatureClassMixin,),
            namespace
        )

    def __iter__(self):
        for field_name in self._fields:
            yield getattr(self, field_name)

    def __len__(self):
        return len(self._fields)


class FeatureType(metaclass=ABCMeta):
    _index = None
    feature_count = 1

    def __get__(self, instance, owner):
        if instance is None:
            return self

        if self._index is None:
            self._index = next(
                i
                for i, name in enumerate(owner._fields)
                if getattr(owner, name) is self
            )

        return instance[self._index]

    def to_series(self, value):
        return Series([value])

    def from_series(self, features):
        return features[0]


class Number(FeatureType):
    pass


class Label(FeatureType, namedtuple('Label', ['labels'])):
    @property
    def feature_count(self):
        return len(self.labels)

    def to_series(self, value):
        return Series([int(value == label) for label in self.labels])

    def from_series(self, features):
        return max(zip(features, enumerate(self.labels)))[1][1]
