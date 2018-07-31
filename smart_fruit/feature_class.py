from collections import namedtuple

from smart_fruit.feature_types import FeatureType

__all__ = ["FeatureClassMeta"]


class FeatureClassMixin:
    def validate(self):
        return self.__class__(*(
            feature_type.validate(value)
            for feature_type, value in zip(self.__class__, self)
        ))

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
