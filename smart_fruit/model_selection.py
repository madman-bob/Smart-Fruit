from sklearn.model_selection import train_test_split as sk_train_test_split


def train_test_split(features, train_test_split_ratio=None, test_sample_count=None):
    if (train_test_split_ratio is None) == (test_sample_count is None):
        raise ValueError(
            "Must provide exactly one of train_test_split_ratio or test_sample_count "
            "to perform train/test split"
        )

    if train_test_split_ratio is not None:
        train_test_split_ratio = float(train_test_split_ratio)

        if train_test_split_ratio <= 0 or train_test_split_ratio >= 1:
            raise ValueError(
                "train_test_split_ratio must be strictly between 0 and 1 (given {})".format(train_test_split_ratio)
            )

    if test_sample_count is not None:
        test_sample_count = round(test_sample_count)

        if test_sample_count <= 0:
            raise ValueError(
                "test_sample_count must be strictly positive (given {})".format(test_sample_count)
            )

    return sk_train_test_split(
        list(features),
        test_size=train_test_split_ratio or test_sample_count
    )
