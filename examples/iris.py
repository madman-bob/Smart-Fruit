from smart_fruit import Model
from smart_fruit.feature_types import Number, Label


class Iris(Model):
    """
    Example class for the "Iris" data set:

    https://archive.ics.uci.edu/ml/datasets/Iris
    """

    class Input:
        sepal_length_cm = Number()
        sepal_width_cm = Number()
        petal_length_cm = Number()
        petal_width_cm = Number()

    class Output:
        iris_class = Label(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])


def main():
    features = list(Iris.features_from_csv('iris_data.csv'))

    model = Iris.train(features)

    input_features = [input_feature for input_feature, output_feature in features]

    for (input_, output), predicted_output in zip(features, model.predict(input_features)):
        print(list(input_), output.iris_class, predicted_output.iris_class)


if __name__ == "__main__":
    main()
