from smart_fruit import Model
from smart_fruit.feature_types import Number


class TrivialModel(Model):
    class Input:
        input_ = Number()

    class Output:
        output = Number()


def main():
    multiply_by_10 = TrivialModel.train([
        (TrivialModel.Input(n), TrivialModel.Output(10 * n))
        for n in [1, 2, 4, 5, 7, 8, 10]
    ])

    inputs = [3, 6, 9]
    predictions = multiply_by_10.predict([TrivialModel.Input(n) for n in inputs])
    for input_, predicted_output in zip(inputs, predictions):
        print(input_, predicted_output.output)


if __name__ == "__main__":
    main()
