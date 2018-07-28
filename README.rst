Smart Fruit
===========

Purpose
-------

A Python machine learning library, for creating quick and easy machine learning models.
It is schema-based, and wraps `scikit-learn <http://scikit-learn.org/stable/>`_.

Usage
-----

Create and use a machine learning model in 3 steps:

1. Create a schema representing your input and output features.
2. Train a model from your data.
3. Make predictions from your model.

Example
-------

To get a feel for the library, consider the classic `Iris <https://archive.ics.uci.edu/ml/datasets/Iris>`_ dataset,
where we predict the class of iris plant from measurements of the sepal, and petal.

First, we create a schema describing our inputs and outputs.
For our inputs, we have the length, and width, of both the sepal, and the petal.
All of these input values happen to be numbers.
For our output, we have just the class of iris, which may be one of the labels ``Iris-setosa``, ``Iris-versicolor``, or ``Iris-virginica``.

We define this in code as follows:

.. code:: python

    from smart_fruit import Model
    from smart_fruit.feature_types import Number, Label


    class Iris(Model):
        class Input:
            sepal_length_cm = Number()
            sepal_width_cm = Number()
            petal_length_cm = Number()
            petal_width_cm = Number()

        class Output:
            iris_class = Label(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])

Then, we train a model:

.. code:: python

    model = Iris.train(Iris.features_from_csv('iris_data.csv'))

with data file `iris_data.csv <https://github.com/madman-bob/Smart-Fruit/blob/master/examples/iris_data.csv>`_.

::

    sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm,iris_class
    5.1,3.5,1.4,0.2,Iris-setosa
    ...

Finally, we use our new model to make predictions:

.. code:: python

    for prediction in model.predict([Iris.Input(5.1, 3.5, 1.4, 0.2)]):
        print(prediction.iris_class)

Reference
---------

Models
~~~~~~

- ``Model.Input`` - Schema for defining your input features.

- ``Model.Output`` - Schema for defining your output features.

  Define ``Model.Input`` and ``Model.Output`` as classes with ``FeatureType`` attributes.

  eg. Consider the ``Iris`` class defined above.

  These classes can then be used to create objects representing the appropriate collections of features.

  eg.

  .. code:: python

    >>> iris_input = Iris.Input(5.1, 3.5, 1.4, 0.2)
    >>> iris_input
    Input(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2)
    >>> iris_input.sepal_length
    5.1

    >>> Iris.Input.from_json({'sepal_length_cm': 5.1, 'sepal_width_cm': 3.5, 'petal_length_cm': 1.4, 'petal_width_cm': 0.2})
    Input(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2)

- ``Model.features_from_list(lists)`` - Deserialize an iterable of lists into an iterable of input/output feature pairs.

  eg.

  .. code:: python

    >>> list(Iris.features_from_list([[5.1, 3.5, 1.4, 0.2, 'Iris-setosa']]))
    [(Input(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2), Output(iris_class='Iris-setosa'))]

- ``Model.input_features_from_list(lists)`` - Deserialize an iterable of lists into an iterable of input features.

  eg.

  .. code:: python

    >>> list(Iris.input_features_from_list([[5.1, 3.5, 1.4, 0.2]]))
    [Input(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2)]

- ``Model.features_from_json(json)`` - Deserialize an iterable of dictionaries into an iterable of input/output feature pairs.

  eg.

  .. code:: python

    >>> list(Iris.features_from_json([{'sepal_length_cm': 5.1, 'sepal_width_cm': 3.5, 'petal_length_cm': 1.4, 'petal_width_cm': 0.2, 'iris_class': 'Iris-setosa'}]))
    [(Input(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2), Output(iris_class='Iris-setosa'))]

- ``Model.input_features_from_json(json)`` - Deserialize an iterable of dictionaries into an iterable of input features.

  eg.

  .. code:: python

    >>> list(Iris.input_features_from_json([{'sepal_length_cm': 5.1, 'sepal_width_cm': 3.5, 'petal_length_cm': 1.4, 'petal_width_cm': 0.2}]))
    [Input(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2)]

- ``Model.features_from_csv(csv_path)`` - Take a path to a CSV file, or a file-like object, and deserialize it into an iterable of input/output feature pairs.

  If column headings are not given in the file, assume the input features are followed by the output features, in the order they are defined in their respective classes.

  eg.

  .. code:: python

    >>> list(Iris.features_from_csv('iris_data.csv'))
    [(Input(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2), Output(iris_class='Iris-setosa')), ...]

- ``Model.input_features_from_csv(csv_path)`` - Take a path to a CSV file, or a file-like object, and deserialize it into an iterable of input features.

  If column headings are not given in the file, assume they are in the order they are defined in the ``Input`` class.

  eg.

  .. code:: python

    >>> list(Iris.input_features_from_csv('iris_data.csv'))
    [Input(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2), ...]

- ``Model.model_class`` - How to model the relation between the input and output data.

  Default: ``sklearn.linear_model.LinearRegression``

  This attribute accepts any class with ``fit``, ``predict``, and ``score`` methods defined as for ``scikit-learn`` multi-response regression models.
  In particular, this attribute accepts any ``scikit-learn`` multi-response regression models,
  ie. any ``scikit-learn`` regression model where the ``y`` parameter of ``fit`` accepts a numpy array of shape ``[n_samples, n_targets]``.

- ``Model.train(features, train_test_split_ratio=None, test_sample_count=None)``

  Train a new model on the given iterable of input/output pairs.

  Parameters:

  - ``features`` - An iterable of input/output pairs.

  - ``train_test_split_ratio`` - Proportion of data to use as cross-validation test data.

  - ``test_sample_count`` - Number of samples of data to use as cross-validation test data.

    If ``train_test_split_ratio`` or ``test_sample_count`` are provided, perform cross-validation of the given data.
    Return both the trained model, and the score of the test data on that model.

  eg.

  .. code:: python

    >>> iris_model = Iris.train([(Iris.Input(5.1, 3.5, 1.4, 0.2), Iris.Output('Iris-setosa'))])

- ``model.predict(input_features, yield_inputs=False)`` - Predict the outputs for a given iterable of inputs.

  If ``yield_inputs`` is ``True`` then yield the prediction with the input used to generate it, as ``input``, ``output`` pairs.
  Otherwise, yield just the predictions, in the same order the inputs are given to the model.

  eg.

  .. code:: python

    >>> list(iris_model.predict([Iris.Input(5.1, 3.5, 1.4, 0.2)]))
    [Output(iris_class='Iris-setosa')]

Feature Types
~~~~~~~~~~~~~

Smart Fruit recognizes the following data types for input and output features.
Custom types may be made by extending the ``FeatureType`` class.

- ``Number()`` - A real-valued feature.

  eg. ``0``, ``1``, ``3.141592``, ``-17``, ...

- ``Integer()`` - A whole number feature.

  eg. ``0``, ``1``, ``3``, ``-17``, ...

- ``Complex()`` - A complex-valued number feature.

  eg. ``0``, ``1``, ``3 + 4j``, ``-1 + 7j``, ...

- ``Label(labels)`` - An enumerated feature, ie. one which may take one of a pre-defined list of available values.

  eg. For ``labels = ['red', 'green', 'blue']``, our label may take the value ``'red'``, but not ``'purple'``.

- ``Vector(feature_types)`` - A feature made of other features. Useful for grouping conceptually related features.

  eg. For ``feature_types = [Number(), Label(['red', 'green', 'blue'])]``, we may take values such as ``(0, 'red')``, and ``(1, 'blue')``.

Requirements
------------

Smart Fruit requires Python 3.6+, and uses
`scikit-learn <http://scikit-learn.org/stable/>`_,
`scipy <https://www.scipy.org/>`_,
and `pandas <https://pandas.pydata.org/>`_.

Installation
------------

Install and update using the standard Python package manager `pip <https://pip.pypa.io/en/stable/>`_:

.. code:: text

    pip install smart-fruit

Donate
------

To support the continued development of Smart Fruit, please
`donate <https://salt.bountysource.com/checkout/amount?team=smart-fruit>`_.
