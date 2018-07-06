Smart Fruit
===========

A Python schema-based machine learning library, wrapping ``scikit-learn``, for creating quick and easy machine learning models.

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

First, we create a schema describing our inputs and outputs:

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

with data file ``iris_data.csv``

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

- ``Model.Input``, ``Model.Output``

  ``Model.Input`` and ``Model.Output`` are the schemas for defining your input and output features.
  Define them as classes with ``FeatureType`` attributes.

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

- ``Model.features_from_list(lists)``, ``Model.input_features_from_list(lists)``

  Deserialize an iterable of lists into an iterable of the appropriate feature objects.

  eg.

  .. code:: python

    >>> list(Iris.features_from_list([[5.1, 3.5, 1.4, 0.2, 'Iris-setosa']]))
    [(Input(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2), Output(iris_class='Iris-setosa'))]

    >>> list(Iris.input_features_from_list([[5.1, 3.5, 1.4, 0.2]]))
    [Input(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2)]

- ``Model.features_from_json(json)``, ``Model.input_features_from_json(json)``

  Deserialize an iterable of dictionaries into an iterable of the appropriate feature objects.

  eg.

  .. code:: python

    >>> list(Iris.features_from_json([{'sepal_length_cm': 5.1, 'sepal_width_cm': 3.5, 'petal_length_cm': 1.4, 'petal_width_cm': 0.2, 'iris_class': 'Iris-setosa'}]))
    [(Input(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2), Output(iris_class='Iris-setosa'))]

    >>> list(Iris.input_features_from_json([{'sepal_length_cm': 5.1, 'sepal_width_cm': 3.5, 'petal_length_cm': 1.4, 'petal_width_cm': 0.2}]))
    [Input(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2)]

- ``Model.features_from_csv(csv_path)``, ``Model.input_features_from_csv(csv_path)``

  Take a path to a CSV file, and deserialize it into an iterable of the appropriate feature objects.

  eg.

  .. code:: python

    >>> list(Iris.features_from_csv('iris_data.csv'))
    [(Input(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2), Output(iris_class='Iris-setosa')), ...]

    >>> list(Iris.input_features_from_csv('iris_data.csv'))
    [Input(sepal_length_cm=5.1, sepal_width_cm=3.5, petal_length_cm=1.4, petal_width_cm=0.2), ...]

- ``Model.train(features)``

  Train a new model on the given iterable of input/output pairs.

  eg.

  .. code:: python

    >>> iris_model = Iris.train([(Iris.Input(5.1, 3.5, 1.4, 0.2), Iris.Output('Iris-setosa'))])

- ``model.predict(input_features)``

  Predict the outputs for a given iterable of inputs.

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

- ``Label(labels)`` - An enumerated feature, which may take one of a pre-defined list of available values.

  eg. For ``labels = ['red', 'green', 'blue']``, our label may take the value ``'red'``, but not ``'purple'``.

Requirements
------------

Smart Fruit requires Python 3.6+, and uses scikit-learn, scipy, and pandas.

Installation
------------

Install and update using pip:

.. code:: text

    pip install smart-fruit

Donate
------

To support the continued development of Smart Fruit, please
`donate <https://salt.bountysource.com/checkout/amount?team=smart-fruit>`_.
