Changelog
=========

1.2.1
-----

Bug fixes:

- Fix broken build.

1.2.0
-----

Features:

- When reading CSV files, assume columns in same order as defined in class if not given in file.
- Add the ``Integer``, ``Complex``, ``Vector``, and ``Tag`` feature types.
- Add the ``Model.predict`` ``yield_inputs`` parameter.
- Add the ``Model.train`` ``random_state`` parameter.

Bug fixes:

- Support unicode in CSV files.
- Fix bug which raised an error when predicting a ``Number`` that wasn't the first feature in ``Output``.

1.1.0
-----

Features:

- Add basic feature type validation, and coercion, function.
- Add test/train split parameters to ``Model.train``.

Bug fixes:

- Allow use of non-orderable types as ``Label`` labels.
