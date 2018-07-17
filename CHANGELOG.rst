Changelog
=========

1.2.0
-----

Features:

- When reading CSV files, assume columns in same order as defined in class if not given in file.

1.1.0
-----

Features:

- Add basic feature type validation, and coercion, function.
- Add test/train split parameters to ``Model.train``.

Bug fixes:

- Allow use of non-orderable types as ``Label`` labels.
