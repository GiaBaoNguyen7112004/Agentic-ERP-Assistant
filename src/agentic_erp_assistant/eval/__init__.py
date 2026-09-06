"""Measured quality, not assumed quality.

Everything else in this project can be argued about from the code. Whether
retrieval actually finds the right document, and whether it actually refuses the
ones it should, can only be settled by running it against cases somebody wrote
down in advance and counting.

This package holds those cases and the harness that scores them. It starts with
retrieval, because retrieval is what the rest of the answer rests on: a grounded
answer built from the wrong passages is wrong in a way no answer-level rubric
can detect, since it will be fluent, cited, and false.
"""
