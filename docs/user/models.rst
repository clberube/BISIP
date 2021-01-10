The Inversion Class
===================

Standard usage of ``bisip`` involves instantiating an
:class:`Inversion` object. This is normally done by invoking one of the
inversion model classes described in the SIP models section below.

.. autoclass:: bisip.models.Inversion
    :members:
    :show-inheritance:

SIP models
----------

Pelton Cole-Cole
^^^^^^^^^^^^^^^^

.. autoclass:: bisip.models.PeltonColeCole
    :members:
    :show-inheritance:

Polynomial decomposition
^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: bisip.models.PolynomialDecomposition
    :members:
    :show-inheritance:

Dias (2000)
^^^^^^^^^^^

.. autoclass:: bisip.models.Dias2000
    :members:
    :show-inheritance:

Shin (2015)
^^^^^^^^^^^

.. autoclass:: bisip.models.Shin2015
    :members:
    :show-inheritance:

Plotting methods
----------------
These functions may be called as methods of the :class:`Inversion` class
after fitting the model to a dataset.

.. autoclass:: bisip.plotlib.plotlib
    :members:

Utility methods
----------------
These utility functions may be called as methods of the :class:`Inversion`
class.

.. autoclass:: bisip.utils.utils
    :members:
