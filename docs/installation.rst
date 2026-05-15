.. _installation:

Installation
============
`PHRINGE` requires Python **>=3.10** to run.

.. _pip_install:

Installation From PyPI (Recommended)
------------------------------------

To install `PHRINGE` from PyPI, run the following command in your terminal:

.. code-block:: console

    pip install phringe


Installation From GitHub
------------------------
To install `PHRINGE` from GitHub, run the following command in your terminal:

.. code-block:: console

    pip install git+https://github.com/pahuber/PHRINGE


Alternatively, the repository can be cloned from GitHub and then installed using:

.. code-block:: console

    git clone https://github.com/pahuber/PHRINGE.git
    cd PHRINGE
    pip install .


Test Installation
-----------------

You can test the installation in a Python console with:

.. code-block:: python

    from phringe.util.installation import which_animals_have_fringes

    which_animals_have_fringes()

If everything worked well, you should get an approprioate answer.