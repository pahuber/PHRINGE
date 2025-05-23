.. _installation:

Installation
============

Prerequisites
-------------
* | **Python Installation:**
  | `PHRINGE` requires Python **3.10** or **3.11** to run. If you do not have Python installed, you can download it `here <https://www.python.org/downloads/>`_.
* | **Virtual Environment:**
  | We recommend installing `PHRINGE` in a virtual environment to avoid conflicts with other Python packages. For instructions on how to create and activate a virtual environment, see the `virtualenv user guide <https://virtualenv.pypa.io/en/latest/user_guide.html>`_.

.. _pip_install:

Installation From PyPI (Recommended)
------------------------------------

To install `PHRINGE` from PyPI, run the following command in your terminal:

.. code-block:: console

    pip install phringe

You can test the installation in a Python console with:

.. code-block:: python

    from phringe.util.installation import which_animals_have_fringes

    which_animals_have_fringes()

If everything worked well, you should get an approprioate answer.

Installation From GitHub
------------------------
To install `PHRINGE` from GitHub, run the following command in your terminal:

.. code-block:: console

    pip install git+https://github.com/pahuber/PHRINGE


Alternatively, the repository can be cloned from GitHub using:

.. code-block:: console

    git clone https://github.com/pahuber/PHRINGE.git

After navigating to the cloned repository, the package can be installed using:

.. code-block:: console

    pip install .