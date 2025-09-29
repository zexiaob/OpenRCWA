.. Rigorous Coupled Wave Analysis documentation master file, created by
   sphinx-quickstart on Mon Sep 28 12:56:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Materials in rcwa
==========================================================

.. autoclass:: rcwa.Material

The `Material` class handles materials whose material properties (permittivity, permeability, or refractive index) change as a function of wavelength. There are three primary ways to describe materials:

**1. Constant numerical value**

   This is the simplest model for a material - a constant permittivity / permeability or refractive index, passed in with the `er`, `ur` and `n` arguments.

**2. Custom function of wavelength**

   In addition to being constants, `er`, `ur`, and `n` may be user-specified single-argument functions, which take wavelength as an argument. Units of wavelength in this case do not matter and returns a (potentially complex) scalar.

**3. Tabulated data**

    If a value is passed to the :code:`filename` argument, then the :code:`Material` will load in that file. Csv formats are supported by default, with the first axis being wavelength and the second axis being the complex-valued refractive index. You may instead choose to separate the real- and imaginary parts into two separate columns. This works as well. An example tabulated data file is shown below.

**4. Using a materials database - tabulated or dispersion formula**

    The refractiveindex.info database is supported by default, and contains many user-specified materials (i.e. :code:`Si`, :code:`SiO2`, :code:`Ti`, :code:`Pt`. This database provides both tabulated data and dispersion formulas (see `dispersion formulas <https://refractiveindex.info/database/doc/Dispersion%20formulas.pdf>`_), depending on the material used. CAUTION: if using this database, wavelength units must be specified in micrometers. 

Interpolation and Extrapolation
-------------------------------------
There are two tabulated-data paths with slightly different policies:

- Database/CSV files (filename or database name): linear interpolation is performed automatically for requests inside the tabulated range; linear extrapolation outside the range is allowed and emits a warning.
- User-supplied inline tables (Material(data=...) or TensorMaterial with a dict): interpolation/extrapolation are strictly disabled by default. You must explicitly opt-in via allow_interpolation=True and/or allow_extrapolation=True. Otherwise a ValueError is raised when an off-grid wavelength is requested.

Notes:
- For scalar inline tables in :code:`Material`, you can pass either 'n' or 'er' (optionally 'ur') with a 'wavelength' array.
- For anisotropic inline tables in :code:`TensorMaterial`, you can provide epsilon_* or n_* component arrays (xx, xy, xz, yx, yy, yz, zx, zy, zz) or full [N,3,3] arrays under 'epsilon_tensor' or 'n_tensor'. When 'n' data are provided, interpolation is performed on n first and then squared to epsilon.

Imaginary Sign Convention
----------------------------------------------------------------------
Lossy materials are represented with a positive imaginary refractive index (i.e. :code:`2 + 0.1j`. Materials with gain are represented by a negative imaginary refractive index (i.e. :code:`2 - 0.1j`).

Example Tabulated Data Files
----------------------------------------------------------------------

Below is what an example file for tabulated n/k data should look like, let's call it :code:`custom_material.csv`:

.. code-block::

    wavelength, n, k
    0.76, 2, 0.1
    0.77, 2.1, 0.2
    0.78, 2.1, 0.4

And a valid alternative :code:`custom_material2.csv`:

.. code-block::

    wavelength, n
    0.76, 2 + 0.1j
    0.77, 2.1 + 0.2j
    0.78, 2.1 + 0.4j

Note: the files must have a header.

Material Examples
---------------------------------

.. code-block::

    from rcwa import Material

    # Use a constant index or permittivity
    my_material = Material(n=5 + 0.1j)
    my_material_2 = Material(er=4, ur=1.1)

    # Use a custom function
    def n_func(wavelength):
        return 1 + wavelength / 1.5

    my_dispersive_material = Material(n=n_func)

    # Use built-in databases
    Si = Material('Si')
    SiO2 = Material('SiO2')

    # Use a custom file
    custom_material = Material(filename='custom_filename.csv')

    # User-supplied inline table with strict interpolation/extrapolation flags
    iso_data = {
        'wavelength': np.array([1.50, 1.55, 1.60]),
        'n': np.array([1.45+0j, 1.46+0j, 1.47+0j]),
    }
    # No interpolation/extrapolation unless explicitly allowed
    mat_tab = Material(data=iso_data, allow_interpolation=True, allow_extrapolation=False)


Anisotropic Materials (TensorMaterial)
--------------------------------------

.. autoclass:: rcwa.model.material.TensorMaterial
    :members:

Tabulated anisotropic data examples:

.. code-block:: python

    import numpy as np
    from rcwa.model.material import TensorMaterial

    # Provide n_* components; non-specified off-diagonals default to 0 (identity for missing diagonals)
    ani_table = {
        'wavelength': np.array([1.50, 1.55, 1.60]),
        'n_xx': np.array([1.50, 1.51, 1.52]),
        'n_yy': np.array([1.48, 1.49, 1.50]),
        'n_zz': np.array([1.60, 1.60, 1.60]),
    }
    ani_tab = TensorMaterial(
        epsilon_tensor=ani_table,               # pass the dict here
        allow_interpolation=True,               # explicit opt-in
        allow_extrapolation=False,
        name='ani_tabulated'
    )
    # When using n_* data, interpolation is performed on n first, then epsilon = n**2

    # Alternatively, full arrays per wavelength
    eps_wl = np.stack([
        np.diag([2.25, 2.21, 2.56]),
        np.diag([2.27, 2.22, 2.56]),
        np.diag([2.30, 2.23, 2.56]),
    ], axis=0)
    ani_table_eps = {
        'wavelength': np.array([1.50, 1.55, 1.60]),
        'epsilon_tensor': eps_wl,
    }
    ani_tab_eps = TensorMaterial(epsilon_tensor=ani_table_eps, allow_interpolation=True)

