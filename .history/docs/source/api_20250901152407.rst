.. Rigorous Coupled Wave Analysis documentation master file, created by
   sphinx-quickstart on Mon Sep 28 12:56:28 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

rcwa API
==========================================================

The main user-facing classes: :code:`Solver`, :code:`Layer`, :code:`LayerStack` (:code:`Stack`), :code:`Source`, :code:`Material` (:code:`TensorMaterial`), :code:`Crystal`, and :code:`Plotter`. :code:`PatternedLayer` provides a declarative geometry API for periodic layers with boolean shapes and parametric templates. Once all individual layers are constructed, they are used to create a :code:`LayerStack`, which contains information about which region is considered the "incident" region and which is the "transmission" region (both are semi-infinite).

The :code:`Source` class is used to define the excitation source - the wavelength, polarization, and incident angle. 

Once the user has created a :code:`Source` and :code:`LayerStack` class, these are passed into the :code:`Solver` class, which then runs the simulation, and makes the results available as a dictionary :code:`Solver.results`.

.. autoclass:: rcwa.Solver
    :members: Solve

.. autoclass:: rcwa.Layer

.. autoclass:: rcwa.LayerStack

.. autodata:: rcwa.Stack

.. autoclass:: rcwa.Source

.. autoclass:: rcwa.Crystal

.. autoclass:: rcwa.Material

.. autoclass:: rcwa.model.material.TensorMaterial
    :members:

Geometry API
-----------------

.. autoclass:: rcwa.geom.patterned.PatternedLayer
    :members:

.. autofunction:: rcwa.geom.patterned.rectangular_lattice

.. autofunction:: rcwa.geom.patterned.square_lattice

.. autofunction:: rcwa.geom.patterned.hexagonal_lattice

.. autoclass:: rcwa.geom.shape.Rectangle

.. autoclass:: rcwa.geom.shape.Circle

.. autoclass:: rcwa.geom.shape.Ellipse

.. autoclass:: rcwa.geom.shape.Polygon
    :members:

.. autoclass:: rcwa.geom.shape.RegularPolygon

.. autoclass:: rcwa.geom.shape.UnionShape

.. autoclass:: rcwa.geom.shape.IntersectionShape

.. autoclass:: rcwa.geom.shape.DifferenceShape

.. autoclass:: rcwa.utils.Plotter
    :members:
