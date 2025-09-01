"""
Shape definitions for declarative geometry.

This module provides Shape base class and concrete implementations for basic
geometric shapes, as well as complex shapes created through boolean operations.
"""

import numpy as np
import warnings
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional, Any, Dict, Callable
from dataclasses import dataclass
import hashlib
import json


@dataclass 
class Point:
    """2D point representation."""
    x: float
    y: float
    
    def __iter__(self):
        """Allow tuple unpacking."""
        yield self.x
        yield self.y
        
    def __add__(self, other):
        """Add two points."""
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """Subtract two points."""
        return Point(self.x - other.x, self.y - other.y)
        
    def __mul__(self, scalar):
        """Scale point by scalar."""
        return Point(self.x * scalar, self.y * scalar)
        
    def distance_to(self, other):
        """Calculate distance to another point."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


class Shape(ABC):
    """
    Abstract base class for all geometric shapes.
    
    Each shape must be able to:
    1. Test if a point is inside the shape
    2. Generate a tensor function for material distribution
    3. Provide bounds for optimization
    4. Support parameterization for sweeps
    """
    
    def __init__(self, material=None, **params):
        """
        Initialize shape with optional material and parameters.
        
        :param material: Material for this shape region, or None for background
        :param params: Named parameters for parameterization support
        """
        self.material = material
        self.params = params.copy() if params else {}
        self._param_dependencies = set(self.params.keys()) if self.params else set()
        # Subclasses may declare parametric fields; default none
        self._parametric_fields = tuple()
        
    @abstractmethod
    def contains(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """
        Test if point(s) are inside the shape.
        
        :param x: x-coordinate(s) 
        :param y: y-coordinate(s)
        :return: Boolean or boolean array indicating containment
        """
        pass
    
    def to_tensor_function(self) -> Callable[[float, float], np.ndarray]:
        """
        Convert shape to a tensor function epsilon_tensor_fn(x, y) → (3, 3).
        
        This function returns the material's epsilon tensor at each point,
        supporting full anisotropic material distributions.
        
        :return: Function (x, y) → epsilon_tensor as (3, 3) complex array
        """
        if self.material is None:
            # Return identity for vacuum/air background
            def tensor_fn(x, y):
                shape = np.broadcast_shapes(np.asarray(x).shape, np.asarray(y).shape)
                return np.broadcast_to(np.eye(3, dtype=complex), shape + (3, 3))
            return tensor_fn
        else:
            def tensor_fn(x, y):
                # Get shape containment
                inside = self.contains(x, y)
                
                # Get material tensor - handle different material types
                if hasattr(self.material, 'epsilon_tensor'):
                    # TensorMaterial
                    if callable(self.material.epsilon_tensor):
                        # Wavelength-dependent - use default wavelength for now
                        # TODO: Pass wavelength context through the tensor function
                        base_tensor = self.material.epsilon_tensor(550e-9)  # 550nm default
                    else:
                        base_tensor = self.material.epsilon_tensor
                elif hasattr(self.material, 'er'):
                    # Regular Material - convert to tensor
                    if callable(self.material.er):
                        er = self.material.er(550e-9)
                    else:
                        er = self.material.er
                    base_tensor = np.eye(3, dtype=complex) * er
                else:
                    # Fallback to identity
                    base_tensor = np.eye(3, dtype=complex)
                
                # Broadcast tensor to match coordinate arrays
                x_arr, y_arr = np.asarray(x), np.asarray(y)
                shape = np.broadcast_shapes(x_arr.shape, y_arr.shape)
                
                # Create output array
                result = np.broadcast_to(np.eye(3, dtype=complex), shape + (3, 3)).copy()
                
                # Apply material where shape contains points
                if np.any(inside):
                    result[inside] = base_tensor
                    
                return result
                
            return tensor_fn
    
    def cross_section(self, z_fraction: float) -> 'Shape':
        """
        Return a cross-sectioned shape for a given relative z position (0..1).

        Base shapes are z-uniform; subclasses can override to provide
        z-dependent geometry (e.g., tapered walls). The default implementation
        returns self.

        :param z_fraction: Relative position along thickness in [0,1]
        :return: Shape at this z position
        """
        return self

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get bounding box of the shape.
        
        :return: (x_min, x_max, y_min, y_max)
        """
        # Default implementation - subclasses should override for efficiency
        # Use a coarse sampling approach
        x_test = np.linspace(-10, 10, 201)
        y_test = np.linspace(-10, 10, 201)
        X, Y = np.meshgrid(x_test, y_test)
        inside = self.contains(X, Y)
        
        if not np.any(inside):
            return (0, 0, 0, 0)
            
        x_inside = X[inside]
        y_inside = Y[inside]
        
        return (float(x_inside.min()), float(x_inside.max()), 
                float(y_inside.min()), float(y_inside.max()))
    
    def with_params(self, **kwargs) -> 'Shape':
        """
        Create a new shape with updated parameters.
        
        This enables parametric geometry for sweep applications.
        For shapes with parametric properties (functions), this method
        evaluates those functions with the new parameters to create
        a concrete shape instance.
        
        :param kwargs: Parameter updates
        :return: New shape instance with updated parameters
        """
        new_params = self.params.copy()
        new_params.update(kwargs)
        
        # Create new instance, evaluating any parametric properties
        constructor_args = {}

        # Copy all attributes from current instance, evaluating parametric fields
        allowed = set(self._parametric_fields)
        for key, value in self.__dict__.items():
            if key == 'params':
                continue
            if callable(value) and (not allowed or key in allowed):
                try:
                    evaluated_value = value(new_params)
                except Exception:
                    # If evaluation fails, keep the callable for later resolution
                    constructor_args[key] = value
                else:
                    # Perform validation after successful evaluation so errors aren't swallowed
                    if key in ['width', 'height', 'radius'] and evaluated_value <= 0:
                        raise ValueError(f"{key.capitalize()} must be positive")
                    constructor_args[key] = evaluated_value
            else:
                constructor_args[key] = value
        
        # Add new parameters
        constructor_args['params'] = new_params
        
        # Create new instance of same type
        new_shape = self.__class__(**constructor_args)
        return new_shape
    
    def get_hash(self) -> str:
        """
        Generate hash for caching purposes.
        
        :return: Hash string uniquely identifying this shape configuration
        """
        # Create deterministic hash from class name, parameters, and material
        hash_dict = {
            'class': self.__class__.__name__,
            'params': sorted(self.params.items()) if self.params else [],
            'material_hash': getattr(self.material, '__hash__', lambda: None)()
        }
        
        hash_str = json.dumps(hash_dict, sort_keys=True, default=str)
        return hashlib.md5(hash_str.encode()).hexdigest()


class Rectangle(Shape):
    """Rectangular shape."""
    
    def __init__(self, center: Tuple[float, float] = (0, 0), 
                 width: float = 1.0, height: float = 1.0, 
                 rotation: float = 0.0, material=None, **params):
        """
        Initialize rectangle.
        
        :param center: Center position (x, y)
        :param width: Width in x direction
        :param height: Height in y direction  
        :param rotation: Rotation angle in radians
        :param material: Material for this region
        :param params: Additional parameters for parameterization
        """
        super().__init__(material, **params)
        self.center = Point(*center)
        self.width = width
        self.height = height 
        self.rotation = rotation
        self._parametric_fields = ('width', 'height', 'rotation')

        # Update params with current values for consistency
        self.params.update({
            'center': center,
            'width': width,
            'height': height,
            'rotation': rotation
        })
    
    def contains(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Test if points are inside the rectangle."""
        x_arr, y_arr = np.asarray(x), np.asarray(y)
        
        # Translate to rectangle center
        dx = x_arr - self.center.x
        dy = y_arr - self.center.y
        
        # Rotate coordinates if needed
        if self.rotation != 0:
            cos_r, sin_r = np.cos(-self.rotation), np.sin(-self.rotation)
            dx_rot = dx * cos_r - dy * sin_r
            dy_rot = dx * sin_r + dy * cos_r
            dx, dy = dx_rot, dy_rot
        
        # Check bounds
        return (np.abs(dx) <= self.width/2) & (np.abs(dy) <= self.height/2)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of rectangle."""
        # For rotated rectangles, calculate corner positions
        corners_x = [-self.width/2, self.width/2, self.width/2, -self.width/2]
        corners_y = [-self.height/2, -self.height/2, self.height/2, self.height/2]
        
        if self.rotation != 0:
            cos_r, sin_r = np.cos(self.rotation), np.sin(self.rotation)
            corners_x_rot = [x * cos_r - y * sin_r for x, y in zip(corners_x, corners_y)]
            corners_y_rot = [x * sin_r + y * cos_r for x, y in zip(corners_x, corners_y)]
            corners_x, corners_y = corners_x_rot, corners_y_rot
        
        # Translate to center
        corners_x = [x + self.center.x for x in corners_x]
        corners_y = [y + self.center.y for y in corners_y]
        
        return (min(corners_x), max(corners_x), min(corners_y), max(corners_y))


class Circle(Shape):
    """Circular shape."""
    
    def __init__(self, center: Tuple[float, float] = (0, 0), 
                 radius: float = 1.0, material=None, **params):
        """
        Initialize circle.
        
        :param center: Center position (x, y)
        :param radius: Circle radius
        :param material: Material for this region
        :param params: Additional parameters for parameterization
        """
        super().__init__(material, **params)
        self.center = Point(*center)
        self.radius = radius
        self._parametric_fields = ('radius',)

        self.params.update({
            'center': center,
            'radius': radius
        })
    
    def contains(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Test if points are inside the circle."""
        x_arr, y_arr = np.asarray(x), np.asarray(y)
        
        dx = x_arr - self.center.x
        dy = y_arr - self.center.y
        
        return (dx**2 + dy**2) <= self.radius**2
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of circle."""
        return (self.center.x - self.radius, self.center.x + self.radius,
                self.center.y - self.radius, self.center.y + self.radius)


class Ellipse(Shape):
    """Elliptical shape."""
    
    def __init__(self, center: Tuple[float, float] = (0, 0),
                 a: float = 1.0, b: float = 0.5,
                 rotation: float = 0.0, material=None, **params):
        """
        Initialize ellipse.
        
        :param center: Center position (x, y)
        :param a: Semi-major axis length
        :param b: Semi-minor axis length
        :param rotation: Rotation angle in radians
        :param material: Material for this region
        :param params: Additional parameters for parameterization
        """
        super().__init__(material, **params)
        self.center = Point(*center)
        self.a = a  # Semi-major axis
        self.b = b  # Semi-minor axis
        self.rotation = rotation
        self._parametric_fields = ('a', 'b', 'rotation')

        self.params.update({
            'center': center,
            'a': a,
            'b': b,
            'rotation': rotation
        })
    
    def contains(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Test if points are inside the ellipse."""
        x_arr, y_arr = np.asarray(x), np.asarray(y)
        
        # Translate to ellipse center
        dx = x_arr - self.center.x
        dy = y_arr - self.center.y
        
        # Rotate coordinates if needed
        if self.rotation != 0:
            cos_r, sin_r = np.cos(-self.rotation), np.sin(-self.rotation)
            dx_rot = dx * cos_r - dy * sin_r
            dy_rot = dx * sin_r + dy * cos_r
            dx, dy = dx_rot, dy_rot
        
        # Ellipse equation: (x/a)^2 + (y/b)^2 <= 1
        return (dx/self.a)**2 + (dy/self.b)**2 <= 1.0
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of ellipse."""
        # For rotated ellipse, the bounding box is more complex
        if self.rotation == 0:
            return (self.center.x - self.a, self.center.x + self.a,
                    self.center.y - self.b, self.center.y + self.b)
        else:
            # Use parametric form to find extremes
            cos_r, sin_r = np.cos(self.rotation), np.sin(self.rotation)
            
            # Extreme x: d/dt(a*cos(t)*cos(r) - b*sin(t)*sin(r)) = 0
            # Extreme y: d/dt(a*cos(t)*sin(r) + b*sin(t)*cos(r)) = 0
            dx_ext = np.sqrt(self.a**2 * cos_r**2 + self.b**2 * sin_r**2)
            dy_ext = np.sqrt(self.a**2 * sin_r**2 + self.b**2 * cos_r**2)
            
            return (self.center.x - dx_ext, self.center.x + dx_ext,
                    self.center.y - dy_ext, self.center.y + dy_ext)


class Polygon(Shape):
    """Polygonal shape with arbitrary vertices."""
    
    def __init__(self, vertices: Union[List[Tuple[float, float]], Callable[[Dict[str, Any]], List[Tuple[float, float]]]], 
                 holes: Optional[Union[List[List[Tuple[float, float]]], Callable[[Dict[str, Any]], List[List[Tuple[float, float]]]]]] = None,
                 material=None, **params):
        """
        Initialize polygon.
        
        :param vertices: List of vertex coordinates [(x1, y1), (x2, y2), ...]
        :param holes: Optional list of hole polygons, each as vertex list
        :param material: Material for this region
        :param params: Additional parameters for parameterization
        """
        super().__init__(material, **params)
        self._vertices_template = vertices if callable(vertices) else None
        self._holes_template = holes if callable(holes) else None
        resolved_vertices = vertices(params) if callable(vertices) else vertices
        self.vertices = [Point(*v) for v in (resolved_vertices or [])]
        self.holes = []
        if holes and not callable(holes):
            self.holes = [[Point(*v) for v in hole] for hole in holes]
        elif callable(holes):
            resolved_holes = holes(params)
            self.holes = [[Point(*v) for v in hole] for hole in (resolved_holes or [])]
        self._parametric_fields = ('_vertices_template', '_holes_template')

        # Validate polygon
        self._validate_polygon()
        # Enforce orientations: outer CCW, holes CW
        self._ensure_orientations()
        # Validate holes containment
        self._validate_holes_inside()

        self.params.update({
            'vertices': resolved_vertices if not callable(vertices) else 'template',
            'holes': (holes or []) if not callable(holes) else 'template'
        })

    @classmethod
    def from_template(cls, template_func: Callable[[Dict[str, Any]], Tuple[List[Tuple[float, float]], Optional[List[List[Tuple[float, float]]]]]],
                      material=None, **params) -> 'Polygon':
        """
        Build a polygon from a template function returning (vertices, holes).
        The template will be retained for future with_params evaluations.
        """
        verts, holes = template_func(params)
        poly = cls(verts, holes=holes, material=material, **params)
        poly._vertices_template = lambda p: template_func(p)[0]
        poly._holes_template = lambda p: template_func(p)[1]
        return poly
    
    def _validate_polygon(self):
        """Validate polygon geometry."""
        if len(self.vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        
        # Check for self-intersections (basic check)
        self._check_vertex_order()
        self._check_self_intersection()
        
        # Validate holes
        for hole in self.holes:
            if len(hole) < 3:
                raise ValueError("Hole must have at least 3 vertices")

    def _ensure_orientations(self):
        """Ensure exterior is CCW and holes are CW for robustness."""
        def signed_area(pts: List[Point]) -> float:
            area = 0.0
            n = len(pts)
            for i in range(n):
                j = (i + 1) % n
                area += pts[i].x * pts[j].y - pts[j].x * pts[i].y
            return area / 2.0
        # Exterior CCW
        if signed_area(self.vertices) < 0:
            self.vertices.reverse()
        # Holes CW
        for hole in self.holes:
            if signed_area(hole) > 0:
                hole.reverse()

    def _validate_holes_inside(self):
        """Check that holes lie within the outer polygon and do not self-intersect with boundary."""
        if not self.holes:
            return
        # Simple check: all hole vertices must be inside outer polygon
        for hole in self.holes:
            xs = np.array([p.x for p in hole])
            ys = np.array([p.y for p in hole])
            inside = self._point_in_polygon(xs, ys, self.vertices)
            if not np.all(inside):
                warnings.warn("Hole vertices found outside outer polygon; geometry may be invalid.", UserWarning)
    
    def _check_vertex_order(self):
        """Check if vertices are in consistent order and warn if needed."""
        # Calculate signed area to determine orientation
        area = 0
        n = len(self.vertices)
        for i in range(n):
            j = (i + 1) % n
            area += (self.vertices[j].x - self.vertices[i].x) * (self.vertices[j].y + self.vertices[i].y)
        
        if area > 0:
            warnings.warn("Polygon vertices appear to be in clockwise order. "
                         "Counter-clockwise is preferred for exterior boundaries.",
                         UserWarning)

    def _segments(self, vertices: List[Point]):
        n = len(vertices)
        for i in range(n):
            a = vertices[i]
            b = vertices[(i + 1) % n]
            yield a, b

    @staticmethod
    def _ccw(a: Point, b: Point, c: Point) -> float:
        return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)

    @staticmethod
    def _on_segment(a: Point, b: Point, c: Point) -> bool:
        return (min(a.x, b.x) - 1e-15 <= c.x <= max(a.x, b.x) + 1e-15 and
                min(a.y, b.y) - 1e-15 <= c.y <= max(a.y, b.y) + 1e-15 and
                abs(Polygon._ccw(a, b, c)) < 1e-12)

    def _segments_intersect(self, p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
        # Proper intersection test with colinearity handling
        d1 = Polygon._ccw(p1, p2, q1)
        d2 = Polygon._ccw(p1, p2, q2)
        d3 = Polygon._ccw(q1, q2, p1)
        d4 = Polygon._ccw(q1, q2, p2)

        if (d1 * d2 < 0) and (d3 * d4 < 0):
            return True
        # Colinear cases
        if abs(d1) < 1e-12 and Polygon._on_segment(p1, p2, q1):
            return True
        if abs(d2) < 1e-12 and Polygon._on_segment(p1, p2, q2):
            return True
        if abs(d3) < 1e-12 and Polygon._on_segment(q1, q2, p1):
            return True
        if abs(d4) < 1e-12 and Polygon._on_segment(q1, q2, p2):
            return True
        return False

    def _check_self_intersection(self):
        """Detect self-intersections; raise by default, allow opt-in warning."""
        n = len(self.vertices)
        allow = bool(self.params.get('allow_self_intersection', False))
        for i, (a1, a2) in enumerate(self._segments(self.vertices)):
            for j, (b1, b2) in enumerate(self._segments(self.vertices)):
                # Skip same or adjacent edges
                if abs(i - j) <= 1 or (i == 0 and j == n - 1) or (j == 0 and i == n - 1):
                    continue
                if self._segments_intersect(a1, a2, b1, b2):
                    msg = "Polygon has self-intersections between edges"
                    if allow:
                        warnings.warn(msg, UserWarning)
                        return
                    raise ValueError(msg)
    
    def contains(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Test if points are inside the polygon using ray casting."""
        x_arr, y_arr = np.asarray(x), np.asarray(y)
        flat_shape = x_arr.shape
        x_flat, y_flat = x_arr.flatten(), y_arr.flatten()
        
        # Ray casting algorithm for main polygon
        inside_main = self._point_in_polygon(x_flat, y_flat, self.vertices)
        
        # Subtract holes
        for hole in self.holes:
            inside_hole = self._point_in_polygon(x_flat, y_flat, hole)
            inside_main = inside_main & ~inside_hole
        
        return inside_main.reshape(flat_shape)
    
    def _point_in_polygon(self, x: np.ndarray, y: np.ndarray, 
                         vertices: List[Point]) -> np.ndarray:
        """Ray casting algorithm for point-in-polygon test."""
        n = len(vertices)
        inside = np.zeros(len(x), dtype=bool)
        
        j = n - 1
        for i in range(n):
            xi, yi = vertices[i].x, vertices[i].y
            xj, yj = vertices[j].x, vertices[j].y
            
            # Check if ray crosses edge
            cond1 = (yi > y) != (yj > y)
            # Avoid division by zero on horizontal edges
            denom = (yj - yi)
            cond2 = np.zeros_like(x, dtype=bool)
            non_horizontal = np.abs(denom) > 1e-18
            if np.any(non_horizontal):
                x_intersect = (xj - xi) * (y[non_horizontal] - yi) / denom + xi
                cond2[non_horizontal] = x[non_horizontal] < x_intersect
            
            inside ^= cond1 & cond2
            j = i
            
        return inside

    def with_params(self, **kwargs) -> 'Polygon':
        """Evaluate template-based vertices/holes against new params."""
        new_params = self.params.copy()
        new_params.update(kwargs)
        # Resolve templates if present
        if self._vertices_template is not None:
            verts = self._vertices_template(new_params)
        else:
            verts = [(v.x, v.y) for v in self.vertices]
        if self._holes_template is not None:
            holes = self._holes_template(new_params)
        else:
            holes = [[(p.x, p.y) for p in hole] for hole in self.holes] if self.holes else None
        return Polygon(verts, holes=holes, material=self.material, **new_params)
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of polygon."""
        if not self.vertices:
            return (0, 0, 0, 0)
            
        x_coords = [v.x for v in self.vertices]
        y_coords = [v.y for v in self.vertices]
        
        return (min(x_coords), max(x_coords), min(y_coords), max(y_coords))
    
    def convex_hull(self) -> 'Polygon':
        """
        Compute convex hull of the polygon.
        
        :return: New Polygon representing the convex hull
        """
        # Graham scan algorithm
        def cross_product(o, a, b):
            return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)
        
        # Find bottom-most point (and leftmost if tie)
        vertices = sorted(self.vertices, key=lambda p: (p.y, p.x))
        
        if len(vertices) <= 3:
            return Polygon([(v.x, v.y) for v in vertices], material=self.material)
        
        # Sort points by polar angle
        def polar_angle(p):
            return np.arctan2(p.y - vertices[0].y, p.x - vertices[0].x)
        
        sorted_vertices = [vertices[0]] + sorted(vertices[1:], key=polar_angle)
        
        # Build convex hull
        hull = []
        for p in sorted_vertices:
            while len(hull) > 1 and cross_product(hull[-2], hull[-1], p) <= 0:
                hull.pop()
            hull.append(p)
        
        return Polygon([(v.x, v.y) for v in hull], material=self.material)


class RegularPolygon(Polygon):
    """Regular polygon (n-sided with equal sides and angles)."""

    def __init__(self, center: Tuple[float, float] = (0, 0),
                 radius: float = 1.0, n_sides: int = None,
                 rotation: float = 0.0, material=None, **params):
        """
        Initialize regular polygon.

        :param center: Center position (x, y)
        :param radius: Circumradius (distance from center to vertex)
        :param n_sides: Number of sides (alias: num_sides)
        :param rotation: Rotation angle in radians
        :param material: Material for this region
        :param params: Additional parameters for parameterization
        """
        # Support alias 'num_sides' used by tests/docs
        if n_sides is None and 'num_sides' in params:
            n_sides = params.pop('num_sides')

        if n_sides is None:
            raise ValueError("Regular polygon requires n_sides or num_sides >= 3")
        if n_sides < 3:
            raise ValueError("Regular polygon must have at least 3 sides")

        # Generate vertices
        angles = np.linspace(0, 2*np.pi, n_sides + 1)[:-1] + rotation
        vertices = [(center[0] + radius * np.cos(a),
                     center[1] + radius * np.sin(a)) for a in angles]

        # Store original parameters
        self.center = Point(*center)
        self.radius = radius
        self.n_sides = n_sides
        self.rotation = rotation

        super().__init__(vertices, material=material, **params)

        # Update params with regular polygon specific values
        self.params.update({
            'center': center,
            'radius': radius,
            'n_sides': n_sides,
            'rotation': rotation
        })

    def with_params(self, **kwargs) -> 'RegularPolygon':
        """Return a new RegularPolygon with updated parameters.

        Supports updating center, radius, n_sides/num_sides, and rotation.
        """
        new_params = self.params.copy()
        new_params.update(kwargs)

        center = kwargs.get('center', (self.center.x, self.center.y))
        radius = kwargs.get('radius', self.radius)
        n_sides = kwargs.get('n_sides', new_params.get('num_sides', self.n_sides))
        rotation = kwargs.get('rotation', self.rotation)

        if n_sides is None or n_sides < 3:
            raise ValueError("Regular polygon must have at least 3 sides")

        angles = np.linspace(0, 2*np.pi, int(n_sides) + 1)[:-1] + rotation
        vertices = [(center[0] + radius * np.cos(a),
                     center[1] + radius * np.sin(a)) for a in angles]

        return RegularPolygon(center=center, radius=radius, n_sides=int(n_sides),
                              rotation=rotation, material=self.material, **new_params)


class TaperedPolygon(Polygon):
    """Polygon whose cross-section linearly tapers along z (0..thickness)."""

    def __init__(self,
                 bottom_vertices: List[Tuple[float, float]],
                 top_vertices: List[Tuple[float, float]],
                 material=None, **params):
        if len(bottom_vertices) != len(top_vertices):
            raise ValueError("Top and bottom vertex lists must have the same length")
        self._bottom_vertices = [Point(*v) for v in bottom_vertices]
        self._top_vertices = [Point(*v) for v in top_vertices]
        super().__init__(bottom_vertices, material=material, **params)

    def cross_section(self, z_fraction: float) -> 'TaperedPolygon':
        zf = min(max(z_fraction, 0.0), 1.0)
        interp = []
        for vb, vt in zip(self._bottom_vertices, self._top_vertices):
            x = vb.x + (vt.x - vb.x) * zf
            y = vb.y + (vt.y - vb.y) * zf
            interp.append((x, y))
        return TaperedPolygon(interp, interp, material=self.material, **self.params)


# Boolean operation shapes

class ComplexShape(Shape):
    """Base class for shapes created through boolean operations."""
    
    def __init__(self, shapes: List[Shape], operation: str, material=None, **params):
        """
        Initialize complex shape.
        
        :param shapes: List of shapes to combine
        :param operation: Boolean operation ('union', 'intersection', 'difference')  
        :param material: Material for result region (overrides component materials)
        :param params: Additional parameters
        """
        super().__init__(material, **params)
        self.shapes = shapes
        self.operation = operation
        
        if len(shapes) < 1:
            raise ValueError("ComplexShape requires at least 1 shape")
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of complex shape."""
        if not self.shapes:
            return (0, 0, 0, 0)
        
        bounds_list = [shape.get_bounds() for shape in self.shapes]
        
        if self.operation == 'union':
            # Union: envelope of all bounds
            x_min = min(b[0] for b in bounds_list)
            x_max = max(b[1] for b in bounds_list)  
            y_min = min(b[2] for b in bounds_list)
            y_max = max(b[3] for b in bounds_list)
            return (x_min, x_max, y_min, y_max)
        else:
            # Intersection/Difference: use first shape bounds as approximation
            return bounds_list[0]
    
    def get_hash(self) -> str:
        """Generate hash including all component shapes."""
        component_hashes = [shape.get_hash() for shape in self.shapes]
        hash_dict = {
            'class': self.__class__.__name__,
            'operation': self.operation,
            'components': component_hashes,
            'material_hash': getattr(self.material, '__hash__', lambda: None)()
        }
        
        hash_str = json.dumps(hash_dict, sort_keys=True, default=str)
        return hashlib.md5(hash_str.encode()).hexdigest()

    def with_params(self, **kwargs) -> 'ComplexShape':
        """Propagate parameter updates to all child shapes."""
        updated = []
        for s in self.shapes:
            if hasattr(s, 'with_params'):
                updated.append(s.with_params(**kwargs))
            else:
                updated.append(s)
        return self.__class__(updated, material=self.material, **{**self.params, **kwargs})


class UnionShape(ComplexShape):
    """Union of multiple shapes (logical OR)."""
    
    def __init__(self, shapes: List[Shape], material=None, **params):
        """
        Initialize union shape.
        
        :param shapes: List of shapes to union
        :param material: Material for union region
        :param params: Additional parameters
        """
        super().__init__(shapes, 'union', material, **params)
    
    def contains(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Test if points are in any of the shapes."""
        if not self.shapes:
            return np.zeros_like(np.asarray(x), dtype=bool)
        
        result = self.shapes[0].contains(x, y)
        for shape in self.shapes[1:]:
            result |= shape.contains(x, y)
            
        return result


class IntersectionShape(ComplexShape):
    """Intersection of multiple shapes (logical AND)."""
    
    def __init__(self, shapes: List[Shape], material=None, **params):
        """
        Initialize intersection shape.
        
        :param shapes: List of shapes to intersect
        :param material: Material for intersection region
        :param params: Additional parameters
        """
        super().__init__(shapes, 'intersection', material, **params)
    
    def contains(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Test if points are in all of the shapes."""
        if not self.shapes:
            return np.zeros_like(np.asarray(x), dtype=bool)
        
        result = self.shapes[0].contains(x, y)
        for shape in self.shapes[1:]:
            result &= shape.contains(x, y)
            
        return result


class DifferenceShape(ComplexShape):
    """Difference of shapes (first shape minus others)."""
    
    def __init__(self, outer_shape: Shape, inner_shapes: List[Shape], material=None, **params):
        """
        Initialize difference shape.
        
        :param outer_shape: Main shape to subtract from
        :param inner_shapes: List of shapes to subtract (holes)
        :param material: Material for remaining region
        :param params: Additional parameters
        """
        super().__init__([outer_shape] + inner_shapes, 'difference', material, **params)
        self.outer_shape = outer_shape
        self.inner_shapes = inner_shapes
    
    def contains(self, x: Union[float, np.ndarray], y: Union[float, np.ndarray]) -> Union[bool, np.ndarray]:
        """Test if points are in outer shape but not in any inner shape."""
        # Must be in outer shape
        result = self.outer_shape.contains(x, y)
        
        # But not in any inner shape
        for inner_shape in self.inner_shapes:
            result &= ~inner_shape.contains(x, y)
            
        return result

    def with_params(self, **kwargs) -> 'DifferenceShape':
        """Propagate parameters to outer and inner shapes properly."""
        new_outer = self.outer_shape.with_params(**kwargs) if hasattr(self.outer_shape, 'with_params') else self.outer_shape
        new_inners = [s.with_params(**kwargs) if hasattr(s, 'with_params') else s for s in self.inner_shapes]
        new_params = {**self.params, **kwargs}
        return DifferenceShape(new_outer, new_inners, material=self.material, **new_params)
