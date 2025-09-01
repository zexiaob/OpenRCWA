"""
Sweep engine for parameter scans (Stage 3.1).

Features:
- Parse sweep specs for source params and object params (layers, patterned layers, shapes).
- Cartesian product over parameters; serial or parallel execution (joblib if available).
- Parametric geometry support via with_params() on PatternedLayer/Shape; falls back to setattr.

Usage examples:
    sweep = Sweep({
        'wavelength': [500e-9, 600e-9],
        'theta': [0.0, 10*deg(1)],
        (patterned_layer,): {'width': [0.2, 0.4]},
    }, backend='serial')
    out = sweep.run(base_stack, source, n_harmonics=(3,3))
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from copy import deepcopy

try:  # optional parallel backend marker
    import joblib  # type: ignore
    _HAS_JOBLIB = True
except Exception:  # pragma: no cover
    joblib = None
    _HAS_JOBLIB = False

from rcwa.model.layer import Layer, LayerStack


ParamKey = Union[str, Tuple[object, ...]]


@dataclass
class SweepSpec:
    key: ParamKey
    values: Sequence[Any]


class Sweep:
    """Parameter sweep engine with parametric geometry support.

    params can contain:
    - Source params: keys 'wavelength', 'theta', 'phi', etc. Values are sequences.
    - Object params: key is a tuple containing the target object(s), and value is a dict
      mapping attribute/param name to sequence. For parametric geometry, if the object
      has with_params(), it will be called to produce a new object for each combination.

    Example:
        params = {
            'wavelength': [500e-9, 600e-9],
            (patterned_layer,): {'width': [0.2, 0.4]},
        }
    """

    def __init__(
        self,
        params: Dict[ParamKey, Any],
        backend: str = 'loky',
        n_jobs: int = -1,
        progress: bool = True,
    ) -> None:
        self.params = params or {}
        self.backend = backend
        self.n_jobs = n_jobs
        self.progress = progress

        if self.backend != 'serial' and not _HAS_JOBLIB:
            # Fallback to serial if joblib unavailable
            self.backend = 'serial'

        if self.backend not in ('serial', 'loky', 'thread', 'process'):
            raise ValueError("backend must be one of: serial, loky, thread, process")

        # Normalize into two groups for execution
        self._source_specs: Dict[str, Sequence[Any]] = {}
        self._object_specs: List[Tuple[Tuple[object, ...], Dict[str, Sequence[Any]]]] = []
        self._parse_params()

    def _parse_params(self) -> None:
        for key, val in self.params.items():
            if isinstance(key, tuple) and key and not isinstance(key[0], str):
                # Object param: val must be a dict of attr -> sequence
                if not isinstance(val, dict):
                    raise TypeError("Object parameter spec must be a dict of {attr: sequence}")
                # Ensure sequences
                norm: Dict[str, Sequence[Any]] = {}
                for attr, seq in val.items():
                    if isinstance(seq, (list, tuple)):
                        norm[attr] = list(seq)
                    else:
                        norm[attr] = [seq]
                self._object_specs.append((key, norm))
            else:
                # Source param
                seq = val if isinstance(val, (list, tuple)) else [val]
                self._source_specs[str(key)] = list(seq)

    def _iter_combinations(self) -> Iterable[Tuple[Dict[str, Any], List[Tuple[Tuple[object, ...], Dict[str, Any]]]]]:
        """Yield all Cartesian combinations as (source_updates, object_updates).

        object_updates is a list aligned with self._object_specs, each entry is a dict attr->value.
        """
        # Build product of source params
        src_keys = list(self._source_specs.keys())
        src_values = [self._source_specs[k] for k in src_keys]
        src_product = list(product(*src_values)) if src_values else [()]

        # Build products for each object spec separately, then full product
        obj_choices_per_spec: List[List[Dict[str, Any]]] = []
        for _, spec_dict in self._object_specs:
            keys = list(spec_dict.keys())
            vals = [spec_dict[k] for k in keys]
            combos = []
            for tup in product(*vals):
                combos.append({k: v for k, v in zip(keys, tup)})
            obj_choices_per_spec.append(combos or [{}])

        if obj_choices_per_spec:
            obj_product = list(product(*obj_choices_per_spec))
        else:
            obj_product = [()]

        for src_tup in src_product:
            src_update = {k: v for k, v in zip(src_keys, src_tup)}
            for obj_tup in obj_product:
                obj_updates = list(obj_tup) if obj_tup else []
                yield src_update, obj_updates

    @staticmethod
    def _apply_object_updates(
        stack: LayerStack,
        objects: Tuple[object, ...],
        updates: Dict[str, Any],
    ) -> LayerStack:
        """Return a new LayerStack with updates applied to given objects.

        If an object has with_params(), it will be called; otherwise setattr is used.
        The updated object will be replaced in the stack's internal_layers (and also
        incident/transmission if applicable).
        """
        new_stack = deepcopy(stack)
        for target in objects:
            updated_obj = None
            if hasattr(target, 'with_params') and callable(getattr(target, 'with_params')):
                try:
                    updated_obj = target.with_params(**updates)
                except Exception:
                    updated_obj = None
            if updated_obj is None:
                # fallback: shallow update
                updated_obj = deepcopy(target)
                for attr, val in updates.items():
                    try:
                        setattr(updated_obj, attr, val)
                    except Exception:
                        # ignore silently; this param may not apply
                        pass

            # Replace references in stack
            if new_stack.incident_layer is target:
                new_stack.incident_layer = updated_obj  # type: ignore
            if new_stack.transmission_layer is target:
                new_stack.transmission_layer = updated_obj  # type: ignore
            for i, lyr in enumerate(new_stack.internal_layers):
                if lyr is target:
                    new_stack.internal_layers[i] = updated_obj
        return new_stack

    @staticmethod
    def _build_solver(base_stack: LayerStack, source, n_harmonics: Union[int, Tuple[int, int, int], Tuple[int, int]]):
        # Local import to avoid circular import during package initialization
        from rcwa.core.solver import Solver  # type: ignore
        return Solver(deepcopy(base_stack), deepcopy(source), n_harmonics=n_harmonics)

    def run(
        self,
        base_stack: LayerStack,
        source,
        n_harmonics: Union[int, Tuple[int, int, int], Tuple[int, int]] = 1,
    ) -> Dict[str, Any]:
        """Execute the sweep and return a dictionary with coordinates and results.

        Returns a dict:
            {
              'coords': {name: list(values), ...},
              'results': list[Results],  # solver-packaged Results per point
            }
        """
        combos = list(self._iter_combinations())

        def _eval_point(src_update: Dict[str, Any], obj_updates: List[Dict[str, Any]]):
            # Construct a per-point stack
            stack_pt = deepcopy(base_stack)
            # Apply object updates
            for (objects, _spec), upd in zip(self._object_specs, obj_updates):
                stack_pt = self._apply_object_updates(stack_pt, objects, upd)

            # Build solver and apply source updates
            solver = self._build_solver(stack_pt, source, n_harmonics)
            for k, v in src_update.items():
                setattr(solver.source, k, v)
            # Single-point run
            res = solver.solve()  # no internal sweep
            return res

        if self.backend == 'serial':
            results = [_eval_point(su, ou) for su, ou in combos]
        else:
            # Import lazily to avoid hard dependency warnings
            from joblib import Parallel, delayed  # type: ignore
            backend = {
                'loky': 'loky',
                'thread': 'threading',
                'process': 'multiprocessing',
            }[self.backend]
            results = Parallel(n_jobs=self.n_jobs, backend=backend)(
                delayed(_eval_point)(su, ou) for su, ou in combos
            )

        # Build coordinate dict for reporting
        coord_dict: Dict[str, List[Any]] = {}
        # Source coords
        coord_dict.update(self._source_specs)
        # Object coords: flatten into composite names
        for (objects, spec) in self._object_specs:
            obj_name = '+'.join([obj.__class__.__name__ for obj in objects])
            for attr, seq in spec.items():
                coord_dict[f"{obj_name}.{attr}"] = list(seq)

        return {
            'coords': coord_dict,
            'results': results,
        }
