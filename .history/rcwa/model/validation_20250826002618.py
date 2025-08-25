"""
Enhanced material classes with Pydantic validation.

This module provides strongly validated material classes that follow ROADMAP
requirements for SI unit consistency and robust input validation.
"""
import numpy as np
from pydantic import BaseModel, Field, validator, root_validator
from typing import Union, Callable, Optional, Dict, Any
from numpy.typing import ArrayLike


class MaterialValidationBase(BaseModel):
    """Base class for all material validation models."""
    
    class Config:
        # Allow numpy arrays and other non-serializable types
        arbitrary_types_allowed = True
        # Validate field assignment
        validate_assignment = True


class TensorMaterialValidated(MaterialValidationBase):
    """
    Pydantic-validated anisotropic material class supporting 3x3 tensors.
    
    Follows ROADMAP requirements:
    - SI units (meters for all lengths)
    - Strong Pydantic validation
    - Support for constant/function/table input
    """
    
    # Core properties
    name: str = Field(default="anisotropic", description="Material identifier")
    epsilon_tensor: Union[np.ndarray, Callable, Dict[str, Any]] = Field(
        description="3x3 permittivity tensor (constant/function/table)"
    )
    mu_tensor: Optional[Union[np.ndarray, Callable, Dict[str, Any]]] = Field(
        default=None, description="3x3 permeability tensor (optional)"
    )
    
    # Physical constraints
    wavelength_range: Optional[tuple] = Field(
        default=None, 
        description="Valid wavelength range in meters [min, max]"
    )
    thickness_range: Optional[tuple] = Field(
        default=(1e-12, 1e-3),  # 1pm to 1mm in SI units
        description="Valid thickness range in meters [min, max]"
    )
    
    # Computational properties
    dispersive: bool = Field(default=False, description="Whether material is dispersive")
    source: Optional[Any] = Field(default=None, description="Associated source object")
    
    @validator('epsilon_tensor')
    def validate_epsilon_tensor(cls, v):
        """Validate epsilon tensor shape and properties."""
        if isinstance(v, np.ndarray):
            if v.shape != (3, 3):
                raise ValueError(f"Epsilon tensor must be 3x3, got shape {v.shape}")
            if not np.iscomplexobj(v) and not np.isrealobj(v):
                raise ValueError("Epsilon tensor must be numeric (real or complex)")
        elif callable(v):
            # For functions, we'll validate at runtime
            pass
        elif isinstance(v, dict):
            # For tables, validate required keys
            required_keys = ['wavelength', 'epsilon']
            if not all(key in v for key in required_keys):
                raise ValueError(f"Table must contain keys: {required_keys}")
        else:
            raise ValueError("Epsilon tensor must be array, function, or table dict")
        return v
    
    @validator('mu_tensor')
    def validate_mu_tensor(cls, v):
        """Validate mu tensor shape and properties."""
        if v is None:
            return np.eye(3, dtype=complex)  # Default to identity
        
        if isinstance(v, np.ndarray):
            if v.shape != (3, 3):
                raise ValueError(f"Mu tensor must be 3x3, got shape {v.shape}")
        elif callable(v):
            pass  # Validate at runtime
        elif isinstance(v, dict):
            required_keys = ['wavelength', 'mu']
            if not all(key in v for key in required_keys):
                raise ValueError(f"Mu table must contain keys: {required_keys}")
        else:
            raise ValueError("Mu tensor must be array, function, or table dict")
        return v
    
    @validator('wavelength_range')
    def validate_wavelength_range(cls, v):
        """Validate wavelength range is in SI units (meters)."""
        if v is None:
            return None
        
        if len(v) != 2:
            raise ValueError("Wavelength range must be [min, max]")
        
        min_wl, max_wl = v
        if min_wl <= 0 or max_wl <= 0:
            raise ValueError("Wavelengths must be positive")
        if min_wl >= max_wl:
            raise ValueError("Min wavelength must be < max wavelength")
        
        # Check if values seem reasonable for SI units (meters)
        if min_wl > 1e-3:  # > 1mm seems too large for wavelength
            raise ValueError("Wavelength seems too large - ensure SI units (meters)")
        if max_wl < 1e-9:  # < 1nm seems too small
            raise ValueError("Wavelength seems too small - ensure SI units (meters)")
        
        return v
    
    @validator('thickness_range')
    def validate_thickness_range(cls, v):
        """Validate thickness range is in SI units (meters)."""
        if v is None:
            return None
        
        min_t, max_t = v
        if min_t <= 0 or max_t <= 0:
            raise ValueError("Thicknesses must be positive")
        if min_t >= max_t:
            raise ValueError("Min thickness must be < max thickness")
        
        # Check if values seem reasonable for SI units (meters)
        if min_t > 1e-2:  # > 1cm seems too large for typical layers
            raise ValueError("Thickness seems too large - ensure SI units (meters)")
        if max_t < 1e-12:  # < 1pm seems too small
            raise ValueError("Thickness seems too small - ensure SI units (meters)")
            
        return v
    
    @root_validator
    def validate_consistency(cls, values):
        """Validate overall consistency of material definition."""
        epsilon_tensor = values.get('epsilon_tensor')
        mu_tensor = values.get('mu_tensor')
        
        # If either tensor is callable/dict, material is dispersive
        if (callable(epsilon_tensor) or isinstance(epsilon_tensor, dict) or
            callable(mu_tensor) or isinstance(mu_tensor, dict)):
            values['dispersive'] = True
            
            # Dispersive materials should have a source
            if values.get('source') is None:
                import warnings
                warnings.warn("Dispersive materials should have an associated source")
        
        return values


def nm(value: float) -> float:
    """Convert nanometers to meters (SI units)."""
    return value * 1e-9


def um(value: float) -> float:
    """Convert micrometers to meters (SI units)."""
    return value * 1e-6


def deg(value: float) -> float:
    """Convert degrees to radians."""
    return value * np.pi / 180
