import ctypes  # C interop helpers
import math
from enum import Enum

import numpy as np  # Packing of structures in C-compatible format
from numba import cuda, float32, int32, types, uint8, uint32
from numba.core.extending import overload
from numba.cuda import get_current_device
from numba.cuda.compiler import compile_cuda as numba_compile_cuda
from numba.cuda.libdevice import fast_powf, float_as_int, int_as_float

from operator import add, mul, sub, truediv
from typing import List, Tuple
from enum import IntEnum

from llvmlite import ir
from numba import cuda, float32, int32, types, uchar, uint8, uint32
from numba.core import cgutils
import numba
from numba.core.extending import (
    make_attribute_wrapper,
    models,
    overload,
    register_model,
    type_callable,
    typeof_impl,
)
from numba.core.imputils import lower_constant
from numba.core.typing.templates import (
    AttributeTemplate,
    ConcreteTemplate,
    signature,
)
from numba.cuda.cudadecl import register, register_attr, register_global
from numba.cuda.cudadrv import nvvm
from numba.cuda.cudaimpl import lower
from numba.cuda.types import dim3


class VectorType(types.Type):
    def __init__(self, name, base_type, attr_names):
        self._base_type = base_type
        self._attr_names = attr_names
        super().__init__(name=name)

    @property
    def base_type(self):
        return self._base_type

    @property
    def attr_names(self):
        return self._attr_names

    @property
    def num_elements(self):
        return len(self._attr_names)


def make_vector_type(
    name: str, base_type: types.Type, attr_names: List[str]
) -> types.Type:
    """Create a vector type.

    Parameters
    ----------
    name: str
        The name of the type.
    base_type: numba.types.Type
        The primitive type for each element in the vector.
    attr_names: list of str
        Name for each attribute.
    """

    class _VectorType(VectorType):
        """Internal instantiation of VectorType."""

        pass

    class VectorTypeModel(models.StructModel):
        def __init__(self, dmm, fe_type):
            members = [(attr_name, base_type) for attr_name in attr_names]
            super().__init__(dmm, fe_type, members)

    vector_type = _VectorType(name, base_type, attr_names)
    register_model(_VectorType)(VectorTypeModel)
    for attr_name in attr_names:
        make_attribute_wrapper(_VectorType, attr_name, attr_name)

    return vector_type


def make_vector_type_factory(
    vector_type: types.Type, overloads: List[Tuple[types.Type]]
):
    """Make a factory function for ``vector_type``

    Parameters
    ----------
    vector_type: VectorType
        The type to create factory function for.
    overloads: List of argument types tuples
        A list containing different overloads of the factory function. Each
        base type in the tuple should either be primitive type or VectorType.
    """

    def func():
        pass

    class FactoryTemplate(ConcreteTemplate):
        key = func
        cases = [signature(vector_type, *arglist) for arglist in overloads]

    def make_lower_factory(fml_arg_list):
        """Meta function to create a lowering for the factory function. Flattens
        the arguments by converting vector_type into load instructions for each
        of its attributes. Such as float2 -> float2.x, float2.y.
        """

        def lower_factory(context, builder, sig, actual_args):
            # A list of elements to assign from
            source_list = []
            # Convert the list of argument types to a list of load IRs.
            for argidx, fml_arg in enumerate(fml_arg_list):
                if isinstance(fml_arg, VectorType):
                    pxy = cgutils.create_struct_proxy(fml_arg)(
                        context, builder, actual_args[argidx]
                    )
                    source_list += [getattr(pxy, attr) for attr in fml_arg.attr_names]
                else:
                    # assumed primitive type
                    source_list.append(actual_args[argidx])

            if len(source_list) != vector_type.num_elements:
                raise numba.core.TypingError(
                    f"Unmatched number of source elements ({len(source_list)}) "
                    "and target elements ({vector_type.num_elements})."
                )

            out = cgutils.create_struct_proxy(vector_type)(context, builder)

            for attr_name, source in zip(vector_type.attr_names, source_list):
                setattr(out, attr_name, source)
            return out._getvalue()

        return lower_factory

    func.__name__ = f"make_{vector_type.name.lower()}"
    register(FactoryTemplate)
    register_global(func, types.Function(FactoryTemplate))
    for arglist in overloads:
        lower_factory = make_lower_factory(arglist)
        lower(func, *arglist)(lower_factory)
    return func


def lower_vector_type_binops(
    binop, vector_type: VectorType, overloads: List[Tuple[types.Type]]
):
    """Lower binops for ``vector_type``

    Parameters
    ----------
    binop: operation
        The binop to lower
    vector_type: VectorType
        The type to lower op for.
    overloads: List of argument types tuples
        A list containing different overloads of the binop. Expected to be either
            - vector_type x vector_type
            - primitive_type x vector_type
            - vector_type x primitive_type.
        In case one of the oprand is primitive_type, the operation is broadcasted.
    """
    # Should we assume the above are the only possible cases?
    class Vector_op_template(ConcreteTemplate):
        key = binop
        cases = [signature(vector_type, *arglist) for arglist in overloads]

    def make_lower_op(fml_arg_list):
        def op_impl(context, builder, sig, actual_args):
            def _make_load_IR(typ, actual_arg):
                if isinstance(typ, VectorType):
                    pxy = cgutils.create_struct_proxy(typ)(context, builder, actual_arg)
                    oprands = [getattr(pxy, attr) for attr in typ.attr_names]
                else:
                    # Assumed primitive type, broadcast
                    oprands = [actual_arg for _ in range(vector_type.num_elements)]
                return oprands

            def element_wise_op(lhs, rhs, res, attr):
                setattr(
                    res,
                    attr,
                    context.compile_internal(
                        builder,
                        lambda x, y: binop(x, y),
                        signature(types.float32, types.float32, types.float32),
                        (lhs, rhs),
                    ),
                )

            lhs_typ, rhs_typ = fml_arg_list
            # Construct a list of load IRs
            lhs = _make_load_IR(lhs_typ, actual_args[0])
            rhs = _make_load_IR(rhs_typ, actual_args[1])

            if not len(lhs) == len(rhs) == vector_type.num_elements:
                raise numba.core.TypingError(
                    f"Unmatched number of lhs elements ({len(lhs)}), rhs elements ({len(rhs)}) "
                    "and target elements ({vector_type.num_elements})."
                )

            out = cgutils.create_struct_proxy(vector_type)(context, builder)
            for attr, l, r in zip(vector_type.attr_names, lhs, rhs):
                element_wise_op(l, r, out, attr)

            return out._getvalue()

        return op_impl

    register_global(binop, types.Function(Vector_op_template))
    for arglist in overloads:
        impl = make_lower_op(arglist)
        lower(binop, *arglist)(impl)


# Register basic types
uchar4 = make_vector_type("UChar4", uchar, ["x", "y", "z", "w"])
float3 = make_vector_type("Float3", float32, ["x", "y", "z"])
float2 = make_vector_type("Float2", float32, ["x", "y"])
uint3 = make_vector_type("UInt3", uint32, ["x", "y", "z"])

# Register factory functions
make_uchar4 = make_vector_type_factory(uchar4, [(uchar,) * 4])
make_float3 = make_vector_type_factory(float3, [(float32,) * 3, (float2, float32)])
make_float2 = make_vector_type_factory(float2, [(float32,) * 2])
make_uint3 = make_vector_type_factory(uint3, [(uint32,) * 3])

# Lower Vector Type Ops
## float3
lower_vector_type_binops(
    add, float3, [(float3, float3), (float32, float3), (float3, float32)]
)
lower_vector_type_binops(
    sub, float3, [(float3, float3), (float32, float3), (float3, float32)]
)
lower_vector_type_binops(
    mul, float3, [(float3, float3), (float32, float3), (float3, float32)]
)
lower_vector_type_binops(
    truediv, float3, [(float3, float3), (float32, float3), (float3, float32)]
)
## float2
lower_vector_type_binops(
    mul, float2, [(float2, float2), (float32, float2), (float2, float32)]
)
lower_vector_type_binops(
    sub, float2, [(float2, float2), (float32, float2), (float2, float32)]
)


# Overload for Clamp
def clamp(x, a, b):
    pass


@overload(clamp, target="cuda", fastmath=True)
def jit_clamp(x, a, b):
    if (
        isinstance(x, types.Float)
        and isinstance(a, types.Float)
        and isinstance(b, types.Float)
    ):

        def clamp_float_impl(x, a, b):
            return max(a, min(x, b))

        return clamp_float_impl
    elif (
        isinstance(x, type(float3))
        and isinstance(a, types.Float)
        and isinstance(b, types.Float)
    ):

        def clamp_float3_impl(x, a, b):
            return make_float3(clamp(x.x, a, b), clamp(x.y, a, b), clamp(x.z, a, b))

        return clamp_float3_impl


def dot(a, b):
    pass


@overload(dot, target="cuda", fastmath=True)
def jit_dot(a, b):
    if isinstance(a, type(float3)) and isinstance(b, type(float3)):

        def dot_float3_impl(a, b):
            return a.x * b.x + a.y * b.y + a.z * b.z

        return dot_float3_impl


@cuda.jit(device=True, fastmath=True)
def normalize(v):
    invLen = float32(1.0) / math.sqrt(dot(v, v))
    return v * invLen


@cuda.jit(device=True, fastmath=True)
def length(v):
    return math.sqrt(dot(v, v))

@cuda.jit(device=True, fastmath=True)
def rotate(u, v, theta):
    return make_float3(
        (math.cos(theta) + u.x * u.x * (1. - math.cos(theta))) * v.x + v.y * (u.x * u.y * (1. - math.cos(theta)) - u.z * math.sin(theta)) + v.z * (u.x * u.z * (1 - math.cos(theta)) + u.y * math.sin(theta)),
        v.x * (u.y * u.x * (1. - math.cos(theta)) + u.z * math.sin(theta)) + v.y * (math.cos(theta) + u.y * u.y * (1. - math.cos(theta))) + v.z * (u.y * u.z * (1. - math.cos(theta)) - u.x * math.sin(theta)),
        v.x * (u.z * u.x * (1. - math.cos(theta)) - u.y * math.sin(theta)) + v.y * (u.z * u.y * (1. - math.cos(theta)) + u.x * math.sin(theta)) + v.z * (math.cos(theta) + u.z * u.z * (1. - math.cos(theta)))
    )


def cross(a, b):
    pass


@overload(cross, target="cuda", fastmath=True)
def jit_cross(a, b):
    if isinstance(a, type(float3)) and isinstance(b, type(float3)):

        def cross_float3_impl(a, b):
            return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)

        return cross_float3_impl