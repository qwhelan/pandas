# -*- coding: utf-8 -*-
from decimal import Decimal
from fractions import Fraction
from numbers import Number

import sys

import cython
from cython import Py_ssize_t

from cpython cimport (Py_INCREF, PyTuple_SET_ITEM,
                      PyTuple_New,
                      Py_EQ,
                      PyObject_RichCompareBool)

from cpython.datetime cimport (PyDateTime_Check, PyDate_Check,
                               PyTime_Check, PyDelta_Check,
                               PyDateTime_IMPORT)
PyDateTime_IMPORT

import numpy as np
cimport numpy as cnp
from numpy cimport (ndarray, PyArray_GETITEM,
                    PyArray_ITER_DATA, PyArray_ITER_NEXT, PyArray_IterNew,
                    flatiter, NPY_OBJECT,
                    int64_t,
                    float32_t, float64_t,
                    uint8_t, uint64_t,
                    complex128_t)
cnp.import_array()

cdef extern from "numpy/arrayobject.h":
    # cython's numpy.dtype specification is incorrect, which leads to
    # errors in issubclass(self.dtype.type, np.bool_), so we directly
    # include the correct version
    # https://github.com/cython/cython/issues/2022

    ctypedef class numpy.dtype [object PyArray_Descr]:
        # Use PyDataType_* macros when possible, however there are no macros
        # for accessing some of the fields, so some are defined. Please
        # ask on cython-dev if you need more.
        cdef int type_num
        cdef int itemsize "elsize"
        cdef char byteorder
        cdef object fields
        cdef tuple names


cdef extern from "src/parse_helper.h":
    int floatify(object, float64_t *result, int *maybe_int) except -1

cimport pandas._libs.util as util
from pandas._libs.util cimport is_nan, UINT64_MAX, INT64_MAX, INT64_MIN

#from pandas._libs.tslib cimport array_to_datetime
from pandas._libs.tslibs.nattype cimport NPY_NAT
from pandas._libs.tslibs.nattype import NaT
from pandas._libs.tslibs.conversion cimport convert_to_tsobject
from pandas._libs.tslibs.timedeltas cimport convert_to_timedelta64
from pandas._libs.tslibs.timezones cimport get_timezone, tz_compare

from pandas._libs.missing cimport (
    checknull, isnaobj, is_null_datetime64, is_null_timedelta64, is_null_period
)


# constants that will be compared to potentially arbitrarily large
# python int
cdef object oINT64_MAX = <int64_t>INT64_MAX
cdef object oINT64_MIN = <int64_t>INT64_MIN
cdef object oUINT64_MAX = <uint64_t>UINT64_MAX

cdef bint PY2 = sys.version_info[0] == 2
cdef float64_t NaN = <float64_t>np.NaN


def is_scalar(val: object) -> bool:
    """
    Return True if given value is scalar.

    Parameters
    ----------
    val : object
        This includes:

        - numpy array scalar (e.g. np.int64)
        - Python builtin numerics
        - Python builtin byte arrays and strings
        - None
        - datetime.datetime
        - datetime.timedelta
        - Period
        - decimal.Decimal
        - Interval
        - DateOffset
        - Fraction
        - Number

    Returns
    -------
    bool
        Return True if given object is scalar, False otherwise

    Examples
    --------
    >>> dt = pd.datetime.datetime(2018, 10, 3)
    >>> pd.is_scalar(dt)
    True

    >>> pd.api.types.is_scalar([2, 3])
    False

    >>> pd.api.types.is_scalar({0: 1, 2: 3})
    False

    >>> pd.api.types.is_scalar((0, 2))
    False

    pandas supports PEP 3141 numbers:

    >>> from fractions import Fraction
    >>> pd.api.types.is_scalar(Fraction(3, 5))
    True
    """

    return (cnp.PyArray_IsAnyScalar(val)
            # PyArray_IsAnyScalar is always False for bytearrays on Py3
            or isinstance(val, (Fraction, Number))
            # We differ from numpy, which claims that None is not scalar;
            # see np.isscalar
            or val is None
            or PyDate_Check(val)
            or PyDelta_Check(val)
            or PyTime_Check(val)
            or util.is_period_object(val)
            or is_decimal(val)
            or is_interval(val)
            or util.is_offset_object(val))


# core.common import for fast inference checks

def is_float(obj: object) -> bool:
    return util.is_float_object(obj)


def is_integer(obj: object) -> bool:
    return util.is_integer_object(obj)


def is_bool(obj: object) -> bool:
    return util.is_bool_object(obj)


def is_complex(obj: object) -> bool:
    return util.is_complex_object(obj)


cpdef bint is_decimal(object obj):
    return isinstance(obj, Decimal)


cpdef bint is_interval(object obj):
    return getattr(obj, '_typ', '_typ') == 'interval'


def is_period(val: object) -> bool:
    """ Return a boolean if this is a Period object """
    return util.is_period_object(val)


cdef inline bint is_timedelta(object o):
    return PyDelta_Check(o) or util.is_timedelta64_object(o)
