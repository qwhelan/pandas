import pandas as pd
from collections import OrderedDict
from pandas import to_timedelta
from pandas import read_csv, read_table
from random import shuffle
from pandas.util.decorators import cache_readonly
from random import randrange
from pandas.core.reshape import melt
import scipy.sparse
from numpy.random import randint
try:
    from pandas.tseries.offsets import *
except:
    from pandas.core.datetools import *
from itertools import product
try:
    from pandas import date_range
except ImportError:

    def date_range(start=None, end=None, periods=None, freq=None):
        return DatetimeIndex(start, end, periods=periods, offset=freq)
from pandas.core import common as com
from datetime import timedelta
import string
import sqlite3
from pandas_vb_common import *
import os
from pandas.compat import range
from cStringIO import StringIO
from pandas import concat, Timestamp
from string import ascii_letters, digits
import pandas.sparse.series
import sqlalchemy
import pandas.computation.expressions as expr
from pandas.core.sparse import SparseDataFrame
import pandas.util.testing as testing
import itertools as IT
from sqlalchemy import create_engine
import numpy as np
from pandas.core.sparse import SparseSeries, SparseDataFrame


class timedelta_convert_int(object):

    def setup(self):
        self.arr = np.random.randint(0, 1000, size=10000)

    def time_timedelta_convert_int(self):
        to_timedelta(self.arr, unit='s')


class timedelta_convert_string(object):

    def setup(self):
        self.arr = np.random.randint(0, 1000, size=10000)
        self.arr = ['{0} days'.format(i) for i in self.arr]

    def time_timedelta_convert_string(self):
        to_timedelta(self.arr)


class timedelta_convert_string_seconds(object):

    def setup(self):
        self.arr = np.random.randint(0, 60, size=10000)
        self.arr = ['00:00:{0:02d}'.format(i) for i in self.arr]

    def time_timedelta_convert_string_seconds(self):
        to_timedelta(self.arr)