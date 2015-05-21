import pandas as pd
from collections import OrderedDict
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
from sqlalchemy import create_engine
import numpy as np
from pandas.core.sparse import SparseSeries, SparseDataFrame


class sparse_series_to_frame(object):

    def setup(self):
        self.K = 50
        self.N = 50000
        self.rng = np.asarray(date_range('1/1/2000', periods=self.N, freq='T'))
        self.series = {}
        for i in range(1, (self.K + 1)):
            self.data = np.random.randn(self.N)[:(- i)]
            self.this_rng = self.rng[:(- i)]
            self.data[100:] = np.nan
            self.series[i] = SparseSeries(self.data, index=self.this_rng)

    def time_sparse_series_to_frame(self):
        SparseDataFrame(self.series)


class sparse_frame_constructor(object):

    def time_sparse_frame_constructor(self):
        SparseDataFrame(columns=np.arange(100), index=np.arange(1000))


class sparse_series_from_coo(object):

    def setup(self):
        self.A = scipy.sparse.coo_matrix(([3.0, 1.0, 2.0], ([1, 0, 0], [0, 2, 3])), shape=(100, 100))

    def time_sparse_series_from_coo(self):
        self.ss = pandas.sparse.series.from_coo(self.A)


class sparse_series_to_coo(object):

    def setup(self):
        self.s = pd.Series(([nan] * 10000))
        self.s[0] = 3.0
        self.s[100] = (-1.0)
        self.s[999] = 12.1
        self.s.index = pd.MultiIndex.from_product((range(10), range(10), range(10), range(10)))
        self.ss = self.s.to_sparse()

    def time_sparse_series_to_coo(self):
        self.ss.to_coo(row_levels=[0, 1], column_levels=[2, 3], sort_labels=True)