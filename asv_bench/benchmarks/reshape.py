import pandas as pd
from collections import OrderedDict
from pandas import read_csv, read_table
from random import shuffle
from pandas.util.decorators import cache_readonly
from random import randrange
from pandas.core.reshape import melt
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
import sqlalchemy
import pandas.computation.expressions as expr
from sqlalchemy import create_engine
import numpy as np


class melt_dataframe(object):

    def setup(self):
        self.index = MultiIndex.from_arrays([np.arange(100).repeat(100), np.roll(np.tile(np.arange(100), 100), 25)])
        self.df = DataFrame(np.random.randn(10000, 4), index=self.index)
        self.df = DataFrame(np.random.randn(10000, 3), columns=['A', 'B', 'C'])
        self.df['id1'] = np.random.randint(0, 10, 10000)
        self.df['id2'] = np.random.randint(100, 1000, 10000)

    def time_melt_dataframe(self):
        melt(self.df, id_vars=['id1', 'id2'])


class reshape_pivot_time_series(object):

    def setup(self):
        self.index = MultiIndex.from_arrays([np.arange(100).repeat(100), np.roll(np.tile(np.arange(100), 100), 25)])
        self.df = DataFrame(np.random.randn(10000, 4), index=self.index)

        def unpivot(frame):
            (N, K) = frame.shape
            self.data = {'value': frame.values.ravel('F'), 'variable': np.asarray(frame.columns).repeat(N), 'date': np.tile(np.asarray(frame.index), K), }
            return DataFrame(self.data, columns=['date', 'variable', 'value'])
        self.index = date_range('1/1/2000', periods=10000, freq='h')
        self.df = DataFrame(randn(10000, 50), index=self.index, columns=range(50))
        self.pdf = unpivot(self.df)
        self.f = (lambda : self.pdf.pivot('date', 'variable', 'value'))

    def time_reshape_pivot_time_series(self):
        self.f()


class reshape_stack_simple(object):

    def setup(self):
        self.index = MultiIndex.from_arrays([np.arange(100).repeat(100), np.roll(np.tile(np.arange(100), 100), 25)])
        self.df = DataFrame(np.random.randn(10000, 4), index=self.index)
        self.udf = self.df.unstack(1)

    def time_reshape_stack_simple(self):
        self.udf.stack()


class reshape_unstack_simple(object):

    def setup(self):
        self.index = MultiIndex.from_arrays([np.arange(100).repeat(100), np.roll(np.tile(np.arange(100), 100), 25)])
        self.df = DataFrame(np.random.randn(10000, 4), index=self.index)

    def time_reshape_unstack_simple(self):
        self.df.unstack(1)


class unstack_sparse_keyspace(object):

    def setup(self):
        self.index = MultiIndex.from_arrays([np.arange(100).repeat(100), np.roll(np.tile(np.arange(100), 100), 25)])
        self.df = DataFrame(np.random.randn(10000, 4), index=self.index)
        self.NUM_ROWS = 1000
        for iter in range(10):
            self.df = DataFrame({'A': np.random.randint(50, size=self.NUM_ROWS), 'B': np.random.randint(50, size=self.NUM_ROWS), 'C': np.random.randint((-10), 10, size=self.NUM_ROWS), 'D': np.random.randint((-10), 10, size=self.NUM_ROWS), 'E': np.random.randint(10, size=self.NUM_ROWS), 'F': np.random.randn(self.NUM_ROWS), })
            self.idf = self.df.set_index(['A', 'B', 'C', 'D', 'E'])
            if (len(self.idf.index.unique()) == self.NUM_ROWS):
                break

    def time_unstack_sparse_keyspace(self):
        self.idf.unstack()