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


class query_with_boolean_selection(object):

    def setup(self):
        self.N = 1000000
        self.halfway = ((self.N // 2) - 1)
        self.index = date_range('20010101', periods=self.N, freq='T')
        self.s = Series(self.index)
        self.ts = self.s.iloc[self.halfway]
        self.N = 1000000
        self.df = DataFrame({'a': np.random.randn(self.N), })
        self.min_val = self.df['a'].min()
        self.max_val = self.df['a'].max()

    def time_query_with_boolean_selection(self):
        self.df.query('(a >= @min_val) & (a <= @max_val)')


class query_with_boolean_selection(object):

    def setup(self):
        self.N = 1000000
        self.halfway = ((self.N // 2) - 1)
        self.index = date_range('20010101', periods=self.N, freq='T')
        self.s = Series(self.index)
        self.ts = self.s.iloc[self.halfway]
        self.N = 1000000
        self.df = DataFrame({'a': np.random.randn(self.N), })
        self.min_val = self.df['a'].min()
        self.max_val = self.df['a'].max()

    def time_query_with_boolean_selection(self):
        self.df.query('(a >= @min_val) & (a <= @max_val)')