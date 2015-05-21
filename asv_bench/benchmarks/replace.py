import pandas as pd
from collections import OrderedDict
from pandas import read_csv, read_table
from random import shuffle
from pandas.util.decorators import cache_readonly
from random import randrange
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


class replace_fillna(object):

    def setup(self):
        self.N = 1000000
        try:
            self.rng = date_range('1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex('1/1/2000', periods=self.N, offset=datetools.Minute())
            self.date_range = DateRange
        self.ts = Series(np.random.randn(self.N), index=self.rng)

    def time_replace_fillna(self):
        self.ts.fillna(0.0, inplace=True)


class replace_large_dict(object):

    def setup(self):
        self.n = (10 ** 6)
        self.start_value = (10 ** 5)
        self.to_rep = dict(((i, (self.start_value + i)) for i in range(self.n)))
        self.s = Series(np.random.randint(self.n, size=(10 ** 3)))

    def time_replace_large_dict(self):
        self.s.replace(self.to_rep, inplace=True)


class replace_replacena(object):

    def setup(self):
        self.N = 1000000
        try:
            self.rng = date_range('1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex('1/1/2000', periods=self.N, offset=datetools.Minute())
            self.date_range = DateRange
        self.ts = Series(np.random.randn(self.N), index=self.rng)

    def time_replace_replacena(self):
        self.ts.replace(np.nan, 0.0, inplace=True)