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


class series_nlargest1(object):

    def setup(self):
        self.s1 = Series(np.random.randn(10000))
        self.s2 = Series(np.random.randint(1, 10, 10000))

    def time_series_nlargest1(self):
        self.s1.nlargest(3, take_last=True)
        self.s1.nlargest(3, take_last=False)


class series_nlargest2(object):

    def setup(self):
        self.s1 = Series(np.random.randn(10000))
        self.s2 = Series(np.random.randint(1, 10, 10000))

    def time_series_nlargest2(self):
        self.s2.nlargest(3, take_last=True)
        self.s2.nlargest(3, take_last=False)


class series_nsmallest2(object):

    def setup(self):
        self.s1 = Series(np.random.randn(10000))
        self.s2 = Series(np.random.randint(1, 10, 10000))

    def time_series_nsmallest2(self):
        self.s2.nsmallest(3, take_last=True)
        self.s2.nsmallest(3, take_last=False)