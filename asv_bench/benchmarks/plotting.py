from numpy.random import randint
import pandas as pd
from pandas.util.decorators import cache_readonly
import sqlalchemy
from collections import OrderedDict
import os
try:
    from pandas.tseries.offsets import *
except:
    from pandas.core.datetools import *
from pandas import read_csv, read_table
from pandas_vb_common import *
import pandas.computation.expressions as expr
from pandas import concat, Timestamp
import sqlite3
try:
    from pandas import date_range
except ImportError:

    def date_range(start=None, end=None, periods=None, freq=None):
        return DatetimeIndex(start, end, periods=periods, offset=freq)
from cStringIO import StringIO
from sqlalchemy import create_engine
from itertools import product
from string import ascii_letters, digits
from random import randrange
import numpy as np
from pandas.core import common as com


class plot_timeseries_period(object):

    def setup(self):
        self.N = 2000
        self.M = 5
        self.df = DataFrame(np.random.randn(self.N, self.M), index=date_range('1/1/1975', periods=self.N))

    def time_plot_timeseries_period(self):
        self.df.plot()