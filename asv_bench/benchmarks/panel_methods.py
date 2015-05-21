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
from pandas_vb_common import *
import pandas.computation.expressions as expr
from pandas import concat, Timestamp
import sqlite3
from sqlalchemy import create_engine
from itertools import product
from string import ascii_letters, digits
from random import randrange
import numpy as np
from pandas.core import common as com


class panel_pct_change_items(object):

    def setup(self):
        self.index = date_range(start='2000', freq='D', periods=1000)
        self.panel = Panel(np.random.randn(100, len(self.index), 1000))

    def time_panel_pct_change_items(self):
        self.panel.pct_change(1, axis='items')


class panel_pct_change_major(object):

    def setup(self):
        self.index = date_range(start='2000', freq='D', periods=1000)
        self.panel = Panel(np.random.randn(100, len(self.index), 1000))

    def time_panel_pct_change_major(self):
        self.panel.pct_change(1, axis='major')


class panel_pct_change_minor(object):

    def setup(self):
        self.index = date_range(start='2000', freq='D', periods=1000)
        self.panel = Panel(np.random.randn(100, len(self.index), 1000))

    def time_panel_pct_change_minor(self):
        self.panel.pct_change(1, axis='minor')


class panel_shift(object):

    def setup(self):
        self.index = date_range(start='2000', freq='D', periods=1000)
        self.panel = Panel(np.random.randn(100, len(self.index), 1000))

    def time_panel_shift(self):
        self.panel.shift(1)


class panel_shift_minor(object):

    def setup(self):
        self.index = date_range(start='2000', freq='D', periods=1000)
        self.panel = Panel(np.random.randn(100, len(self.index), 1000))

    def time_panel_shift_minor(self):
        self.panel.shift(1, axis='minor')