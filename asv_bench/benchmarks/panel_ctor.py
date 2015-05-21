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


class panel_from_dict_all_different_indexes(object):

    def setup(self):
        self.data_frames = {}
        self.start = datetime(1990, 1, 1)
        self.end = datetime(2012, 1, 1)
        for x in xrange(100):
            self.end += timedelta(days=1)
            self.dr = np.asarray(date_range(self.start, self.end))
            self.df = DataFrame({'a': ([0] * len(self.dr)), 'b': ([1] * len(self.dr)), 'c': ([2] * len(self.dr)), }, index=self.dr)
            self.data_frames[x] = self.df

    def time_panel_from_dict_all_different_indexes(self):
        Panel.from_dict(self.data_frames)


class panel_from_dict_equiv_indexes(object):

    def setup(self):
        self.data_frames = {}
        for x in xrange(100):
            self.dr = np.asarray(DatetimeIndex(start=datetime(1990, 1, 1), end=datetime(2012, 1, 1), freq=datetools.Day(1)))
            self.df = DataFrame({'a': ([0] * len(self.dr)), 'b': ([1] * len(self.dr)), 'c': ([2] * len(self.dr)), }, index=self.dr)
            self.data_frames[x] = self.df

    def time_panel_from_dict_equiv_indexes(self):
        Panel.from_dict(self.data_frames)


class panel_from_dict_same_index(object):

    def setup(self):
        self.dr = np.asarray(DatetimeIndex(start=datetime(1990, 1, 1), end=datetime(2012, 1, 1), freq=datetools.Day(1)))
        self.data_frames = {}
        for x in xrange(100):
            self.df = DataFrame({'a': ([0] * len(self.dr)), 'b': ([1] * len(self.dr)), 'c': ([2] * len(self.dr)), }, index=self.dr)
            self.data_frames[x] = self.df

    def time_panel_from_dict_same_index(self):
        Panel.from_dict(self.data_frames)


class panel_from_dict_two_different_indexes(object):

    def setup(self):
        self.data_frames = {}
        self.start = datetime(1990, 1, 1)
        self.end = datetime(2012, 1, 1)
        for x in xrange(100):
            if (x == 50):
                self.end += timedelta(days=1)
            self.dr = np.asarray(date_range(self.start, self.end))
            self.df = DataFrame({'a': ([0] * len(self.dr)), 'b': ([1] * len(self.dr)), 'c': ([2] * len(self.dr)), }, index=self.dr)
            self.data_frames[x] = self.df

    def time_panel_from_dict_two_different_indexes(self):
        Panel.from_dict(self.data_frames)