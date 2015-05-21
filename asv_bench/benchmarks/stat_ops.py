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


class stat_ops_frame_mean_float_axis_0(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(100000, 4))
        self.dfi = DataFrame(np.random.randint(1000, size=self.df.shape))

    def time_stat_ops_frame_mean_float_axis_0(self):
        self.df.mean()


class stat_ops_frame_mean_float_axis_1(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(100000, 4))
        self.dfi = DataFrame(np.random.randint(1000, size=self.df.shape))

    def time_stat_ops_frame_mean_float_axis_1(self):
        self.df.mean(1)


class stat_ops_frame_mean_int_axis_0(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(100000, 4))
        self.dfi = DataFrame(np.random.randint(1000, size=self.df.shape))

    def time_stat_ops_frame_mean_int_axis_0(self):
        self.dfi.mean()


class stat_ops_frame_mean_int_axis_1(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(100000, 4))
        self.dfi = DataFrame(np.random.randint(1000, size=self.df.shape))

    def time_stat_ops_frame_mean_int_axis_1(self):
        self.dfi.mean(1)


class stat_ops_frame_sum_float_axis_0(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(100000, 4))
        self.dfi = DataFrame(np.random.randint(1000, size=self.df.shape))

    def time_stat_ops_frame_sum_float_axis_0(self):
        self.df.sum()


class stat_ops_frame_sum_float_axis_1(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(100000, 4))
        self.dfi = DataFrame(np.random.randint(1000, size=self.df.shape))

    def time_stat_ops_frame_sum_float_axis_1(self):
        self.df.sum(1)


class stat_ops_frame_sum_int_axis_0(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(100000, 4))
        self.dfi = DataFrame(np.random.randint(1000, size=self.df.shape))

    def time_stat_ops_frame_sum_int_axis_0(self):
        self.dfi.sum()


class stat_ops_frame_sum_int_axis_1(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(100000, 4))
        self.dfi = DataFrame(np.random.randint(1000, size=self.df.shape))

    def time_stat_ops_frame_sum_int_axis_1(self):
        self.dfi.sum(1)


class stat_ops_level_frame_sum(object):

    def setup(self):
        self.index = MultiIndex(levels=[np.arange(10), np.arange(100), np.arange(100)], labels=[np.arange(10).repeat(10000), np.tile(np.arange(100).repeat(100), 10), np.tile(np.tile(np.arange(100), 100), 10)])
        random.shuffle(self.index.values)
        self.df = DataFrame(np.random.randn(len(self.index), 4), index=self.index)
        self.df_level = DataFrame(np.random.randn(100, 4), index=self.index.levels[1])

    def time_stat_ops_level_frame_sum(self):
        self.df.sum(level=1)


class stat_ops_level_frame_sum_multiple(object):

    def setup(self):
        self.index = MultiIndex(levels=[np.arange(10), np.arange(100), np.arange(100)], labels=[np.arange(10).repeat(10000), np.tile(np.arange(100).repeat(100), 10), np.tile(np.tile(np.arange(100), 100), 10)])
        random.shuffle(self.index.values)
        self.df = DataFrame(np.random.randn(len(self.index), 4), index=self.index)
        self.df_level = DataFrame(np.random.randn(100, 4), index=self.index.levels[1])

    def time_stat_ops_level_frame_sum_multiple(self):
        self.df.sum(level=[0, 1])


class stat_ops_level_series_sum(object):

    def setup(self):
        self.index = MultiIndex(levels=[np.arange(10), np.arange(100), np.arange(100)], labels=[np.arange(10).repeat(10000), np.tile(np.arange(100).repeat(100), 10), np.tile(np.tile(np.arange(100), 100), 10)])
        random.shuffle(self.index.values)
        self.df = DataFrame(np.random.randn(len(self.index), 4), index=self.index)
        self.df_level = DataFrame(np.random.randn(100, 4), index=self.index.levels[1])

    def time_stat_ops_level_series_sum(self):
        self.df[1].sum(level=1)


class stat_ops_level_series_sum_multiple(object):

    def setup(self):
        self.index = MultiIndex(levels=[np.arange(10), np.arange(100), np.arange(100)], labels=[np.arange(10).repeat(10000), np.tile(np.arange(100).repeat(100), 10), np.tile(np.tile(np.arange(100), 100), 10)])
        random.shuffle(self.index.values)
        self.df = DataFrame(np.random.randn(len(self.index), 4), index=self.index)
        self.df_level = DataFrame(np.random.randn(100, 4), index=self.index.levels[1])

    def time_stat_ops_level_series_sum_multiple(self):
        self.df[1].sum(level=[0, 1])


class stat_ops_series_std(object):

    def setup(self):
        self.s = Series(np.random.randn(100000), index=np.arange(100000))
        self.s[::2] = np.nan

    def time_stat_ops_series_std(self):
        self.s.std()


class stats_corr_spearman(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(1000, 30))

    def time_stats_corr_spearman(self):
        self.df.corr(method='spearman')


class stats_rank2d_axis0_average(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(5000, 50))

    def time_stats_rank2d_axis0_average(self):
        self.df.rank()


class stats_rank2d_axis1_average(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(5000, 50))

    def time_stats_rank2d_axis1_average(self):
        self.df.rank(1)


class stats_rank_average(object):

    def setup(self):
        self.values = np.concatenate([np.arange(100000), np.random.randn(100000), np.arange(100000)])
        self.s = Series(self.values)

    def time_stats_rank_average(self):
        self.s.rank()


class stats_rank_average_int(object):

    def setup(self):
        self.values = np.random.randint(0, 100000, size=200000)
        self.s = Series(self.values)

    def time_stats_rank_average_int(self):
        self.s.rank()


class stats_rank_pct_average(object):

    def setup(self):
        self.values = np.concatenate([np.arange(100000), np.random.randn(100000), np.arange(100000)])
        self.s = Series(self.values)

    def time_stats_rank_pct_average(self):
        self.s.rank(pct=True)


class stats_rank_pct_average_old(object):

    def setup(self):
        self.values = np.concatenate([np.arange(100000), np.random.randn(100000), np.arange(100000)])
        self.s = Series(self.values)

    def time_stats_rank_pct_average_old(self):
        (self.s.rank() / len(self.s))


class stats_rolling_mean(object):

    def setup(self):
        self.arr = np.random.randn(100000)

    def time_stats_rolling_mean(self):
        rolling_mean(self.arr, 100)