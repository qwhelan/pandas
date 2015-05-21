import pandas as pd
from collections import OrderedDict
from pandas import to_timedelta
from pandas import read_csv, read_table
from random import shuffle
from pandas.util.decorators import cache_readonly
from pandas.tseries.frequencies import infer_freq
import pandas.tseries.holiday
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

    raise

from pandas.core import common as com
from datetime import timedelta
import string
import sqlite3
import datetime as dt
from pandas_vb_common import *
import os
from pandas.compat import range
from cStringIO import StringIO
from pandas import concat, Timestamp
from string import ascii_letters, digits
import pandas.sparse.series
from pandas.tseries.converter import DatetimeConverter
import sqlalchemy
import pandas.computation.expressions as expr
from pandas.core.sparse import SparseDataFrame
import pandas.util.testing as testing
import itertools as IT
from sqlalchemy import create_engine
import numpy as np
from pandas.core.sparse import SparseSeries, SparseDataFrame


def iter_n(iterable, n=None):
    i = 0
    for _ in iterable:
        i += 1
        if ((n is not None) and (i > n)):
            break


class dataframe_resample_max_numpy(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='20130101', periods=100000, freq='50L')
        self.df = DataFrame(np.random.randn(100000, 2), index=self.rng)

    def time_dataframe_resample_max_numpy(self):
        self.df.resample('1s', how=np.max)


class dataframe_resample_max_string(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='20130101', periods=100000, freq='50L')
        self.df = DataFrame(np.random.randn(100000, 2), index=self.rng)

    def time_dataframe_resample_max_string(self):
        self.df.resample('1s', how='max')


class dataframe_resample_mean_numpy(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='20130101', periods=100000, freq='50L')
        self.df = DataFrame(np.random.randn(100000, 2), index=self.rng)

    def time_dataframe_resample_mean_numpy(self):
        self.df.resample('1s', how=np.mean)


class dataframe_resample_mean_string(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='20130101', periods=100000, freq='50L')
        self.df = DataFrame(np.random.randn(100000, 2), index=self.rng)

    def time_dataframe_resample_mean_string(self):
        self.df.resample('1s', how='mean')


class dataframe_resample_min_numpy(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='20130101', periods=100000, freq='50L')
        self.df = DataFrame(np.random.randn(100000, 2), index=self.rng)

    def time_dataframe_resample_min_numpy(self):
        self.df.resample('1s', how=np.min)


class dataframe_resample_min_string(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='20130101', periods=100000, freq='50L')
        self.df = DataFrame(np.random.randn(100000, 2), index=self.rng)

    def time_dataframe_resample_min_string(self):
        self.df.resample('1s', how='min')


class datetimeindex_add_offset(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='1/1/2000', periods=10000, freq='T')

    def time_datetimeindex_add_offset(self):
        (self.rng + timedelta(minutes=2))


class datetimeindex_converter(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)

    def time_datetimeindex_converter(self):
        DatetimeConverter.convert(self.rng, None, None)


class datetimeindex_infer_dst(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.dst_rng = date_range(start='10/29/2000 1:00:00', end='10/29/2000 1:59:59', freq='S')
        self.index = date_range(start='10/29/2000', end='10/29/2000 00:59:59', freq='S')
        self.index = self.index.append(self.dst_rng)
        self.index = self.index.append(self.dst_rng)
        self.index = self.index.append(date_range(start='10/29/2000 2:00:00', end='10/29/2000 3:00:00', freq='S'))

    def time_datetimeindex_infer_dst(self):
        self.index.tz_localize('US/Eastern', infer_dst=True)


class datetimeindex_normalize(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='1/1/2000 9:30', periods=10000, freq='S', tz='US/Eastern')

    def time_datetimeindex_normalize(self):
        self.rng.normalize()


class datetimeindex_unique(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='1/1/2000', periods=1000, freq='T')
        self.index = self.rng.repeat(10)

    def time_datetimeindex_unique(self):
        self.index.unique()


class dti_reset_index(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='1/1/2000', periods=1000, freq='H')
        self.df = DataFrame(np.random.randn(len(self.rng), 2), self.rng)

    def time_dti_reset_index(self):
        self.df.reset_index()


class dti_reset_index_tz(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='1/1/2000', periods=1000, freq='H', tz='US/Eastern')
        self.df = DataFrame(np.random.randn(len(self.rng), 2), index=self.rng)

    def time_dti_reset_index_tz(self):
        self.df.reset_index()


class period_setitem(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = period_range(start='1/1/1990', freq='S', periods=20000)
        self.df = DataFrame(index=range(len(self.rng)))

    def time_period_setitem(self):
        self.df['col'] = self.rng


class timeseries_1min_5min_mean(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)

    def time_timeseries_1min_5min_mean(self):
        self.ts[:10000].resample('5min', how='mean')


class timeseries_1min_5min_ohlc(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)

    def time_timeseries_1min_5min_ohlc(self):
        self.ts[:10000].resample('5min', how='ohlc')


class timeseries_add_irregular(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.lindex = np.random.permutation(self.N)[:(self.N // 2)]
        self.rindex = np.random.permutation(self.N)[:(self.N // 2)]
        self.left = Series(self.ts.values.take(self.lindex), index=self.ts.index.take(self.lindex))
        self.right = Series(self.ts.values.take(self.rindex), index=self.ts.index.take(self.rindex))

    def time_timeseries_add_irregular(self):
        (self.left + self.right)


class timeseries_asof(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.N = 10000
        self.rng = date_range(start='1/1/1990', periods=self.N, freq='53s')
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.dates = date_range(start='1/1/1990', periods=(self.N * 10), freq='5s')

    def time_timeseries_asof(self):
        self.ts.asof(self.dates)


class timeseries_asof_nan(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.N = 10000
        self.rng = date_range(start='1/1/1990', periods=self.N, freq='53s')
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.dates = date_range(start='1/1/1990', periods=(self.N * 10), freq='5s')
        self.ts[250:5000] = np.nan

    def time_timeseries_asof_nan(self):
        self.ts.asof(self.dates)


class timeseries_asof_single(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.N = 10000
        self.rng = date_range(start='1/1/1990', periods=self.N, freq='53s')
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.dates = date_range(start='1/1/1990', periods=(self.N * 10), freq='5s')

    def time_timeseries_asof_single(self):
        self.ts.asof(self.dates[0])


class timeseries_custom_bday_apply(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_custom_bday_apply(self):
        self.cday.apply(self.date)


class timeseries_custom_bday_apply_dt64(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_custom_bday_apply_dt64(self):
        self.cday.apply(self.dt64)


class timeseries_custom_bday_cal_decr(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_custom_bday_cal_decr(self):
        (self.date - (1 * self.cdayh))


class timeseries_custom_bday_cal_incr(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_custom_bday_cal_incr(self):
        (self.date + (1 * self.cdayh))


class timeseries_custom_bday_cal_incr_n(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_custom_bday_cal_incr_n(self):
        (self.date + (10 * self.cdayh))


class timeseries_custom_bday_cal_incr_neg_n(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_custom_bday_cal_incr_neg_n(self):
        (self.date - (10 * self.cdayh))


class timeseries_custom_bday_decr(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_custom_bday_decr(self):
        (self.date - self.cday)


class timeseries_custom_bday_incr(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_custom_bday_incr(self):
        (self.date + self.cday)


class timeseries_custom_bmonthbegin_decr_n(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_custom_bmonthbegin_decr_n(self):
        (self.date - (10 * self.cmb))


class timeseries_custom_bmonthbegin_incr_n(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_custom_bmonthbegin_incr_n(self):
        (self.date + (10 * self.cmb))


class timeseries_custom_bmonthend_decr_n(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_custom_bmonthend_decr_n(self):
        (self.date - (10 * self.cme))


class timeseries_custom_bmonthend_incr(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_custom_bmonthend_incr(self):
        (self.date + self.cme)


class timeseries_custom_bmonthend_incr_n(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_custom_bmonthend_incr_n(self):
        (self.date + (10 * self.cme))


class timeseries_day_apply(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_day_apply(self):
        self.day.apply(self.date)


class timeseries_day_incr(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_day_incr(self):
        (self.date + self.day)


class timeseries_infer_freq(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='1/1/1700', freq='D', periods=100000)
        self.a = self.rng[:50000].append(self.rng[50002:])

    def time_timeseries_infer_freq(self):
        infer_freq(self.a)


class timeseries_is_month_start(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.N = 10000
        self.rng = date_range(start='1/1/1', periods=self.N, freq='B')

    def time_timeseries_is_month_start(self):
        self.rng.is_month_start


class timeseries_iter_datetimeindex(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.N = 1000000
        self.M = 10000
        self.idx1 = date_range(start='20140101', freq='T', periods=self.N)
        self.idx2 = period_range(start='20140101', freq='T', periods=self.N)

    def time_timeseries_iter_datetimeindex(self):
        iter_n(self.idx1)


class timeseries_iter_datetimeindex_preexit(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.N = 1000000
        self.M = 10000
        self.idx1 = date_range(start='20140101', freq='T', periods=self.N)
        self.idx2 = period_range(start='20140101', freq='T', periods=self.N)


    def time_timeseries_iter_datetimeindex_preexit(self):
        iter_n(self.idx1, self.M)


class timeseries_iter_periodindex(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.N = 1000000
        self.M = 10000
        self.idx1 = date_range(start='20140101', freq='T', periods=self.N)
        self.idx2 = period_range(start='20140101', freq='T', periods=self.N)

    def time_timeseries_iter_periodindex(self):
        iter_n(self.idx2)


class timeseries_iter_periodindex_preexit(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.N = 1000000
        self.M = 10000
        self.idx1 = date_range(start='20140101', freq='T', periods=self.N)
        self.idx2 = period_range(start='20140101', freq='T', periods=self.N)

    def time_timeseries_iter_periodindex_preexit(self):
        iter_n(self.idx2, self.M)


class timeseries_large_lookup_value(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='1/1/2000', periods=1500000, freq='s')
        self.ts = Series(1, index=self.rng)

    def time_timeseries_large_lookup_value(self):
        self.ts[self.ts.index[(len(self.ts) // 2)]]
        self.ts.index._cleanup()


class timeseries_period_downsample_mean(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = period_range(start='1/1/2000', end='1/1/2001', freq='T')
        self.ts = Series(np.random.randn(len(self.rng)), index=self.rng)

    def time_timeseries_period_downsample_mean(self):
        self.ts.resample('D', how='mean')


class timeseries_resample_datetime64(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='2000-01-01 00:00:00', end='2000-01-01 10:00:00', freq='555000U')
        self.int_ts = Series(5, self.rng, dtype='int64')
        self.ts = self.int_ts.astype('datetime64[ns]')

    def time_timeseries_resample_datetime64(self):
        self.ts.resample('1S', how='last')


class timeseries_slice_minutely(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)

    def time_timeseries_slice_minutely(self):
        self.ts[:10000]


class timeseries_sort_index(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.N = 100000
        self.rng = date_range(start='1/1/2000', periods=self.N, freq='s')
        self.rng = self.rng.take(np.random.permutation(self.N))
        self.ts = Series(np.random.randn(self.N), index=self.rng)

    def time_timeseries_sort_index(self):
        self.ts.sort_index()


class timeseries_timestamp_downsample_mean(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='1/1/2000', end='1/1/2001', freq='T')
        self.ts = Series(np.random.randn(len(self.rng)), index=self.rng)

    def time_timeseries_timestamp_downsample_mean(self):
        self.ts.resample('D', how='mean')


class timeseries_timestamp_tzinfo_cons(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='1/1/2000', end='3/1/2000', tz='US/Eastern')

    def time_timeseries_timestamp_tzinfo_cons(self):
        self.rng[0]


class timeseries_to_datetime_YYYYMMDD(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='1/1/2000', periods=10000, freq='D')
        self.strings = Series((((self.rng.year * 10000) + (self.rng.month * 100)) + self.rng.day), dtype=np.int64).apply(str)

    def time_timeseries_to_datetime_YYYYMMDD(self):
        to_datetime(self.strings, format='%Y%m%d')


class timeseries_to_datetime_iso8601(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.rng = date_range(start='1/1/2000', periods=20000, freq='h')
        self.strings = [x.strftime('%Y-%m-%d %H:%M:%S') for x in self.rng]

    def time_timeseries_to_datetime_iso8601(self):
        to_datetime(self.strings)


class timeseries_with_format_no_exact(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.s = Series((['19MAY11', '19MAY11:00:00:00'] * 100000))

    def time_timeseries_with_format_no_exact(self):
        to_datetime(self.s, format='%d%b%y', exact=False)


class timeseries_with_format_replace(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.s = Series((['19MAY11', '19MAY11:00:00:00'] * 100000))

    def time_timeseries_with_format_replace(self):
        to_datetime(self.s.str.replace(':\\S+$', ''), format='%d%b%y')


class timeseries_year_apply(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_year_apply(self):
        self.year.apply(self.date)


class timeseries_year_incr(object):

    def setup(self):
        self.N = 100000
        try:
            self.rng = date_range(start='1/1/2000', periods=self.N, freq='min')
        except NameError:
            self.rng = DatetimeIndex(start='1/1/2000', periods=self.N, freq='T')

        if hasattr(Series, 'convert'):
            Series.resample = Series.convert
        self.ts = Series(np.random.randn(self.N), index=self.rng)
        self.date = dt.datetime(2011, 1, 1)
        self.dt64 = np.datetime64('2011-01-01 09:00Z')
        self.hcal = pd.tseries.holiday.USFederalHolidayCalendar()
        self.day = pd.offsets.Day()
        self.year = pd.offsets.YearBegin()
        self.cday = pd.offsets.CustomBusinessDay()
        self.cmb = pd.offsets.CustomBusinessMonthBegin(calendar=self.hcal)
        self.cme = pd.offsets.CustomBusinessMonthEnd(calendar=self.hcal)
        self.cdayh = pd.offsets.CustomBusinessDay(calendar=self.hcal)

    def time_timeseries_year_incr(self):
        (self.date + self.year)
