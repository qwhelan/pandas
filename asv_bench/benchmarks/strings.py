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


class strings_cat(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_cat(self):
        self.many.str.cat(sep=',')


class strings_center(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_center(self):
        self.many.str.center(100)


class strings_contains_few(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_contains_few(self):
        self.few.str.contains('matchthis')


class strings_contains_few_noregex(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_contains_few_noregex(self):
        self.few.str.contains('matchthis', regex=False)


class strings_contains_many(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_contains_many(self):
        self.many.str.contains('matchthis')


class strings_contains_many_noregex(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_contains_many_noregex(self):
        self.many.str.contains('matchthis', regex=False)


class strings_count(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_count(self):
        self.many.str.count('matchthis')


class strings_encode_decode(object):

    def setup(self):
        self.ser = Series(testing.makeUnicodeIndex())

    def time_strings_encode_decode(self):
        self.ser.str.encode('utf-8').str.decode('utf-8')


class strings_endswith(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_endswith(self):
        self.many.str.endswith('matchthis')


class strings_extract(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_extract(self):
        self.many.str.extract('(\\w*)matchthis(\\w*)')


class strings_findall(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_findall(self):
        self.many.str.findall('[A-Z]+')


class strings_get(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_get(self):
        self.many.str.get(0)


class strings_get_dummies(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)
        self.s = make_series(string.uppercase, strlen=10, size=10000).str.join('|')

    def time_strings_get_dummies(self):
        self.s.str.get_dummies('|')


class strings_join_split(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_join_split(self):
        self.many.str.join('--').str.split('--')


class strings_len(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_len(self):
        self.many.str.len()


class strings_lower(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_lower(self):
        self.many.str.lower()


class strings_lstrip(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_lstrip(self):
        self.many.str.lstrip('matchthis')


class strings_match(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_match(self):
        self.many.str.match('mat..this')


class strings_pad(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_pad(self):
        self.many.str.pad(100, side='both')


class strings_repeat(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_repeat(self):
        self.many.str.repeat(list(IT.islice(IT.cycle(range(1, 4)), len(self.many))))


class strings_replace(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_replace(self):
        self.many.str.replace('(matchthis)', '\x01\x01')


class strings_rstrip(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_rstrip(self):
        self.many.str.rstrip('matchthis')


class strings_slice(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_slice(self):
        self.many.str.slice(5, 15, 2)


class strings_startswith(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_startswith(self):
        self.many.str.startswith('matchthis')


class strings_strip(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_strip(self):
        self.many.str.strip('matchthis')


class strings_title(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_title(self):
        self.many.str.title()


class strings_upper(object):

    def setup(self):

        def make_series(letters, strlen, size):
            return Series(np.fromiter(IT.cycle(letters), count=(size * strlen), dtype='|S1').view('|S{}'.format(strlen)))
        self.many = make_series(('matchthis' + string.uppercase), strlen=19, size=10000)
        self.few = make_series(('matchthis' + (string.uppercase * 42)), strlen=19, size=10000)

    def time_strings_upper(self):
        self.many.str.upper()