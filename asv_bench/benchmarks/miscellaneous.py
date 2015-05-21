import pandas as pd
from pandas.util.decorators import cache_readonly
import sqlalchemy
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


class match_strings(object):

    def setup(self):
        self.uniques = tm.makeStringIndex(1000).values
        self.all = self.uniques.repeat(10)

    def time_match_strings(self):
        match(self.all, self.uniques)


class misc_cache_readonly(object):

    def setup(self):


        class Foo:

            @cache_readonly
            def prop(self):
                return 5
        self.obj = Foo()

    def time_misc_cache_readonly(self):
        self.obj.prop