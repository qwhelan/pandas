from pandas_vb_common import *
import pandas.computation.expressions as expr
import pandas as pd


class eval_frame_add_all_threads(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))
        self.df3 = DataFrame(np.random.randn(20000, 100))
        self.df4 = DataFrame(np.random.randn(20000, 100))

    def time_eval_frame_add_all_threads(self):
        pd.eval('self.df + self.df2 + self.df3 + self.df4')


class eval_frame_add_one_thread(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))
        self.df3 = DataFrame(np.random.randn(20000, 100))
        self.df4 = DataFrame(np.random.randn(20000, 100))
        expr.set_numexpr_threads(1)

    def time_eval_frame_add_one_thread(self):
        pd.eval('self.df + self.df2 + self.df3 + self.df4')


class eval_frame_add_python(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))
        self.df3 = DataFrame(np.random.randn(20000, 100))
        self.df4 = DataFrame(np.random.randn(20000, 100))

    def time_eval_frame_add_python(self):
        pd.eval('self.df + self.df2 + self.df3 + self.df4', engine='python')


class eval_frame_add_python_one_thread(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))
        self.df3 = DataFrame(np.random.randn(20000, 100))
        self.df4 = DataFrame(np.random.randn(20000, 100))
        expr.set_numexpr_threads(1)

    def time_eval_frame_add_python_one_thread(self):
        pd.eval('self.df + self.df2 + self.df3 + self.df4', engine='python')


class eval_frame_and_all_threads(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))
        self.df3 = DataFrame(np.random.randn(20000, 100))
        self.df4 = DataFrame(np.random.randn(20000, 100))

    def time_eval_frame_and_all_threads(self):
        pd.eval('(self.df > 0) & (self.df2 > 0) & (self.df3 > 0) & (self.df4 > 0)')


class eval_frame_and_python_one_thread(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))
        self.df3 = DataFrame(np.random.randn(20000, 100))
        self.df4 = DataFrame(np.random.randn(20000, 100))
        expr.set_numexpr_threads(1)

    def time_eval_frame_and_python_one_thread(self):
        pd.eval('(self.df > 0) & (self.df2 > 0) & (self.df3 > 0) & (self.df4 > 0)', engine='python')


class eval_frame_and_python(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))
        self.df3 = DataFrame(np.random.randn(20000, 100))
        self.df4 = DataFrame(np.random.randn(20000, 100))

    def time_eval_frame_and_python(self):
        pd.eval('(self.df > 0) & (self.df2 > 0) & (self.df3 > 0) & (self.df4 > 0)', engine='python')


class eval_frame_chained_cmp_all_threads(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))
        self.df3 = DataFrame(np.random.randn(20000, 100))
        self.df4 = DataFrame(np.random.randn(20000, 100))

    def time_eval_frame_chained_cmp_all_threads(self):
        pd.eval('self.df < self.df2 < self.df3 < self.df4')


class eval_frame_chained_cmp_python_one_thread(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))
        self.df3 = DataFrame(np.random.randn(20000, 100))
        self.df4 = DataFrame(np.random.randn(20000, 100))
        expr.set_numexpr_threads(1)

    def time_eval_frame_chained_cmp_python_one_thread(self):
        pd.eval('self.df < self.df2 < self.df3 < self.df4', engine='python')


class eval_frame_chained_cmp_python(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))
        self.df3 = DataFrame(np.random.randn(20000, 100))
        self.df4 = DataFrame(np.random.randn(20000, 100))

    def time_eval_frame_chained_cmp_python(self):
        pd.eval('self.df < self.df2 < self.df3 < self.df4', engine='python')


class eval_frame_mult_all_threads(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))
        self.df3 = DataFrame(np.random.randn(20000, 100))
        self.df4 = DataFrame(np.random.randn(20000, 100))

    def time_eval_frame_mult_all_threads(self):
        pd.eval('self.df * self.df2 * self.df3 * self.df4')


class eval_frame_mult_one_thread(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))
        self.df3 = DataFrame(np.random.randn(20000, 100))
        self.df4 = DataFrame(np.random.randn(20000, 100))
        expr.set_numexpr_threads(1)

    def time_eval_frame_mult_one_thread(self):
        pd.eval('self.df * self.df2 * self.df3 * self.df4')


class eval_frame_mult_python(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))
        self.df3 = DataFrame(np.random.randn(20000, 100))
        self.df4 = DataFrame(np.random.randn(20000, 100))

    def time_eval_frame_mult_python(self):
        pd.eval('self.df * self.df2 * self.df3 * self.df4', engine='python')


class eval_frame_mult_python_one_thread(object):

    def setup(self):
        self.df = DataFrame(np.random.randn(20000, 100))
        self.df2 = DataFrame(np.random.randn(20000, 100))
        self.df3 = DataFrame(np.random.randn(20000, 100))
        self.df4 = DataFrame(np.random.randn(20000, 100))
        expr.set_numexpr_threads(1)

    def time_eval_frame_mult_python_one_thread(self):
        pd.eval('self.df * self.df2 * self.df3 * self.df4', engine='python')


class query_datetime_index(object):

    def setup(self):
        self.N = 1000000
        self.halfway = ((self.N // 2) - 1)
        self.index = date_range('20010101', periods=self.N, freq='T')
        self.s = Series(self.index)
        self.ts = self.s.iloc[self.halfway]
        self.df = DataFrame({'a': np.random.randn(self.N), }, index=self.index)

    def time_query_datetime_index(self):
        self.df.query('index < @self.ts')


class query_datetime_series(object):

    def setup(self):
        self.N = 1000000
        self.halfway = ((self.N // 2) - 1)
        self.index = date_range('20010101', periods=self.N, freq='T')
        self.s = Series(self.index)
        self.ts = self.s.iloc[self.halfway]
        self.df = DataFrame({'dates': self.s.values, })

    def time_query_datetime_series(self):
        self.df.query('dates < @self.ts')


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
        self.df.query('(a >= @self.min_val) & (a <= @self.max_val)')
