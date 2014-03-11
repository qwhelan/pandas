from vbench.benchmark import Benchmark
from datetime import datetime

common_setup = """from pandas_vb_common import *
from pandas.tseries.offsets import *
"""

#----------------------------------------------------------------------
# Series constructors

setup = common_setup + """
dt = DatetimeIndex(start='1/1/2000', freq='D', periods=100)
dt_m = DatetimeIndex(start='1/1/2000', freq='M', periods=100)
dt_h = DatetimeIndex(start='1/1/2000', freq='H', periods=100)
dt_m_list = list(dt_m)
td = np.timedelta64(1)

td = DatetimeIndex([np.timedelta64(1)] * 100)

periods = period_range('1/2000', freq='M', periods=100)
period_list = list(periods)

test_date = dt[0]

dt_nonunique = dt.append(DatetimeIndex([test_date]))

dt_off = dt + Second()

dt_localized = dt.tz_localize('America/New_York')

date_string_list = ['1/1/2000'] * 1000

index_date_string = Index(date_string_list)

index_range1 = Index(range(100))
index_range2 = Index(range(50, 150)[::-1])

index_empty = Index([])

multi_key1 = ['foo', 'bar']
multi_key2 = ['red', 'green']
multi_key3 = ['a', 'b']

mi = MultiIndex.from_product([multi_key1, multi_key2])
mi2 = MultiIndex.from_product([multi_key3, multi_key2])

mi_entry = mi2.values[-1]
"""

dtindex_add_integer = Benchmark("dt + 5", setup=setup)

dtindex_add_nptimedelta64 = Benchmark("dt + td", setup=setup)

dtindex_sub_integer = Benchmark("dt - 5", setup=setup)

dtindex_sub_nptimedelta64 = Benchmark("dt - td", setup=setup)


dtindex_isin = Benchmark("dt.isin([test_date])", setup=setup)

dtindex_get_duplicates = Benchmark("dt_nonunique.get_duplicates()", setup=setup)

dtindex_to_period = Benchmark("dt.to_period()", setup=setup)

dtindex_snap = Benchmark("dt_off.snap('D')", setup=setup)

dtindex_shift = Benchmark("dt.shift(1)", setup=setup)

dtindex_intersection = Benchmark("dt.intersection(dt_m_list)", setup=setup)

dtindex_resolution = Benchmark("dt.resolution", setup=setup)

dtindex_insert = Benchmark("dt.insert(-1, test_date)", setup=setup)

dtindex_delete = Benchmark("dt.delete(0)", setup=setup)

dtindex_tz_convert = Benchmark("dt.tz_convert('America/Los_Angeles')", setup=setup)

dtindex_indexer_at_time = Benchmark('dt_h.indexer_at_time("1AM")', setup=setup)

dtindex_indexer_between_time = Benchmark('dt_h.indexer_at_time("1AM", "4:30AM")', setup=setup)

dtindex_min = Benchmark("dt.min()", setup=setup)

dtindex_max = Benchmark("dt.max()", setup=setup)

dtindex_to_julian_date = Benchmark("dt.to_julian_date()", setup=setup)

dtindex_string = Benchmark("DatetimeIndex(date_string_list)", setup=setup)

dtindex_string = Benchmark("DatetimeIndex(date_string_list, tz='America/Los_Angeles')", setup=setup)


index_datetime_object = Benchmark("Index(dt, dtype='object')", setup=setup)

index_timedelta = Benchmark("Index(td)", setup=setup)

index_period_index = Benchmark("Index(periods)", setup=setup)

index_periods = Benchmark("Index(period_list)", setup=setup)

index_convert_to_datetime = Benchmark("index_date_string.to_datetime()", setup=setup)

index_order = Benchmark("index_date_string.order()", setup=setup)

index_and = Benchmark("index_range1 & index_range2", setup=setup)

index_or = Benchmark("index_range1 | index_range2", setup=setup)

index_xor = Benchmark("index_range1 ^ index_range2", setup=setup)



#join_single_multi

index_join_outer_other_empty = Benchmark("index_range1.join(index_empty, 'outer')", setup=setup)
index_join_inner_self_empty = Benchmark("index_empty.join(index_range1, 'inner')", setup=setup)

index_join_neither_unique = Benchmark("index_date_string.join(index_date_string)", setup=setup)

index_join_one_nonunique_both_monotonic = Benchmark("index_date_string.join(index_range1)", setup=setup)

index_join_one_nonunique = Benchmark("index_date_string.join(index_range2)", setup=setup)


multi_from_product = Benchmark("MultiIndex.from_product([multi_key1, multi_key2])", setup=setup)

multi_has_duplicates = Benchmark("mi.has_duplicates", setup=setup)

multi_get_level_values = Benchmark("mi.get_level_values(1)", setup=setup)

multi_to_hierarchical = Benchmark("mi.to_hierarchical(3)", setup=setup)

multi_swap_level = Benchmark("mi.swaplevel(1, 0)", setup=setup)

multi_reorder = Benchmark("mi.reorder_levels([1, 0])", setup=setup)

multi_level_equals = Benchmark("mi.equal_levels(mi)", setup=setup)

multi_union = Benchmark("mi.union(mi2)", setup=setup)

multi_intersection = Benchmark("mi.intersection(mi2)", setup=setup)

multi_diff = Benchmark("mi.diff(mi2)", setup=setup)

multi_insert = Benchmark("mi.insert(-1, mi_entry)", setup=setup)

multi_delete = Benchmark("mi.delete(0)", setup=setup)
