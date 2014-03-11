from vbench.benchmark import Benchmark
from datetime import datetime

SECTION = "Index / MultiIndex objects"


common_setup = """from pandas_vb_common import *
"""

#----------------------------------------------------------------------
# intersection, union

setup = common_setup + """
rng = DateRange('1/1/2000', periods=10000, offset=datetools.Minute())
if rng.dtype == object:
    rng = rng.view(Index)
else:
    rng = rng.asobject
rng2 = rng[:-1]
"""

index_datetime_intersection = Benchmark("rng.intersection(rng2)", setup)
index_datetime_union = Benchmark("rng.union(rng2)", setup)

setup = common_setup + """
rng = date_range('1/1/2000', periods=10000, freq='T')
rng2 = rng[:-1]
"""

datetime_index_intersection = Benchmark("rng.intersection(rng2)", setup,
                                        start_date=datetime(2013, 9, 27))

datetime_index_union = Benchmark("rng.union(rng2)", setup,
                                 start_date=datetime(2013, 9, 27))

datetime_index_diff = Benchmark('rng.diff(rng2)', setup,
                                start_date=datetime(2013, 9, 27))

datetime_index_sym_diff = Benchmark('rng.sym_diff(rng2)', setup,
                                start_date=datetime(2013, 9, 27))

# integers
setup = common_setup + """
N = 1000000
options = np.arange(N)

left = Index(options.take(np.random.permutation(N)[:N // 2]))
right = Index(options.take(np.random.permutation(N)[:N // 2]))
"""

index_int64_union = Benchmark('left.union(right)', setup,
                              start_date=datetime(2011, 1, 1),
                              logy=True)

index_int64_intersection = Benchmark('left.intersection(right)', setup,
                                     start_date=datetime(2011, 1, 1),
                                     logy=True)

index_int64_diff = Benchmark('left.diff(right)', setup,
                             start_date=datetime(2011, 1, 1))

index_int64_sym_diff = Benchmark('left.sym_diff(right)', setup,
                             start_date=datetime(2011, 1, 1))


# mixed dtypes
setup = common_setup + """
N = 10000
options = np.arange(N)

left = Index(options.take(np.random.permutation(N)[:N // 2]))
right = Index(options.take(np.random.permutation(N)[:N // 2]), dtype='object')
"""

index_mixed_dtype_union = Benchmark('left.union(right)', setup,
                              start_date=datetime(2011, 1, 1))

index_mixed_dtype_intersection = Benchmark('left.intersection(right)', setup,
                                     start_date=datetime(2011, 1, 1))

index_mixed_dtype_diff = Benchmark('left.diff(right)', setup,
                                     start_date=datetime(2011, 1, 1))

index_mixed_dtype_sym_diff = Benchmark('left.sym_diff(right)', setup,
                                     start_date=datetime(2011, 1, 1))

index_mixed_dtype_join = Benchmark('left.join(right)', setup,
                                     start_date=datetime(2011, 1, 1))
