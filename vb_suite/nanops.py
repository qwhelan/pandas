from vbench.benchmark import Benchmark
from datetime import datetime

common_setup = """from pandas_vb_common import *
"""

#----------------------------------------------------------------------
# Series constructors

setup = common_setup + """
s = Series(np.random.randn(100))
s.ix[::2] = float('nan')
"""

nanops_skew = Benchmark("s.skew()", setup=setup)

nanops_median = Benchmark("s.median()", setup=setup)

nanops_min = Benchmark("s.min()", setup=setup)

nanops_max = Benchmark("s.max()", setup=setup)

nanops_argmin = Benchmark("s.argmin()", setup=setup)

nanops_kurt = Benchmark("s.kurt()", setup=setup)

nanops_prod = Benchmark("s.prod()", setup=setup)

nanops_corr = Benchmark("s.corr(s)", setup=setup)

nanops_cov = Benchmark("s.cov(s)", setup=setup)

nanops_comp = Benchmark("s == s", setup=setup)
