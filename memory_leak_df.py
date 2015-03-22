import sys
sys.path.insert(-1, '/usr/local/Cellar/glib/2.42.0/share/glib-2.0/gdb/')
import gc
#gc.set_debug(gc.DEBUG_LEAK)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas
import pandas as pd
#from memory_profiler import profile
#import tracemalloc
#tracemalloc.start()

#pdindex = pd.date_range(start='01/01/2013', freq='15min', end='01/01/2019')
#df = pd.DataFrame({'test':np.random.normal(0,1,len(pdindex))}, index=pdindex)

gc_count = {}
size_count = {}

def attribute(x):
    result = 0
    if isinstance(x, np.ndarray):
        result = x.nbytes
    elif isinstance(x, pd.DataFrame):
        result = x.values.nbytes + x.columns.nbytes + x.index.nbytes

    if result > 100000:
        print result, x

    return result

def mem_df(i):
    mi = pd.MultiIndex.from_tuples([(x, x) for x in range(10000)])
    df = pd.DataFrame(np.random.randn(10000, 10), index=mi)
    df.ix[100]
    gc.collect()

def print_info(i):
    print('*******************************')
    print('i : ' + str(i))
    gc_count[i] = len(gc.get_objects())

    size_count[i] = sum([attribute(x) for x in gc.get_objects()])

    print(gc_count[i])
    print(size_count[i])

def no_loop(i=1, pandas=True):
    mem_df(i)
#    gc.collect(2)

#@profile
def foo(n=5, pandas=True):
    print_info(0)
    for i in range(1, n+1):
        no_loop(i, pandas=pandas)
        print_info(i)

    print(pd.DataFrame({'gc.get_objects': gc_count, 'sys.getsizeof': size_count}))

    import objgraph
    objgraph.show_backrefs([x for x in gc.get_objects() if isinstance(x, pd.core.base.PandasObject)], max_depth=7)

df = pd.DataFrame(np.random.randn(10000))

def f(df):
    for dfi in np.array_split(df, 100):
        shuffle = dfi.reindex(np.random.permutation(dfi.index))
        one = shuffle.iloc[:50]
        two = shuffle.iloc[50:]
    import objgraph
    objgraph.show_backrefs([x for x in gc.get_objects() if isinstance(x, pd.core.base.PandasObject)], max_depth=7)

if __name__ == '__main__':
    f(df)
    #foo(10, True)
