import sys
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

pdindex = pd.date_range(start='01/01/2013', freq='15min', end='01/01/2019')
df = pd.DataFrame({'test':np.random.normal(0,1,len(pdindex))}, index=pdindex)

gc_count = {}
size_count = {}

def memplot_plot(df, i, pandas=True):
    if pandas:
        df.test.plot()
    else:
        plt.plot(df.index, df.test)
    plt.close('all')

def print_info(i):
    print('*******************************')
    print('i : ' + str(i))
    gc_count[i] = len(gc.get_objects())
    size_count[i] = sys.getsizeof(gc.get_objects())
    print(gc_count[i])
    print(size_count[i])

def no_loop(i=1, pandas=True):
    memplot_plot(df, i, pandas=pandas)
    gc.collect(2)

#@profile
def foo(n=5, pandas=True):
    print_info(0)
    for i in range(1, n+1):
        no_loop(i, pandas=pandas)
        print_info(i)

    print(pd.DataFrame({'gc.get_objects': gc_count, 'sys.getsizeof': size_count}))

    ps = [x for x in gc.get_objects() if isinstance(x, pd.Period)]

    print(len(df), len(ps))

if __name__ == '__main__':
    foo(10, True)
