import sys
if sys.platform == 'win32':
    from pandas._libs.tslibs.tzlocal.win32 import get_localzone, reload_localzone
else:
    from pandas._libs.tslibs.tzlocal.unix import get_localzone, reload_localzone
