from sys import gettrace

DEBUG = False
if gettrace() is not None:
    DEBUG = True
