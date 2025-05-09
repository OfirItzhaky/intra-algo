"""
This file patches the pandas_ta package to work with newer versions of numpy
by adding a NaN attribute to numpy if it doesn't exist.
"""
import numpy

# Add NaN to numpy if it doesn't exist (it was renamed to lowercase nan)
if not hasattr(numpy, 'NaN'):
    numpy.NaN = numpy.nan
    print("✅ Added NaN to numpy for pandas_ta compatibility") 