"""
Fix numpy NaN import for pandas_ta
"""
import numpy

# Add NaN to numpy module if it doesn't exist
if not hasattr(numpy, 'NaN'):
    numpy.NaN = numpy.nan
    print("Added NaN to numpy for pandas_ta compatibility") 