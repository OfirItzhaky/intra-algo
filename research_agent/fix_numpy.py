"""
Fix numpy NaN import for pandas_ta
"""
import numpy
from research_agent.logging_setup import get_logger

log = get_logger(__name__)
# Add NaN to numpy module if it doesn't exist
if not hasattr(numpy, 'NaN'):
    numpy.NaN = numpy.nan
    log.info("Added NaN to numpy for pandas_ta compatibility") 