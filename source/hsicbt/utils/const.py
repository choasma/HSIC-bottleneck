from .misc import get_current_timestamp
import os

TTYPE_STANDARD = 'backprop'
TTYPE_HSICTRAIN = 'hsictrain'
TTYPE_FORMAT = 'format'
TTYPE_UNFORMAT = 'unformat'
TTYPE_PLOT = 'plot'

FONTSIZE_TITLE  = 60
FONTSIZE_XLABEL = 50
FONTSIZE_YLABEL = 50
FONTSIZE_XTICKS = 40
FONTSIZE_YTICKS = 40
FONTSIZE_LEDEND = 40
FONTSIZE_FOOTNOTE = 10

DEBUG_MODE = 0
TIMESTAMP = 'HSICBT_TIMESTAMP'
current_time_stamp = get_current_timestamp()
#if not os.environ.get(TIMESTAMP) or \
#   not os.environ.get(TIMESTAMP) == current_time_stamp:
#    os.environ[TIMESTAMP] = current_time_stamp
if not os.environ.get(TIMESTAMP):
    os.environ[TIMESTAMP] = current_time_stamp
TIMESTAMP_CODE = os.environ[TIMESTAMP]
