#####################################################
# @Author: Abhilash Sarwade
# @Date:   2023-04-28 10:58:50
# @email: sarwade@ursc.gov.in
# @File Name: logging.py
# @Project: solexs_pipeline
#
# @Last Modified time: 2023-08-26 08:26:56 am
#####################################################

import logging
import logging.handlers as handlers
import sys


#3ML.io.logging

"""TODO
check logging config files
"""

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def setup_logger(name,log_filename=None):

    # A logger with name name will be created
    # and then add it to the print stream
    log = logging.getLogger(name)

    # this must be set to allow debug messages through
    log.setLevel(logging.DEBUG)

    # add the handlers
    # fh = logging.FileHandler(log_filename)
    # fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    # log.addHandler(fh)
    log.addHandler(ch)

    # we do not want to duplicate the messages in the parents
    log.propagate = False

    return log

def create_fileHandler(log_filename):
    fh = logging.FileHandler(log_filename)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    return fh