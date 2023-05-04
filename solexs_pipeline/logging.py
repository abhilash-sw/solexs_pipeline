#####################################################
# @Author: Abhilash Sarwade
# @Date:   2023-04-28 10:58:50
# @email: sarwade@ursc.gov.in
# @File Name: logging.py
# @Project: solexs_pipeline
#
# @Last Modified time: 2023-04-28 11:55:57
#####################################################

import logging
import logging.handlers as handlers
import sys


#3ML.io.logging

def setup_logger(name,log_filename):

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
    # log.addHandler(fh)
    log.addHandler(ch)

    # we do not want to duplicate the messages in the parents
    #log.propagate = False

    return log