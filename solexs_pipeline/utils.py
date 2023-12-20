#####################################################
# @Author: Abhilash Sarwade
# @Date:   2023-12-08 02:00:44 pm
# @email: sarwade@ursc.gov.in
# @File Name: utils.py
# @Project: solexs_pipeline
#
# @Last Modified time: 2023-12-20 09:12:58 am
#####################################################

import numpy as np
from .binary_read import read_solexs_binary_data
import tempfile
import os
import json
import datetime


def rebin_lc(lc, rebin_sec):
    extra_bins = len(lc) % rebin_sec
    if extra_bins != 0:
        lc = lc[:-extra_bins]
    new_bins = int(len(lc)/rebin_sec)
    new_lc = lc.reshape((new_bins, rebin_sec)).sum(axis=1)
    return new_lc


def generate_spectrogram(spectra, rebin_sec):
    extra_bins = spectra.shape[1] % rebin_sec
    if extra_bins != 0:
        spectra = spectra[:, :-extra_bins]
    new_bins = int(spectra.shape[1]/rebin_sec)
    new_spectra = spectra.reshape(340, new_bins, rebin_sec).sum(axis=2)
    new_tm = np.arange(new_bins)*rebin_sec
    return new_spectra, new_tm


def read_solexs_binary_data_multi(input_files, data_type='L0'):
    temp = tempfile.NamedTemporaryFile(dir='.',delete=False)
    for fl in input_files:
        fp = open(fl, 'rb')
        lns = fp.read()
        temp.write(lns)
    d = read_solexs_binary_data(temp.name, 'L0')
    temp.close()
    os.unlink(temp.name)
    return d


def read_goes_data(goes_json_file,wvl='long'):
    fp = open(goes_json_file, 'r')
    gd = json.load(fp)
    fp.close()

    g_time = []
    g_timestamp = []
    g_flx = []

    if wvl=='long':
        oddeven = 0
    elif wvl=='short':
        oddeven = 1

    for i, gdi in enumerate(gd):
        if i % 2 == oddeven:
            continue

        g_time.append(datetime.datetime.strptime(
            gdi['time_tag'], '%Y-%m-%dT%H:%M:%SZ'))
        
        g_timestamp.append(g_time[-1].timestamp())
        g_flx.append(gdi['observed_flux'])

    g_timestamp = np.array(g_timestamp)
    g_flx = np.array(g_flx)

    return g_time, g_timestamp, g_flx


def datetime2timestamp(datetime_arr):
    timestamp_arr = []
    for da in datetime_arr:
        timestamp_arr.append(da.timestamp())

    timestamp_arr = np.array(timestamp_arr)
    return timestamp_arr

def timestamp2datetime(timestamp_arr):
    datetime_arr = []
    for ta in timestamp_arr:
        datetime_arr.append(datetime.datetime.fromtimestamp(ta))

    return datetime_arr
