#####################################################
# @Author: Abhilash Sarwade
# @Date:   2023-12-08 02:00:44 pm
# @email: sarwade@ursc.gov.in
# @File Name: utils.py
# @Project: solexs_pipeline
#
# @Last Modified time: 2023-12-19 08:58:54 am
#####################################################

import numpy as np
from .binary_read import read_solexs_binary_data
import tempfile
import os


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
    # os.unlink(temp.name)
    return d
