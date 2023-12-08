#####################################################
# @Author: Abhilash Sarwade
# @Date:   2023-12-08 02:00:44 pm
# @email: sarwade@ursc.gov.in
# @File Name: utils.py
# @Project: solexs_pipeline
#
# @Last Modified time: 2023-12-08 02:00:57 pm
#####################################################

import numpy as np

def rebin_lc(lc, rebin_sec):
    extra_bins = len(lc) % rebin_sec
    if extra_bins != 0:
        lc = lc[:-extra_bins]
    new_bins = int(len(lc)/rebin_sec)
    new_lc = lc.reshape((new_bins, rebin_sec)).sum(axis=1)
    return new_lc


def generate_spectrogram(spectra,rebin_sec):
    extra_bins = spectra.shape[1] % rebin_sec
    if extra_bins != 0:
        spectra = spectra[:,:-extra_bins]
    new_bins = int(spectra.shape[1]/rebin_sec)
    new_spectra = spectra.reshape(340,new_bins,rebin_sec).sum(axis=2)
    new_tm = np.arange(new_bins)*rebin_sec
    return new_spectra,new_tm