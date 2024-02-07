#####################################################
# @Author: Abhilash Sarwade
# @Date:   2023-12-08 02:00:44 pm
# @email: sarwade@ursc.gov.in
# @File Name: utils.py
# @Project: solexs_pipeline
#
# @Last Modified time: 2024-02-02 08:46:12 am
#####################################################

import numpy as np
from .binary_read import read_solexs_binary_data
import tempfile
import os
import json
import datetime
import glob
from astropy.io import fits
from .fits_utils import PHAII

# def rebin_lc(lc, rebin_sec):
#     extra_bins = len(lc) % rebin_sec
#     if extra_bins != 0:
#         lc = lc[:-extra_bins]
#     new_bins = int(len(lc)/rebin_sec)
#     new_lc = lc.reshape((new_bins, rebin_sec)).sum(axis=1)
#     return new_lc


def rebin_lc(lc, datetime_arr ,rebin_sec): #lc: counts per second
    extra_bins = len(lc) % rebin_sec
    if extra_bins != 0:
        lc = lc[:-extra_bins]
    new_bins = int(len(lc)/rebin_sec)
    new_lc = lc.reshape((new_bins, rebin_sec)).sum(axis=1)
    new_tm = np.arange(new_bins)*rebin_sec


    new_datetime_arr = []
    for ii in new_tm:
        new_datetime_arr.append(datetime_arr[int(ii)])

    new_lc = new_lc/rebin_sec #counts per second

    return new_lc, new_datetime_arr



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

def make_start_stop_time_db(solexs_data_dir):
    files = glob.glob(os.path.join(solexs_data_dir,'*','*','*','*','*pld')) #2024/01/01/SLX*/*pld
    files.sort()
    tmp_files = glob.glob(os.path.join(solexs_data_dir,'*','*','*','*pld')) #2024/01/01/*pld
    tmp_files.sort()

    files = files + tmp_files

    #reading already existing files
    fid = open(os.path.join(solexs_data_dir,'files_start_stop_times.txt'),'r')
    start_stop_data = fid.readlines()
    fid.close()
    file_names_old = []

    for i in range(len(start_stop_data)):
        file_names_old.append(start_stop_data[i].split('\t')[1])

    start_times = []
    stop_times = []
    dates = []
    files_new = []

    for fl in files:
        if os.path.basename(fl) in file_names_old:
            continue
        tmp_d = read_solexs_binary_data(fl,'L0')
        start_times.append(tmp_d.pld_header_SDD1.pld_utc_datetime[0].isoformat())
        stop_times.append(tmp_d.pld_header_SDD2.pld_utc_datetime[-1].isoformat())
        dates.append(tmp_d.pld_header_SDD1.pld_utc_datetime[0].strftime('%Y%m%d'))
        files_new.append(fl)
    
    fid1 = open(os.path.join(solexs_data_dir,'files_start_stop_times.txt'),'a')
    for i, fl in enumerate(files_new):
        tmp_line = f'{fl}\t{os.path.basename(fl)}\t{start_times[i]}\t{stop_times[i]}\t{dates[i]}\n'
        fid1.write(tmp_line)

    fid1.close()

    fid = open(os.path.join(solexs_data_dir,'files_start_stop_times.txt'),'r')
    start_stop_data = fid.readlines()
    fid.close()

    datewise_files_dict = {}

    for dt in dates:
        tmp_datewise_files = []
        for i in range(len(start_stop_data)):
            tmp_data = start_stop_data[i].split('\t')
            if tmp_data[-1][:-1] == dt:
                tmp_datewise_files.append(tmp_data[0])
        datewise_files_dict[dt] = tmp_datewise_files


def run_pipeline(pld_files_list,output_dir,sdd='12'):
    cmd = 'solexs_pipeline'
    for pld_file in pld_files_list:
        cmd = cmd + f' -i {pld_file}'
    
    cmd = cmd + f' -o {output_dir} -dt L0 -sdd {sdd}'
    os.system(cmd)

def run_pipeline_batch(datewise_files_dict,output_dir,sdd='12'):
    for dt in datewise_files_dict.keys():
        if os.path.isdir(f'AL1_SOLEXS_{dt}'):
            continue
        run_pipeline(datewise_files_dict[dt],output_dir,sdd)

# def make_datewise_files_db(solexs_data_dir):
#     fid = open(os.path.join(solexs_data_dir,'files_start_stop_times.txt'),'r')
#     start_stop_data = fid.readlines()

#     dates = []
#     files = []

#     for i in range(len(start_stop_data)):

def convert_pi_340_to_512(pi_file,rsp_file):
    hdus_pi = fits.open(pi_file)

    hdus_rsp = fits.open(rsp_file)
    rsp_ene_min = hdus_rsp[1].data['E_MIN']
    rsp_ene_max = hdus_rsp[1].data['E_MAX']


    ch = hdus_pi[1].data['CHANNEL'][0]

    ###
    pi_ene_min = np.zeros(len(ch))
    pi_ene_min[:169] = rsp_ene_min[:169] # Correct (double checked)

    j = 170
    for i in range(169,340):
        pi_ene_min[i] = rsp_ene_min[j]
        j = j+2
    
    pi_ene_max = np.zeros(len(ch))
    pi_ene_max[:168] = rsp_ene_max[:168]

    j = 169
    for i in range(168,340):
        pi_ene_max[i] = rsp_ene_max[j]
        j = j+2
    
    counts = hdus_pi[1].data['COUNTS']

    nbins_pi = counts.shape[0]
    counts_512 = np.zeros((nbins_pi,512))

    for i in range(nbins_pi):
        count = counts[i]
        count[168:] = count[168:]/2
        counts_512[i] = np.interp(rsp_ene_min,pi_ene_min,count)


    

    filename = hdus_pi[0].header['FILENAME'].split('.')[0] + '_512.pi'
    all_time = hdus_pi[1].data['TSTART']
    telapse = hdus_pi[1].data['TELAPSE']
    ch = np.arange(512)
    channel = np.tile(ch, (nbins_pi, 1))
    exposure = hdus_pi[1].data['EXPOSURE']
    respfile = hdus_pi[1].data['RESPFILE']

    pi_file_512 = PHAII(filename, all_time, telapse, channel, counts_512, exposure, respfile)
    pi_file_512.writeto(filename)


def solexs_genspec(pi_file_512,tstart,tstop,outfile): # times in seconds since 2017,1,1
    spec_file = pi_file_512
    hdu1 = fits.open(spec_file)
    hdu=fits.BinTableHDU.from_columns(hdu1[1].columns)

    data=hdu.data

    time_solexs = data['TSTART']
    # tbinsize=(data['TSTOP'][0]-data['TSTART'][0])
    
    exposure=data['EXPOSURE']

    inds = (time_solexs >= tstart) & (time_solexs < tstop)

    data_f = data[inds]

    channel = data_f[0][3]
    n_ch = len(channel)
    spec_data = np.zeros(n_ch)
    stat_err = np.zeros(n_ch)
    sys_err = np.zeros(n_ch)
    exposure = 0

    for di in data_f:
        spec_data = spec_data + di[4]
        # stat_err = stat_err + np.sqrt(di[4])
        # sys_err = sys_err
        exposure = exposure + di[5]

    stat_err = np.sqrt(spec_data)
    
    # writing file
    hdu_list = []
    primary_hdu = fits.PrimaryHDU()
                                    
    hdu_list.append(primary_hdu)

    fits_columns = []
    col1 = fits.Column(name='CHANNEL',format='J',array=channel)
    col2 = fits.Column(name='COUNTS',format='E',array=spec_data)
    col3 = fits.Column(name='STAT_ERR',format='E',array=stat_err)
    col4 = fits.Column(name='SYS_ERR',format='E',array=sys_err)

    fits_columns.append(col1)
    fits_columns.append(col2)
    fits_columns.append(col3)
    fits_columns.append(col4)

    hdu_pha = fits.BinTableHDU.from_columns(fits.ColDefs(fits_columns))
    hdu_pha.name = 'SPECTRUM'
                                                                       
    hdu_list.append(hdu_pha)
                                                                       
    _hdu_list = fits.HDUList(hdus=hdu_list)

    tstart_dt = datetime.datetime.fromtimestamp(tstart)
    tstop_dt = datetime.datetime.fromtimestamp(tstop)

    _hdu_list[1].header.set('TSTART',tstart_dt.isoformat())
    _hdu_list[1].header.set('TSTOP',tstop_dt.isoformat())
    _hdu_list[1].header.set('EXPOSURE',f'{exposure:.2f}')

    _hdu_list[0].header.set('TELESCOP','AL1')
    _hdu_list[0].header.set('INSTRUME','SoLEXS')
    _hdu_list.writeto(f'{outfile}.pha')
