#####################################################
# @Author: Abhilash Sarwade
# @Date:   2023-04-28 09:31:35
# @email: sarwade@ursc.gov.in
# @File Name: L0_interm.py
# @Project: solexs_pipeline
#
# @Last Modified time: 2023-08-01 10:47:26 am
#####################################################

from .binary_read import read_solexs_binary_data
import os
from .logging import setup_logger
import importlib.util
import sys
import numpy as np
from .fits_utils import PHAII_INTERM, HOUSEKEEPING, LC_INTERM

## Importing solexs_caldbgen
curr_dir = os.path.dirname(__file__)
caldbgen_fl = f'{curr_dir}/CALDB/aditya-l1/solexs/software/solexs_caldbgen/solexs_caldbgen/__init__.py' #TODO pkg_resources.
caldbgen_ml_name = 'solexs_caldbgen'
caldbgen_spec = importlib.util.spec_from_file_location(caldbgen_ml_name,caldbgen_fl)
solexs_caldbgen = importlib.util.module_from_spec(caldbgen_spec)
sys.modules[caldbgen_ml_name] = solexs_caldbgen
caldbgen_spec.loader.exec_module(solexs_caldbgen)


"""
PACQ files to 
1. Spice parameters
2. HK parameters from instrument data
3. HK parameters from TM
4. Raw Spectrum
5. Time Correlation Table
6. Raw Lightcurve
"""

log = setup_logger(f'solexs_pipeline.{__name__}')
BCF_DIR = f'{curr_dir}/CALDB/aditya-l1/solexs/data/bcf'



""" TODO
1. Find gain and offset wrt temperatures (any other CALDB parameter?)
2. Generate seperate PHAII in raw pha files for each 1 second spectrum with energy bins as per temperatures
3. Generate raw lc files (channel in channel space/energy space? How to find energy?)
4. Add caldb for calculating gain and offset as function of SDD temperature and electronics box temperature
"""



class intermediate_directory():
    # input_file: Path to solexs binary data file
    def __init__(self, input_file, output_dir=None, clobber=True) -> None:
        self.input_file = input_file
        self.input_filename = os.path.basename(input_file)
        #self.make_interm_dir(self.input_filename,output_dir,clobber)

        self.solexs_bd = read_solexs_binary_data(input_file)
        self.n_SDD1 = len(self.solexs_bd.SDD1.hdr_data.gain)
        self.n_SDD2 = len(self.solexs_bd.SDD2.hdr_data.gain)

        self.read_lbt_hk_data()

        #self.pha_file_SDD1 = self.pha_interm_file(1)
        #self.pha_file_SDD2 = self.pha_interm_file(2)

    
    def make_interm_dir(self,input_filename,output_dir=None,clobber=True):
        if output_dir is None:
            output_dir = os.path.curdir
        
        output_dir = os.path.join(output_dir,f'{input_filename}_interm') #TODO remove extention from inputfilename if required
        if os.path.exists(output_dir) and clobber: #TODO add log
            os.removedirs(output_dir)
        os.makedirs(output_dir)
        sdd1_output_dir = os.path.join(output_dir,'SDD1')
        sdd2_output_dir = os.path.join(output_dir,'SDD2')

        os.makedirs(sdd1_output_dir)
        os.makedirs(sdd2_output_dir)

    def load_bcf_caldb(self,SDD_number):
        SDD_no = str(SDD_number)
        GAIN_FILE = f'{BCF_DIR}/gain/gain_SDD{SDD_no}.txt'
        # THICKNESS_FILE = f'{BCF_DIR}/arf/material_thickness.dat'
        # AREA_FILE = f'{BCF_DIR}/aperture_size/SDD{SDD_no}'
        ELE_SIGMA_FILE = f'{BCF_DIR}/electronic_noise/electronic_noise.txt'
        FANO_FILE = f'{BCF_DIR}/electronic_noise/fano.txt'
        OFFSET_FILE = f'{BCF_DIR}/offset/offset_SDD{SDD_no}.txt'


        gain_data = np.loadtxt(GAIN_FILE,usecols=[1,2])
        offset_data = np.loadtxt(OFFSET_FILE)

        gain = gain_data[0,0]
        # offset = gain_data[1,0]

        # thickness = np.loadtxt(THICKNESS_FILE,usecols=[1,2])

        # area = np.loadtxt(AREA_FILE)

        electronic_sigma_data = np.loadtxt(ELE_SIGMA_FILE)*1#0.050 #0.15212#0.050 #keV
        electronic_sigma = electronic_sigma_data[0]
        fano = np.loadtxt(FANO_FILE)*1#0.114

        return gain, offset_data, electronic_sigma, fano
    
    def read_lbt_hk_data(self,lbt_hk_data_file=None): #TODO temporary. assuming electronics box temperature is 0 degrees
        self.ele_box_temp_SDD1 = np.zeros(self.n_SDD1)
        self.ele_box_temp_SDD2 = np.zeros(self.n_SDD2)
    
    def calc_energy_bins(self,SDD_number):
        gain, offset_data, electronic_sigma, fano = self.load_bcf_caldb(SDD_number=SDD_number)
        sdd_data = getattr(self.solexs_bd,f'SDD{SDD_number}')
        gain_f = gain/sdd_data.hdr_data.gain
        ele_box_temp_tmp = getattr(self,f'ele_box_temp_SDD{SDD_number}')
        offset_f = offset_data[0]*ele_box_temp_tmp**2 + \
            offset_data[1]*ele_box_temp_tmp + offset_data[2]
        
        energy_bins_mat = np.zeros((getattr(self,f'n_SDD{SDD_number}'),340,2))

        for i in range(getattr(self, f'n_SDD{SDD_number}')):
            energy_bins_mat[i, :, :] = solexs_caldbgen.calc_ene_bins_out(
                gain_f[i], offset_f[i])[0]
        
        return energy_bins_mat
    
    def get_start_time(self,SDD_number):
        """
        Output: In seconds from standard epoch

        Use TCT file to get UT in seconds
        Temporary: using instrument time
        """
        
        sdd_data = getattr(self.solexs_bd,f'SDD{SDD_number}')
        st_time = sdd_data.hdr_data.ref_count/1000 #in seconds

        return st_time
    
    def pha_interm_file(self,SDD_number):
        sdd_data = getattr(self.solexs_bd,f'SDD{SDD_number}')
        st_time = self.get_start_time(SDD_number=SDD_number)
        
        n_SDD = getattr(self,f'n_SDD{SDD_number}')
        ch = np.arange(340)
        channel = np.tile(ch,(n_SDD,1))
        telapse = np.ones(n_SDD) #assuming integration time is 1 second each
        counts = sdd_data.spectral_data.spectra.T
        quality = np.zeros(n_SDD)
        exposure = np.ones(n_SDD)
        energy_bin_mat = self.calc_energy_bins(SDD_number=SDD_number)
        e_min = energy_bin_mat[:,:,0]
        e_max = energy_bin_mat[:,:,1]
        
        pha_file = PHAII_INTERM(f'{self.input_filename}_SDD{SDD_number}',st_time,telapse,channel,counts,quality,exposure,e_min,e_max)
        
        return pha_file
    
    def hk_interm_file(self,SDD_number):
        sdd_data = getattr(self.solexs_bd, f'SDD{SDD_number}')
        hk_dict = sdd_data.hdr_data.__dict__
        hk_colnames = []
        hk_arr = []

        for colname in hk_dict.keys():
            hk_colnames.append(colname)
            hk_arr.append(hk_dict[colname])
        
        hk_arr = np.array(hk_arr)
        hk_colnames = np.array(hk_colnames)

        # Add LBT HK Data 
        ele_box_temp_tmp = getattr(self,f'ele_box_temp_SDD{SDD_number}')
        # hk_arr = np.vstack((hk_arr,ele_box_temp_tmp))
        hk_dict['Electronic Box Temperature'] = ele_box_temp_tmp
        hk_colnames = np.append(hk_colnames,'Electronic Box Temperature')

        hk_file = HOUSEKEEPING(f'{self.input_filename}_SDD{SDD_number}',hk_dict,hk_colnames) #TODO Some problem with data type of ref counter

        return hk_file
    
    def lc_interm_file(self,SDD_number):
        sdd_data = getattr(self.solexs_bd,f'SDD{SDD_number}')
        tm = sdd_data.hdr_data.ref_count/1000
        rate1 = sdd_data.temporal_data.low_sec
        rate2 = sdd_data.temporal_data.med_sec
        rate3 = sdd_data.temporal_data.high_sec
        rate_all = rate1 + rate2 + rate3
        # rate = np.vstack((rate1, rate2, rate3, rate_all)).T
        lower_thresh = sdd_data.hdr_data.timing_channel_thresh_lower
        higher_thresh = sdd_data.hdr_data.timing_channel_thresh_higher
        lc_file = LC_INTERM(f'{self.input_filename}_SDD{SDD_number}',tm,rate1,rate2,rate3,rate_all, lower_thresh, higher_thresh)

        return lc_file