#####################################################
# @Author: Abhilash Sarwade
# @Date:   2023-04-28 09:31:35
# @email: sarwade@ursc.gov.in
# @File Name: L0_interm.py
# @Project: solexs_pipeline
#
# @Last Modified time: 2023-05-09 02:35:46
#####################################################

from .binary_read import read_solexs_binary_data
import os
from .logging import setup_logger
import importlib.util
import numpy as np

## Importing solexs_caldbgen
caldbgen_fl = 'CALDB/aditya-l1/solexs/software/solexs_caldbgen/solexs_caldbgen/__init__.py'
caldbgen_ml_name = 'solexs_caldbgen'
caldbgen_spec = importlib.util.spec_from_file_location(caldbgen_ml_name,caldbgen_fl)
solexs_caldbgen = importlib.util.module_from_spec(caldbgen_spec)
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
BCF_DIR = 'CALDB/aditya-l1/solexs/data/bcf'



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
        input_filename = os.path.basename(input_file)
        self.make_interm_dir(input_filename,output_dir,clobber)

        self.solexs_bd = read_solexs_binary_data(input_file)
        self.n_SDD1 = len(self.solexs_bd.SDD1.hdr_data.gain)
        self.n_SDD2 = len(self.solexs_bd.SDD2.hdr_data.gain)

        self.energy_bins_mat_SDD1 = self.calc_energy_bins(SDD_number=1)
        self.energy_bins_mat_SDD2 = self.calc_energy_bins(SDD_number=2)
    
    def make_interm_dir(self,input_filename,output_dir=None,clobber=True):
        if output_dir is None:
            output_dir = os.path.curdir()
        
        output_dir = os.path.join(output_dir,input_filename) #TODO remove extention from inputfilename if required
        if os.path.exists(output_dir) and clobber: #TODO add log
            os.removedirs(output_dir)
        os.makedirs(output_dir)

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
        gain_f = gain*sdd_data.hdr_data.gain
        offset_f = offset_data[0]*self.ele_box_temp_SDD1**2 + \
            offset_data[1]*self.ele_box_temp_SDD1 + offset_data[2]
        
        energy_bins_mat = np.zeros((getattr(self,f'n_SDD{SDD_number}'),340,2))

        for i in range(getattr(self, f'n_SDD{SDD_number}')):
            energy_bins_mat[i, :, :] = solexs_caldbgen.calc_ene_bins_out(
                gain_f[i], offset_f[i])
        
        return energy_bins_mat