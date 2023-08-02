#####################################################
# @Author: Abhilash Sarwade
# @Date:   2023-07-28 10:26:10 pm
# @email: sarwade@ursc.gov.in
# @File Name: interm_L1.py
# @Project: solexs_pipeline
#
# @Last Modified time: 2023-08-01 12:31:23 pm
#####################################################

import numpy as np
from astropy.io import fits
import os
import glob
import pkg_resources
from .fits_utils import PHAII, LC

"""TODO
1. Define pi energy bins [DONE]
2. Input should be interm files. Do not use L0 data at all. [DONE]
3. Run caldbgen first. [NOT NEEDED]
4. PI files, rebin to ebounds in CPF. [DONE]
5. Time in PI files should be one entire day in units of seconds from certain epoch. [DONE]
6. Time in interm PHA files to time in PI files? Let us not fix time array for L1. let the first time value be variable. [DONE]
"""

CPF_DIR = pkg_resources.resource_filename(
    'solexs_pipeline', 'CALDB/aditya-l1/solexs/data/cpf')


class L1_directory():

    def __init__(self,interm_dir_path) -> None:
        self.interm_dir_path = interm_dir_path
        self.input_filename = os.path.basename(self.interm_dir_path).split('_interm')[0]
        pass

    def make_l1_dir(self, output_dir=None, clobber=True):
        output_dir = os.path.join(os.path.dirname(self.interm_dir_path),f'{self.input_filename}_L1')
        if os.path.exists(output_dir) and clobber:  # TODO add log
            os.removedirs(output_dir)
        os.makedirs(output_dir)
        sdd1_output_dir = os.path.join(output_dir, 'SDD1')
        sdd2_output_dir = os.path.join(output_dir, 'SDD2')

        self.output_dir = output_dir
        os.makedirs(sdd1_output_dir)
        os.makedirs(sdd2_output_dir)

    def load_interm_file(self,SDD_number,file_type): #file_type -> pha/lc/hk
        file_path = os.path.join(self.interm_dir_path,f'SDD{SDD_number}',f'*.{file_type}')
        file_name = glob.glob(file_path)
        if not file_name:
            # TODO Exit gracefully using proper FileNotFoundError https://stackoverflow.com/questions/36077266/how-do-i-raise-a-filenotfounderror-properly
            print(f'{file_name} not found. Exiting...')
        hdus = fits.open(file_name[0])
        return hdus
    
    def rebin_pha_to_pi(self,SDD_number,hdus_pha):
        pi_ene_file = os.path.join(CPF_DIR,'ebounds','ebounds_out',f'energy_bins_out_PI_SDD{SDD_number}.dat')
        pi_ene = np.loadtxt(pi_ene_file)

        # hdus_pha = self.load_interm_file(SDD_number,'pha')

        n_spec = hdus_pha[1].data.shape[0]
        pi_spec = np.zeros((n_spec,340))

        for i in range(n_spec):
            pha_ene_min = hdus_pha[1].data['E_MIN'][i]
            pha_ene_max = hdus_pha[1].data['E_MAX'][i]
            tmp_pha_spec = hdus_pha[1].data['COUNTS'][i]

            tmp_pi_spec_min = np.interp(pi_ene[:,0],pha_ene_min,tmp_pha_spec)
            tmp_pi_spec_max = np.interp(pi_ene[:,1],pha_ene_max,tmp_pha_spec)
            tmp_pi_spec = (tmp_pi_spec_min + tmp_pi_spec_max)/2
            pi_spec[i,:] = tmp_pi_spec
        
        return pi_spec
        
    def allday_pi_spec(self,time_solexs,pi_spec): #time in seconds
        assert (time_solexs[-1] - time_solexs[0]
                ) < 86400.0  # TODO Exit gracefully

        tbinsize = 1 #second
        nbins = int(86400.0/tbinsize)
        self.nbins_pha = nbins

        tday0 = int(time_solexs[0]/86400.0)*86400.0
        t0 = (time_solexs[0]-int((time_solexs[0]-tday0)/tbinsize)*tbinsize)

        all_time = np.arange(0, nbins)*tbinsize+t0
        all_pi_spec = np.empty((nbins,340))
        all_pi_spec[:] = np.nan

        for i, t in enumerate(time_solexs):
            tbin = int((t-tday0)/tbinsize)
            all_pi_spec[tbin,:] = pi_spec[i,:]
        
        return all_time, all_pi_spec

    def allday_lc(self,time_solexs,lc):
        assert (time_solexs[-1] - time_solexs[0]
                ) < 86400.0  # TODO Exit gracefully

        tbinsize = 1  # second
        nbins = int(86400.0/tbinsize)

        tday0 = int(time_solexs[0]/86400.0)*86400.0
        t0 = (time_solexs[0]-int((time_solexs[0]-tday0)/tbinsize)*tbinsize)

        all_time = np.arange(0, nbins)*tbinsize+t0
        all_lc = np.empty(nbins)
        all_lc[:] = np.nan

        for i, t in enumerate(time_solexs):
            tbin = int((t-tday0)/tbinsize)
            all_lc[tbin] = lc[i]

        return all_time, all_lc
    
    def pi_file(self,SDD_number):
        hdus_pha = self.load_interm_file(SDD_number, 'pha')
        filename = hdus_pha[0].header['filename']
        pi_spec = self.rebin_pha_to_pi(SDD_number, hdus_pha)
        time_solexs = hdus_pha[1].data['TSTART']
        all_time, all_pi_spec = self.allday_pi_spec(time_solexs,pi_spec)

        # n_SDD = getattr(self, f'n_SDD{SDD_number}')
        ch = np.arange(340)
        channel = np.tile(ch, (self.nbins_pha, 1))
        # assuming integration time is 1 second each
        telapse = np.ones(self.nbins_pha,np.int8)
        counts = all_pi_spec
        quality = np.zeros(self.nbins_pha,np.int8)
        exposure = np.ones(self.nbins_pha,np.int8)
        respfile = np.array([None]*self.nbins_pha)
        
        l1_pi_file = PHAII(filename, all_time, telapse, channel, counts, quality, exposure, respfile)
        return l1_pi_file
    
    def lc_file(self,SDD_number):
        hdus_lc = self.load_interm_file(SDD_number, 'lc')
        filename = hdus_lc[0].header['filename']
        lc_data = hdus_lc[2].data
        time_solexs = lc_data['TIME']
        all_time, lc_low = self.allday_lc(time_solexs, lc_data['COUNTS_LOW'])
        all_time, lc_med = self.allday_lc(time_solexs, lc_data['COUNTS_MED'])
        all_time, lc_high = self.allday_lc(time_solexs, lc_data['COUNTS_HIGH'])
        all_time, lc_all = self.allday_lc(time_solexs, lc_data['COUNTS_ALL'])

        lower_thresh = hdus_lc[1].data['LOWER_THRESH']
        higher_thresh = hdus_lc[1].data['HIGHER_THRESH']

        # Handle changes in threshold values separately
        assert len(np.unique(lower_thresh)) == 1
        assert len(np.unique(higher_thresh)) == 1

        minchan = np.array(
            [0, np.unique(lower_thresh), np.unique(higher_thresh), 0]) # low, med, high, all
        maxchan = np.array([np.unique(lower_thresh), np.unique(
            higher_thresh), 10000, 10000])  # low, med, high, all
        #TODO Maxchan for highest band
        #TODO Energy-channel calibration for temporal band

        l1_lc_file = LC(f'{filename}_SDD{SDD_number}', all_time, lc_low,
                        lc_med, lc_high, lc_all, minchan, maxchan)
        
        return l1_lc_file
        

    def calc_gti(self,SDD_number): # Not Implemented #TODO
        pass 

    def create_l1_files(self,SDD_number):
        l1_pi_file = self.pi_file(SDD_number)
        l1_lc_file = self.lc_file(SDD_number)

        return l1_pi_file, l1_lc_file
    
    def write_l1_files(self,SDD_number):
        l1_pi_file, l1_lc_file = self.create_l1_files(SDD_number)
        sdd_l1_dir = os.path.join(self.output_dir,f'SDD{SDD_number}')

        l1_pi_filename = os.path.join(sdd_l1_dir,f'{self.input_filename}_SDD{SDD_number}_L1.pha')
        l1_pi_file.writeto(l1_pi_filename)

        l1_lc_filename = os.path.join(sdd_l1_dir,f'{self.input_filename}_SDD{SDD_number}_L1.lc')
        l1_lc_file.writeto(l1_lc_filename)