#####################################################
# @Author: Abhilash Sarwade
# @Date:   2019-11-13 09:55:47
# @email: sarwade@isac.gov.in
# @File Name: binary_read.py
# @Project: solexs_pipeline

# @Last Modified time: 2023-12-08 02:02:47 pm
#####################################################

import os
import numpy as np
import pkg_resources
import datetime
# from numba import jit#, prange
#from . import calibration_spectrum_fitting
from .logging import setup_logger

log = setup_logger(__name__)

HDR_SIZE = 20 #bytes
SPECTRAL_DATA_SIZE = 680 #bytes
TIMING_DATA_SIZE = 60 #bytes

SPACE_PACKET_HEADER_SIZE = 24 #bytes
PLD_PACKET_HEADER_SIZE = 72 #bytes

BCF_DIR=pkg_resources.resource_filename('solexs_pipeline','CALDB/aditya-l1/solexs/data/bcf')
HK_CONVERSION_FILE_SDD1 = f'{BCF_DIR}/hk/HK_conversion_params_SDD1.txt'
HK_CONVERSION_FILE_SDD2 = f'{BCF_DIR}/hk/HK_conversion_params_SDD2.txt'


class solexs_header():
    def __init__(self,hdr_data_arr):
        n_data_packets = hdr_data_arr.shape[0]

        #checking for packet header F9 A4 2B B1
        self.hdr_check = (hdr_data_arr[:,0]==249) & (hdr_data_arr[:,1]==164) & (hdr_data_arr[:,2]==43) & (hdr_data_arr[:,3]==177)

        assert np.sum(self.hdr_check == False) == 0, "Header inconsistent"

        self.frame_id = hdr_data_arr[:,4] # fifth byte

        assert np.sum(np.diff(self.frame_id) ==  1) == len(self.frame_id)-1, "Frame ID is not continuous"

        hk_conv_data1 = self.read_hk_conversion_file(HK_CONVERSION_FILE_SDD1)
        hk_conv_data2 = self.read_hk_conversion_file(HK_CONVERSION_FILE_SDD2)

        log.info(f'Housekeeping data calibration file for SDD1: {HK_CONVERSION_FILE_SDD1}')
        log.info(f'Housekeeping data calibration file for SDD2: {HK_CONVERSION_FILE_SDD2}')

        #sixth byte
        self.det_id = np.bitwise_and(hdr_data_arr[:,5],1)
        self.flare_trigger = np.right_shift(np.bitwise_and(hdr_data_arr[:,5],2),1)
        self.shaping_time = np.right_shift(np.bitwise_and(hdr_data_arr[:,5],12),2)
        
        for i in range(len(self.shaping_time)):
            if self.shaping_time[i]==0:
                self.shaping_time[i] = 0.5
            elif self.shaping_time[i] ==1:
                self.shaping_time[i] = 1
            else:
                self.shaping_time[i] = 2

        self.hv_enable = np.right_shift(np.bitwise_and(hdr_data_arr[:,5],16),4)
        self.reset_enable = np.right_shift(np.bitwise_and(hdr_data_arr[:,5],32),5)

        #day counter
        self.day_count = np.right_shift(np.bitwise_and(hdr_data_arr[:,6],120),3)

        #reference counter
        ref_count1 = np.fliplr(hdr_data_arr[:,6:10]).copy()
        ref_count1.dtype = 'uint32'

        self.ref_count = np.bitwise_and(ref_count1[:,0],134217724)#134217727) 

        #shaped pulse baseline upload
        shaped_pulse_baseline1 = np.fliplr(hdr_data_arr[:,10:12]).copy()
        shaped_pulse_baseline1.dtype = 'uint16'
        self.shaped_pulse_baseline = shaped_pulse_baseline1[:,0]

        #timing_channel_energy_selection_window_threshold_lower
        timing_channel_thresh_lower1 = np.fliplr(hdr_data_arr[:,12:14]).copy()
        timing_channel_thresh_lower1.dtype = 'uint16'
        self.timing_channel_thresh_lower = np.right_shift(np.bitwise_and(timing_channel_thresh_lower1[:,0],65408),7)#*8      #timing_channel_thresh_lower1[:,0]

        #cooler current
        cooler_current_b = hdr_data_arr[:,14]        
        self.cooler_current = np.zeros(n_data_packets)
        self.cooler_current[self.det_id==0] = self.convert_hk(cooler_current_b[self.det_id==0],hk_conv_data1[1,0],hk_conv_data1[1,1])
        self.cooler_current[self.det_id==1] = self.convert_hk(cooler_current_b[self.det_id==1],hk_conv_data2[1,0],hk_conv_data2[1,1])        

        #back_contact
        back_contact_b = hdr_data_arr[:,15]
        self.back_contact = np.zeros(n_data_packets)
        self.back_contact[self.det_id==0] = self.convert_hk(back_contact_b[self.det_id==0],hk_conv_data1[0,0],hk_conv_data1[0,1])
        self.back_contact[self.det_id==1] = self.convert_hk(back_contact_b[self.det_id==1],hk_conv_data2[0,0],hk_conv_data2[0,1])

        #sdd temperature
        sdd_temp_b = hdr_data_arr[:,16]
        self.sdd_temp = np.zeros(n_data_packets)
        self.sdd_temp[self.det_id==0] = self.convert_hk(sdd_temp_b[self.det_id==0],hk_conv_data1[2,0],hk_conv_data1[2,1])
        self.sdd_temp[self.det_id==1] = self.convert_hk(sdd_temp_b[self.det_id==1],hk_conv_data2[2,0],hk_conv_data2[2,1])

        #timing_channel_energy_selection_window_threshold_higher
        timing_channel_thresh_higher1 = np.fliplr(hdr_data_arr[:,18:20]).copy()
        timing_channel_thresh_higher1.dtype = 'uint16'
        self.timing_channel_thresh_higher = np.right_shift(np.bitwise_and(timing_channel_thresh_higher1[:,0],65408),7)#*8     #timing_channel_thresh_higher1[:,0]

        #fpga data subtraction
        self.input_data_subtraction = np.right_shift(np.bitwise_and(hdr_data_arr[:,5],192),4) + np.right_shift(np.bitwise_and(hdr_data_arr[:,6],128),6) + np.right_shift(np.bitwise_and(hdr_data_arr[:,17],128),7)

        #gain selection
        gain_b = np.bitwise_and(hdr_data_arr[:,17],127)
        self.gain = self.decode_gain(gain_b)

        ## flare threshold for trigger
        flare_threshold1 = np.bitwise_and(ref_count1[:,0],3)*2**14 + np.bitwise_and(timing_channel_thresh_lower1[:,0],127)*2**7+ np.bitwise_and(timing_channel_thresh_higher1[:,0],127) #*8 for actual number
        self.flare_threshold = flare_threshold1*8

    def decode_gain(self,gain_b):
        gain = np.ones(len(gain_b))
        for i in range(7):
            gain = gain + np.bitwise_and(gain_b,2**i)/2**(2*i+1)

        return gain

    def read_hk_conversion_file(self,HK_CONVERSION_FILE):
        hk_conv_data = np.loadtxt(HK_CONVERSION_FILE,usecols=[4,5]) # back contact, cooler current, sdd temperature
        return hk_conv_data

    def convert_hk(self,hk_b,a0,a1):
        hk_u = hk_b*a0 + a1
        return hk_u


        # #sixth byte
        # sixth_byte = np.unpackbits(hdr_data_arr[:,5].reshape(n_data_packets,1),axis=1)

        # self.event_count = np.packbits(np.hstack((np.zeros((n_data_packets,5),dtype='uint8'),sixth_byte[:,:3]))) # 3 MSB bits
        # self.det_id = sixth_byte[:,-1]
        # self.reset_pulse = sixth_byte[:,3]
        # self.hv_enable = sixth_byte[:,4]

        # #seventh byte
        # seventh_byte = np.unpackbits(hdr_data_arr[:,6].reshape(n_data_packets,1),axis=1) #seventh 
        # self.day_count = np.packbits(np.hstack((np.zeros((n_data_packets,3),dtype='uint8'),seventh_byte[:,:5]))) # 5 MSB bits

        
        # eighth_byte = np.unpackbits(hdr_data_arr[:,7].reshape(n_data_packets,1),axis=1) #eighth 
        # ninth_byte = np.unpackbits(hdr_data_arr[:,8].reshape(n_data_packets,1),axis=1) #ninth 
        # tenth_byte = np.unpackbits(hdr_data_arr[:,9].reshape(n_data_packets,1),axis=1) #tenth

        # # self.ref_counter = np.packbits(np.hstack((np.zeros((n_data_packets,5),dtype='uint32'),seventh_byte[:,5:],eighth_byte,ninth_byte,tenth_byte)))
        #self.ref_counter = 

# @jit(nopython=True)#,parallel=True)
def create_spectrum(spectral_data_arr,n_channels):
    n_data_packets = spectral_data_arr.shape[0]
    spectral_data = np.zeros((n_channels,n_data_packets))
    for i in range(n_channels):
        spectral_data[i,:] = spectral_data_arr[:,2*i]*2**8 + spectral_data_arr[:,2*i+1] 
    return spectral_data



class solexs_spectrum():
    n_channels = 340
    def __init__(self,spectral_data_arr,n_channels=n_channels):
        
        # n_data_packets = spectral_data_arr.shape[0]
        # spectral_data = np.zeros((n_channels,n_data_packets))

        # #spectral_data_arr = self.data_full[:,HDR_SIZE:HDR_SIZE+SPECTRAL_DATA_SIZE]

        # for i in range(n_channels):
        #     spectral_data[i,:] = spectral_data_arr[:,2*i]*2**8 + spectral_data_arr[:,2*i+1] #
        #     # spectral_data[i,:] = np.left_shift(spectral_data_arr[:,2*i],8) + spectral_data_arr[:,2*i+1]


        spectral_data = create_spectrum(spectral_data_arr,n_channels)

        self.spectra = spectral_data

    def add_spectra(self):
        self.full_spectrum = self.spectra.sum(axis=1)

    def make_lightcurve(self,channel_start=0,channel_stop=n_channels):
        self.lightcurve = self.spectra[channel_start:channel_stop,:].sum(axis=0)
    
    def make_spectrogram(self,rebin_sec):
        spectra = self.spectra
        extra_bins = spectra.shape[1] % rebin_sec
        if extra_bins != 0:
            spectra = spectra[:,:-extra_bins]
        new_bins = int(spectra.shape[1]/rebin_sec)
        new_spectra = spectra.reshape(340,new_bins,rebin_sec).sum(axis=2)
        new_tm = np.arange(new_bins)*rebin_sec
        return new_spectra,new_tm




class solexs_lightcurve():
    def __init__(self,temporal_data_arr):
        n_data_packets = temporal_data_arr.shape[0]

        temporal_high = np.zeros((10,n_data_packets))
        temporal_med = np.zeros((10,n_data_packets))
        temporal_low = np.zeros((10,n_data_packets))

        for i in range(10):
            temporal_low[i,:] = temporal_data_arr[:,6*i]*2**8 + temporal_data_arr[:,6*i+1]
            temporal_med[i,:] = temporal_data_arr[:,6*i+2]*2**8 + temporal_data_arr[:,6*i+3]
            temporal_high [i,:]= temporal_data_arr[:,6*i+4]*2**8 + temporal_data_arr[:,6*i+5]

 
        temporal_high[1:,:] = np.diff(temporal_high,axis=0)
        temporal_med[1:,:] = np.diff(temporal_med,axis=0)
        temporal_low[1:,:] = np.diff(temporal_low,axis=0)

        tmp_high = temporal_high.T.reshape(n_data_packets*10)
        tmp_med = temporal_med.T.reshape(n_data_packets*10)
        tmp_low = temporal_low.T.reshape(n_data_packets*10)

        high = tmp_high#np.diff(tmp_high)
        med = tmp_med#np.diff(tmp_med)
        low = tmp_low#np.diff(tmp_low)

        high[high<0] = high[high<0] + 2**16
        med[med<0] = med[med<0] + 2**16
        low[low<0] = low[low<0] + 2**16

        high = self.remove_spurious_counts(high)
        med = self.remove_spurious_counts(med)
        low = self.remove_spurious_counts(low)

        self.high = high
        self.med = med
        self.low = low

        self.high_sec = self.reshape_to_per_sec(high)
        self.med_sec = self.reshape_to_per_sec(med)
        self.low_sec = self.reshape_to_per_sec(low)

    def remove_spurious_counts(self,count_rate):# count_rate - [high, med, low]
        SPURIOUS_THRESHOLD = 3e4
        ids = np.where(count_rate>SPURIOUS_THRESHOLD)
        ids = ids[0]
        log.warning(f'Number of spurious counts removed from lightcurve: {len(ids)}')
        new_count_rate = count_rate
        for ii in ids:
            new_count_rate[ii] = np.median(count_rate[ii-5:ii+5])
        return new_count_rate

    def reshape_to_per_sec(self,count_rate):# count_rate - [high, med, low]
        new_count_rate = count_rate.reshape(int(len(count_rate)/10),10).sum(axis=1)
        return new_count_rate


class SDD_data_structure():
    def __init__(self,data_sdd): #sdd_id = 0 or 1
        # det_id = np.bitwise_and(data_full[:,5],1)
        # data_sdd = data_full[det_id==sdd_id,:]
        # data_sdd1 = data_full[det_id==1,:]

        # HDR_SIZE = 20 #bytes
        # SPECTRAL_DATA_SIZE = 680 #bytes
        # TIMING_DATA_SIZE = 60 #bytes        

        hdr_data_arr_sdd = data_sdd[:,:HDR_SIZE]
        
        spectral_data_arr_sdd = data_sdd[:,HDR_SIZE:HDR_SIZE+SPECTRAL_DATA_SIZE]

        temporal_data_arr_sdd = data_sdd[:,-TIMING_DATA_SIZE:]

        
        self.hdr_data = solexs_header(hdr_data_arr_sdd)
        self.spectral_data = solexs_spectrum(spectral_data_arr_sdd)
        self.temporal_data = solexs_lightcurve(temporal_data_arr_sdd)
    

class space_packet_header():
    def __init__(self,space_packet_header_data):
        self.version_number = np.bitwise_and(space_packet_header_data[:,0],224) #0
        self.packet_type = np.bitwise_and(space_packet_header_data[:,0],16) #0
        self.sec_hdr_flag = np.right_shift(np.bitwise_and(space_packet_header_data[:,0],8),3) #1

        self.APID = np.bitwise_and(space_packet_header_data[:,0],7)*2**8 + space_packet_header_data[:,1] #180

        self.packet_sequence_flag = np.right_shift(np.bitwise_and(space_packet_header_data[:,2],192),6) #11

        self.packet_sequence_count = np.bitwise_and(space_packet_header_data[:,2],63)*2**8 + space_packet_header_data[:,3] #sequential

        packet_length1 = np.fliplr(space_packet_header_data[:,4:6]).copy()
        packet_length1.dtype = 'uint16'
        self.packet_length = packet_length1[:,0] #775

        self.DHT = space_packet_header_data[:,9]*2**32 + space_packet_header_data[:,10]*2**24 + space_packet_header_data[:,11]*2**16 + space_packet_header_data[:,12]*2**8 + space_packet_header_data[:,13]

        OBT1 = np.fliplr(space_packet_header_data[:,14:22]).copy()
        OBT1.dtype = 'uint64'
        self.OBT = OBT1[:,0]

        session_id1 = np.fliplr(space_packet_header_data[:,22:]).copy()
        session_id1.dtype = 'uint16'
        self.session_id = session_id1[:,0]


class pld_packet_header():
    def __init__(self, pld_packet_header_data,n_data_packets) -> None:
        self.pld_utc_time, self.pld_utc_datetime, self.pld_utc_timestamp = self.read_pld_utc_time(
            pld_packet_header_data, n_data_packets)
        
        log.info(f'Start Time: {self.pld_utc_datetime[0].isoformat()}')
        log.info(f'Stop Time: {self.pld_utc_datetime[-1].isoformat()}')
        time_duration = self.pld_utc_datetime[-1] - \
            self.pld_utc_datetime[0]
        log.info(f'Duration: {time_duration} seconds')
        
    
    def read_pld_utc_time(self, pld_packet_header_data,n_data_packets):
        pld_utc_time_bin = pld_packet_header_data[:, 32:32+28]
        pld_utc_time = np.zeros((n_data_packets,7),dtype='uint32')

        for i in range(7):
            tmp_tm = pld_utc_time_bin[:,i*4:(i+1)*4].copy()
            tmp_tm.dtype = 'uint32'
            tmp_tm = tmp_tm.reshape(n_data_packets)
            pld_utc_time[:, i] = tmp_tm

        pld_utc_time[:, -1] = 0#pld_utc_time[:, -1]#*100 #converting millisecond*10 to microsends

        pld_utc_datetime = []
        pld_utc_timestamp = np.zeros(n_data_packets)
        for i in range(n_data_packets):
            pld_utc_datetime.append(datetime.datetime(*pld_utc_time[i]))
            pld_utc_timestamp[i] = pld_utc_datetime[-1].timestamp()
        
        return pld_utc_time, pld_utc_datetime, pld_utc_timestamp
        

        

class read_solexs_binary_data():
    """
    Read SoLEXS binary data and segregate header, spectral and timing data. Data Types can be "Raw", "SP" (Space Packet), "L0"
    Output: SDD data structure
    """

    def __init__(self,input_filename,data_type='Raw'):
        log.info(f'Reading binary data.')
        log.info(f'Filename: {input_filename}')
        log.info(f'Data type: {data_type}')
        self.input_filename = input_filename
        self.data_type = data_type#'Raw'

        if self.data_type == 'Raw':

            self.packet_size = HDR_SIZE + SPECTRAL_DATA_SIZE + TIMING_DATA_SIZE

            # if not os.path.isfile(filename):

            file_size = os.path.getsize(input_filename)
            log.info(f'Size of binary file: {file_size}')
            self.n_data_packets = int(np.floor(os.path.getsize(input_filename)/self.packet_size))
            log.info(f'Number of data packets in binary file: {self.n_data_packets}')

            if np.mod(file_size,self.packet_size)==0:
                self.left_over_data_flag = 0
            else:
                self.left_over_data_flag = 1

            data_full = self.read_file()

            det_id = np.bitwise_and(data_full[:,5],1)
            data_sdd1 = data_full[det_id==0,:]
            data_sdd2 = data_full[det_id==1,:]

            log.info('Reading binary data for SDD1')
            self.SDD1 = SDD_data_structure(data_sdd1)
            log.info('Reading binary data for SDD2')
            self.SDD2 = SDD_data_structure(data_sdd2)

        if self.data_type == 'SP':

            self.packet_size = SPACE_PACKET_HEADER_SIZE + HDR_SIZE + SPECTRAL_DATA_SIZE + TIMING_DATA_SIZE  

            # if not os.path.isfile(filename):

            file_size = os.path.getsize(input_filename)
            log.info(f'Size of binary file: {file_size}')
            self.n_data_packets = int(np.floor(os.path.getsize(input_filename)/self.packet_size))
            log.info(
                f'Number of data packets in binary file: {self.n_data_packets}')

            if np.mod(file_size,self.packet_size)==0:
                self.left_over_data_flag = 0
            else:
                self.left_over_data_flag = 1

            data_full = self.read_file()

            space_packet_header_data = data_full[:,:SPACE_PACKET_HEADER_SIZE]
            data_full = data_full[:,SPACE_PACKET_HEADER_SIZE:]

            det_id = np.bitwise_and(data_full[:,5],1)
            data_sdd1 = data_full[det_id==0,:]
            data_sdd2 = data_full[det_id==1,:]

            self.SDD1 = SDD_data_structure(data_sdd1)
            self.SDD2 = SDD_data_structure(data_sdd2)

            self.SP_header = space_packet_header(space_packet_header_data)

        if self.data_type == 'L0':

            self.packet_size = PLD_PACKET_HEADER_SIZE + HDR_SIZE + SPECTRAL_DATA_SIZE + TIMING_DATA_SIZE
            file_size = os.path.getsize(input_filename)
            log.info(f'Size of binary file: {file_size}')
            self.n_data_packets = int(np.floor(os.path.getsize(input_filename)/self.packet_size))
            log.info(
                f'Number of data packets in binary file: {self.n_data_packets}')

            if np.mod(file_size, self.packet_size) == 0:
                self.left_over_data_flag = 0
            else:
                self.left_over_data_flag = 1

            data_full = self.read_file()

            pld_packet_header_data = data_full[:, :PLD_PACKET_HEADER_SIZE]
            data_full = data_full[:, PLD_PACKET_HEADER_SIZE:]

            det_id = np.bitwise_and(data_full[:, 5], 1)
            data_sdd1 = data_full[det_id == 0, :]
            data_sdd2 = data_full[det_id == 1, :]

            n_data_packets_SDD1 = np.sum(det_id==0)
            n_data_packets_SDD2 = np.sum(det_id==1)
            self.pld_header_SDD1 = pld_packet_header(
                pld_packet_header_data[det_id == 0, :], n_data_packets_SDD1)
            self.pld_header_SDD2 = pld_packet_header(
                pld_packet_header_data[det_id == 1, :], n_data_packets_SDD2)

            self.SDD1 = SDD_data_structure(data_sdd1)
            self.SDD2 = SDD_data_structure(data_sdd2)

            # self.PLD_header = pld_packet_header(pld_packet_header_data)

    def read_file(self):
        # fid = open(self.filename,'rb')
        # bin_data_full = fid.read()
        # bin_data_full_arr = bytearray(bin_data_full)

        data_full = np.fromfile(self.input_filename,dtype='uint8')

        if self.left_over_data_flag:
            log.warning('Left over bytes to be ignored.')
            extra_bytes = np.mod(self.n_data_packets,self.packet_size)
            # bin_data_full_arr = bin_data_full_arr[:-extra_bytes]
            data_full = data_full[:-extra_bytes]

        # data_full = np.array(bin_data_full_arr)

        data_full = data_full.reshape(self.n_data_packets,self.packet_size)
        return data_full

