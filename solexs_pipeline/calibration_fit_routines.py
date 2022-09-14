#####################################################
# @Author: Abhilash Sarwade
# @Date:   2020-02-14 14:51:30
# @email: sarwade@isac.gov.in
# @File Name: calibration_fit_routines.py
# @Project: solexs_pipeline

# @Last Modified time: 2022-09-14 11:12:07
#####################################################

import numpy as np
from scipy import optimize
from scipy.signal import argrelextrema
from scipy.signal import savgol_filter

def read_spectrum_mca(filename):
    fid = open(filename,'rb')
    lines = fid.read()
    fid.close()
    spec_lines = re.findall(r'<<DATA>>(.*)<<END>>',str(lines))
    spec_lines = spec_lines[0].split('\\r\\n')[1:-1]
    spec = []
    for l in spec_lines:
        spec.append(int(l))
    spec = np.array(spec)
    return spec

def gaussian(x, height, center, width):
    return height*np.exp(-(x - center)**2/(2*width**2)) 

def two_gaussian(x, h1, c1, w1, h2, c2, w2):
    add_gauss = gaussian(x,h1,c1,w1) + gaussian(x,h2,c2,w2)
    return add_gauss

def fit_gaussian(ch,spec,guess=[5000,112,6],lower=77,upper=185):
    errfunc = lambda p, x, y, y_err: (gaussian(x, *p) - y)/y_err
    lower = int(lower)
    upper = int(upper)
    spec_err = np.sqrt(spec)
    spec_err[spec_err==0]=1
    p_out = optimize.leastsq(errfunc, guess[:], args=(ch[lower:upper], spec[lower:upper], spec_err[lower:upper]),full_output=1, epsfcn=0.0001)#,bounds=bnds)
    pfit = p_out[0]
    pcov = p_out[1]
    if (len(ch[lower:upper]) > len(guess)) and pcov is not None:
        s_sq = (errfunc(pfit, ch[lower:upper], spec[lower:upper], spec_err[lower:upper])**2).sum()/(len(ch[lower:upper])-len(guess))
        pcov = pcov * s_sq
    error = [] 
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 

    return pfit_leastsq, perr_leastsq 

def fit_two_gaussian(ch,spec,guess=[5000,112,6,900,123,12],lower=77,upper=185):
    errfunc = lambda p, x, y, y_err: (two_gaussian(x, *p) - y)/y_err
    lower = int(lower)
    upper = int(upper)
    spec_err = np.sqrt(spec)
    spec_err[spec_err==0]=1
    p_out = optimize.leastsq(errfunc, guess[:], args=(ch[lower:upper], spec[lower:upper], spec_err[lower:upper]),full_output=1, epsfcn=0.0001)#,bounds=bnds)
    pfit = p_out[0]
    pcov = p_out[1]
    if (len(ch[lower:upper]) > len(guess)) and pcov is not None:
        s_sq = (errfunc(pfit, ch[lower:upper], spec[lower:upper], spec_err[lower:upper])**2).sum()/(len(ch[lower:upper])-len(guess))
        pcov = pcov * s_sq
    error = [] 
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 

    return pfit_leastsq, perr_leastsq 

def e_ch(ch,m,c):
    energy = ch*m+c
    return energy

def e_ch_solexs(ch,m,c):
    ch_512 = []
    for chi in ch:
        if chi<=167:
            ch_512.append(chi)
        elif chi>167:
            ch_512.append((chi-167)*2+167)

    ch_512 = np.array(ch_512)
    energy = ch_512*m+c
    return energy

def fit_e_ch(ene_peak,ch_peak,ch_peak_err=None):
    if ch_peak_err is None:
        ch_peak_err = [1]*len(ch_peak)
        ch_peak_err = np.array(ch_peak_err)
    errfunc = lambda p, x, y, y_err: ((x - p[1])/p[0] - y)/y_err
    guess = [14,-30]
    p_out = optimize.leastsq(errfunc, guess[:], args=(ene_peak, ch_peak, ch_peak_err),full_output=1, epsfcn=0.0001)
    pfit = p_out[0]
    pcov = p_out[1]
    if (len(ene_peak) > len(guess)) and pcov is not None:
        s_sq = (errfunc(pfit, ene_peak, ch_peak, ch_peak_err)**2).sum()/(len(ene_peak)-len(guess))
        pcov = pcov * s_sq
    error = [] 
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 
    return pfit_leastsq, perr_leastsq 

def fit_e_ch_solexs(ene_peak,ch_peak,ch_peak_err=None):
    ch_512 = []
    for chi in ch_peak:
        if chi<=167:
            ch_512.append(chi)
        elif chi>167:
            ch_512.append((chi-167)*2+167)

    ch_peak_512 = np.array(ch_512)

    pfit_leastsq, perr_leastsq = fit_e_ch(ene_peak,ch_peak_512,ch_peak_err)
    return pfit_leastsq, perr_leastsq


def e_fwhm(energy,elec_noise_fwhm,fano_factor):
    fwhm = np.sqrt(elec_noise_fwhm**2 + 8*np.log(2)*3.8*fano_factor*energy)
    return fwhm

def fit_e_fwhm(ene_peak,fwhm,fwhm_err=None):
    if fwhm_err is None:
        fwhm_err = [1]*len(fwhm)
        fwhm_err = np.array(fwhm)

    errfunc = lambda p, x, y, y_err: (e_fwhm(x,p[0],p[1]) - y)/y_err
    guess = [50,0.114]
    p_out = optimize.leastsq(errfunc, guess[:], args=(ene_peak, fwhm, fwhm_err),full_output=1, epsfcn=0.0001)
    pfit = p_out[0]
    pcov = p_out[1]
    if (len(ene_peak) > len(guess)) and pcov is not None:
        s_sq = (errfunc(pfit, ene_peak, fwhm,fwhm_err)**2).sum()/(len(ene_peak)-len(guess))
        pcov = pcov * s_sq
    error = [] 
    for i in range(len(pfit)):
        try:
          error.append(np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )
    pfit_leastsq = pfit
    perr_leastsq = np.array(error) 
    return pfit_leastsq, perr_leastsq 


def get_fwhm_fe_spectrum(ch,spec): # expects SoLEXS binning and Fe Ka in range of 60 to 120
    ene_peak = [5.89e3,6.49e3]

    fe_ka_guess_ch = ch[np.argmax(spec[60:120]) + 60]
    fe_ka_guess_a = np.max(spec[60:120])
    fit_results,err_fit_results = fit_two_gaussian(ch,spec,guess=[fe_ka_guess_a,fe_ka_guess_ch,3,fe_ka_guess_a*0.2,fe_ka_guess_ch + 15,4],lower=np.floor(fe_ka_guess_ch*0.9),upper=np.ceil(fe_ka_guess_ch*1.17))
    ch_peak = [fit_results[1],fit_results[4]]
    ch_peak_err = [err_fit_results[1],err_fit_results[1]]
    fit_ene,err_fit_ene =  fit_e_ch(ene_peak,ch_peak,ch_peak_err)
    fwhm = 2*np.sqrt(2*np.log(2))*fit_results[2]*fit_ene[0]
    err_fwhm = 2*np.sqrt(2*np.log(2))*(fit_ene[0]*err_fit_results[2] + err_fit_ene[0]*fit_results[2])

    fitted_spectrum = two_gaussian(ch,*fit_results)

    return fwhm, err_fwhm

class fe_spectrum():
    def __init__(self,ch,spec):
        self.ch = ch
        self.spec = spec

        fit_ene, err_fit_ene = self.calibrate_e_ch(self.ch,self.spec)
        self.gain = fit_ene[0]
        self.offset = fit_ene[1]
        self.err_gain = err_fit_ene[0]
        self.err_offset = err_fit_ene[1]

        fwhm, err_fwhm, fitted_spectrum = self.fit_spec(self.ch,self.spec,self.gain,self.err_gain)
        self.fwhm = fwhm
        self.err_fwhm = err_fwhm
        self.fitted_spectrum = fitted_spectrum


    def calibrate_e_ch(self,ch,spec):
        ene_peak = [5.89e3,6.49e3]

        fe_ka_guess_ch = ch[np.argmax(spec[60:120]) + 60]
        fe_ka_guess_a = np.max(spec[60:120])
        fit_results,err_fit_results = fit_two_gaussian(ch,spec,guess=[fe_ka_guess_a,fe_ka_guess_ch,3,fe_ka_guess_a*0.2,fe_ka_guess_ch*1.1,4],lower=np.floor(fe_ka_guess_ch*0.9),upper=np.ceil(fe_ka_guess_ch*1.17))
        ch_peak = [fit_results[1],fit_results[4]]
        ch_peak_err = [err_fit_results[1],err_fit_results[1]]
        fit_ene,err_fit_ene =  fit_e_ch(ene_peak,ch_peak,ch_peak_err)
        return fit_ene, err_fit_ene

    def fit_spec(self,ch,spec,gain,err_gain=0):
        fe_ka_guess_ch = ch[np.argmax(spec[60:120]) + 60]
        fe_ka_guess_a = np.max(spec[60:120])
        fit_results,err_fit_results = fit_two_gaussian(ch,spec,guess=[fe_ka_guess_a,fe_ka_guess_ch,3,fe_ka_guess_a*0.2,fe_ka_guess_ch*1.1,4],lower=np.floor(fe_ka_guess_ch*0.9),upper=np.ceil(fe_ka_guess_ch*1.17))        

        fwhm = 2*np.sqrt(2*np.log(2))*fit_results[2]*gain
        err_fwhm = 2*np.sqrt(2*np.log(2))*(gain*err_fit_results[2] + err_gain*fit_results[2])

        fitted_spectrum = two_gaussian(ch,*fit_results)

        return fwhm, err_fwhm, fitted_spectrum


class fe_ti_spectrum():
    def __init__(self,ch,spec):
        self.ch = ch
        self.spec = spec

        fit_ene, err_fit_ene = self.calibrate_e_ch(self.ch,self.spec)
        self.gain = fit_ene[0]
        self.offset = fit_ene[1]
        self.err_gain = err_fit_ene[0]
        self.err_offset = err_fit_ene[1]

        fwhm_fe_ka, err_fwhm_fe_ka, fwhm_fe_kb, err_fwhm_fe_kb, fwhm_ti_ka, err_fwhm_ti_ka, fitted_spectrum = self.fit_spec(self.ch,self.spec,self.gain,self.err_gain)
        # self.fwhm_fe_ka = fwhm_fe_ka
        # self.err_fwhm_fe_ka = err_fwhm_fe_ka
        # self.fwhm_fe_kb = fwhm_fe_kb
        # self.err_fwhm_fe_kb = err_fwhm_fe_kb
        # self.fwhm_ti_ka = fwhm_ti_ka
        # self.err_fwhm_ti_ka = err_fwhm_ti_ka

        self.fwhms = [fwhm_fe_ka,fwhm_fe_kb,fwhm_ti_ka]
        self.err_fwhms = [err_fwhm_fe_ka,err_fwhm_fe_kb,err_fwhm_ti_ka]

        self.fitted_spectrum = fitted_spectrum


    def calibrate_e_ch(self,ch,spec):
        ene_peak = [5.89e3,6.49e3,4.51e3]
        self.ene_peak = ene_peak

        fe_ka_guess_ch = ch[np.argmax(spec[60:150]) + 60]
        fe_ka_guess_a = np.max(spec[60:150])
        fit_results,err_fit_results = fit_two_gaussian(ch,spec,guess=[fe_ka_guess_a,fe_ka_guess_ch,3,fe_ka_guess_a*0.2,fe_ka_guess_ch*1.1,4],lower=np.floor(fe_ka_guess_ch*0.9),upper=np.ceil(fe_ka_guess_ch*1.17))
        fit_results_ti, err_fit_results_ti = fit_gaussian(ch,spec,guess=[fe_ka_guess_a*0.2,fe_ka_guess_ch*0.75,2],lower=np.floor(fe_ka_guess_ch*0.7),upper=np.ceil(fe_ka_guess_ch*0.8))
        ch_peak = [fit_results[1],fit_results[4],fit_results_ti[1]]
        ch_peak_err = [err_fit_results[1],err_fit_results[1],err_fit_results_ti[1]]

        

        fit_ene,err_fit_ene =  fit_e_ch(ene_peak,ch_peak,ch_peak_err)
        return fit_ene, err_fit_ene

    def fit_spec(self,ch,spec,gain,err_gain=0):
        fe_ka_guess_ch = ch[np.argmax(spec[60:150]) + 60]
        fe_ka_guess_a = np.max(spec[60:150])
        fit_results,err_fit_results = fit_two_gaussian(ch,spec,guess=[fe_ka_guess_a,fe_ka_guess_ch,3,fe_ka_guess_a*0.2,fe_ka_guess_ch*1.1,4],lower=np.floor(fe_ka_guess_ch*0.9),upper=np.ceil(fe_ka_guess_ch*1.17))        
        fit_results_ti, err_fit_results_ti = fit_gaussian(ch,spec,guess=[fe_ka_guess_a*0.2,fe_ka_guess_ch*0.75,2],lower=np.floor(fe_ka_guess_ch*0.7),upper=np.ceil(fe_ka_guess_ch*0.8))

        fwhm_fe_ka = 2*np.sqrt(2*np.log(2))*fit_results[2]*gain
        err_fwhm_fe_ka = 2*np.sqrt(2*np.log(2))*(gain*err_fit_results[2] + err_gain*fit_results[2])

        fwhm_fe_kb = 2*np.sqrt(2*np.log(2))*fit_results[5]*gain
        err_fwhm_fe_kb = 2*np.sqrt(2*np.log(2))*(gain*err_fit_results[5] + err_gain*fit_results[5])

        fwhm_ti_ka = 2*np.sqrt(2*np.log(2))*fit_results_ti[2]*gain
        err_fwhm_ti_ka = 2*np.sqrt(2*np.log(2))*(gain*err_fit_results_ti[2] + err_gain*fit_results_ti[2])

        fitted_spectrum = two_gaussian(ch,*fit_results)+ gaussian(ch,*fit_results_ti)

        return fwhm_fe_ka, err_fwhm_fe_ka, fwhm_fe_kb, err_fwhm_fe_kb, fwhm_ti_ka, err_fwhm_ti_ka, fitted_spectrum


class jsc_spectrum():
    def __init__(self,ch,spec):
        self.ch = ch
        self.spec = spec

        fit_ene, err_fit_ene = self.calibrate_e_ch(self.ch,self.spec)
        self.gain = fit_ene[0]
        self.offset = fit_ene[1]
        self.err_gain = err_fit_ene[0]
        self.err_offset = err_fit_ene[1]

        fwhms, err_fwhms, fitted_spectrum = self.fit_spec(self.ch,self.spec,self.gain,self.err_gain)
        self.fwhms = fwhms
        self.err_fwhms = err_fwhms
        self.fitted_spectrum = fitted_spectrum


    def calibrate_e_ch(self,ch,spec):
        tmp_ene_peak = [6403,3691,2622,7057]
        ene_peak = [6403,7057,4510,4931,5414,5946,3691,4012]
        self.ene_peak = ene_peak

        ids_max = argrelextrema(spec[20:],np.greater)[0]+20
        spec_max = spec[ids_max]

        tmp_ids = np.flip(np.argsort(spec_max),axis=0)
        spec_max = spec_max[tmp_ids]
        ids_max = ids_max[tmp_ids]

        tmp_ch_peak = ids_max[:4]

        tmp_fit_ene,tmp_err_fit_ene =  fit_e_ch(tmp_ene_peak,tmp_ch_peak)

        fe_ka_guess_ch = ids_max[0]
        fe_ka_guess_a = spec[ids_max[0]]

        fit_results_fe,err_fit_results_fe = fit_two_gaussian(ch,spec,guess=[fe_ka_guess_a,fe_ka_guess_ch,3,fe_ka_guess_a*0.2,ids_max[3],4],lower=np.floor(fe_ka_guess_ch*0.95),upper=np.ceil(fe_ka_guess_ch*1.15))

        fit_results_ti, err_fit_results_ti = fit_two_gaussian(ch,spec,guess=[fe_ka_guess_a*0.2,ids_max[5],2,fe_ka_guess_a*0.01,ids_max[5]*1.09,2],lower=np.floor(ids_max[5]*0.93),upper=np.ceil(ids_max[5]*1.15))

        fit_results_cr, err_fit_results_cr = fit_two_gaussian(ch,spec,guess=[fe_ka_guess_a*0.2,ids_max[4],2,fe_ka_guess_a*0.01,ids_max[4]*1.09,2],lower=np.floor(ids_max[4]*0.93),upper=np.ceil(ids_max[4]*1.13))

        ca_ka_guess_ch = (3691 - tmp_fit_ene[1])/tmp_fit_ene[0]
        fit_results_ca, err_fit_results_ca = fit_two_gaussian(ch,spec,guess=[fe_ka_guess_a*0.2,ca_ka_guess_ch,2,fe_ka_guess_a*0.04,ca_ka_guess_ch*1.09,2],lower=np.floor(ca_ka_guess_ch*0.93),upper=np.ceil(ca_ka_guess_ch*1.13))



        ch_peak = [fit_results_fe[1],fit_results_fe[4],fit_results_ti[1],fit_results_ti[4],fit_results_cr[1],fit_results_cr[4],fit_results_ca[1],fit_results_ca[4]]
        ch_peak_err = [err_fit_results_fe[1],err_fit_results_fe[4],err_fit_results_ti[1],err_fit_results_ti[4],err_fit_results_cr[1],err_fit_results_cr[4],err_fit_results_ca[1],err_fit_results_ca[4]]
        

        fit_ene,err_fit_ene =  fit_e_ch(ene_peak,ch_peak,ch_peak_err)
        return fit_ene, err_fit_ene

    def fit_spec(self,ch,spec,gain,err_gain=0):
        ids_max = argrelextrema(spec[20:],np.greater)[0]+20
        spec_max = spec[ids_max]

        tmp_ids = np.flip(np.argsort(spec_max),axis=0)
        spec_max = spec_max[tmp_ids]
        ids_max = ids_max[tmp_ids]

        tmp_ch_peak = ids_max[:4]
        
        tmp_ene_peak = [6403,3691,2622,7057]
        tmp_fit_ene,tmp_err_fit_ene =  fit_e_ch(tmp_ene_peak,tmp_ch_peak)

        fe_ka_guess_ch = ids_max[0]
        fe_ka_guess_a = spec[ids_max[0]]

        fit_results_fe,err_fit_results_fe = fit_two_gaussian(ch,spec,guess=[fe_ka_guess_a,fe_ka_guess_ch,3,fe_ka_guess_a*0.2,ids_max[3],4],lower=np.floor(fe_ka_guess_ch*0.95),upper=np.ceil(fe_ka_guess_ch*1.15))

        fit_results_ti, err_fit_results_ti = fit_two_gaussian(ch,spec,guess=[fe_ka_guess_a*0.2,ids_max[5],2,fe_ka_guess_a*0.01,ids_max[5]*1.09,2],lower=np.floor(ids_max[5]*0.93),upper=np.ceil(ids_max[5]*1.15))

        fit_results_cr, err_fit_results_cr = fit_two_gaussian(ch,spec,guess=[fe_ka_guess_a*0.2,ids_max[4],2,fe_ka_guess_a*0.01,ids_max[4]*1.09,2],lower=np.floor(ids_max[4]*0.93),upper=np.ceil(ids_max[4]*1.13))

        ca_ka_guess_ch = (3691 - tmp_fit_ene[1])/tmp_fit_ene[0]
        fit_results_ca, err_fit_results_ca = fit_two_gaussian(ch,spec,guess=[fe_ka_guess_a*0.2,ca_ka_guess_ch,2,fe_ka_guess_a*0.04,ca_ka_guess_ch*1.09,2],lower=np.floor(ca_ka_guess_ch*0.93),upper=np.ceil(ca_ka_guess_ch*1.13))

        fitted_spectrum = two_gaussian(ch,*fit_results_fe) + two_gaussian(ch,*fit_results_ti) + two_gaussian(ch,*fit_results_cr) + two_gaussian(ch,*fit_results_ca)

        fwhms = np.array([fit_results_fe[2],fit_results_fe[5],fit_results_ti[2],fit_results_ti[5],fit_results_cr[2],fit_results_cr[5],fit_results_ca[2],fit_results_ca[5]])*2*np.sqrt(2*np.log(2))*gain
        err_fwhms = 2*np.sqrt(2*np.log(2))*(gain*np.array([err_fit_results_fe[2],err_fit_results_fe[5],err_fit_results_ti[2],err_fit_results_ti[5],err_fit_results_cr[2],err_fit_results_cr[5],err_fit_results_ca[2],err_fit_results_ca[5]])) + err_gain*fwhms

        return fwhms, err_fwhms, fitted_spectrum


class mu_sdd2_spectrum():
    def __init__(self,ch,spec,pb_guess_ch,ni_guess_ch,fe_guess_ch,pb_2_guess_ch,zr_guess_ch):
        self.ch = ch
        self.spec = spec

        fit_ene, err_fit_ene = self.calibrate_e_ch(self.ch,self.spec,pb_guess_ch,ni_guess_ch,fe_guess_ch,pb_2_guess_ch,zr_guess_ch)
        self.gain = fit_ene[0]
        self.offset = fit_ene[1]
        self.err_gain = err_fit_ene[0]
        self.err_offset = err_fit_ene[1]

        fwhms, err_fwhms, fitted_spectrum = self.fit_spec(self.ch,self.spec,self.gain,self.err_gain,pb_guess_ch,ni_guess_ch,fe_guess_ch,pb_2_guess_ch,zr_guess_ch)
        self.fwhms = fwhms
        self.err_fwhms = err_fwhms
        self.fitted_spectrum = fitted_spectrum


    def calibrate_e_ch(self,ch,spec,pb_guess_ch,ni_guess_ch,fe_guess_ch,pb_2_guess_ch,zr_guess_ch):
        # tmp_ene_peak = [12613,15775,10551,6403,7478]#[6403,3691,2622,7057]
        ene_peak = [6403,7478,10551,12613,15775]#[6403,7057,4510,4931,5414,5946,3691,4012]
        self.ene_peak = ene_peak

        ids_max1 = argrelextrema(savgol_filter(spec,5,1)[180:270],np.greater)[0]+180
        ids_max2 = argrelextrema(savgol_filter(spec,3,1)[100:170],np.greater)[0]+100


        spec_max = spec[ids_max1]

        tmp_ids = np.flip(np.argsort(spec_max),axis=0)
        spec_max = spec_max[tmp_ids]
        ids_max1 = ids_max1[tmp_ids]


        spec_max = spec[ids_max2]

        tmp_ids = np.flip(np.argsort(spec_max),axis=0)
        spec_max = spec_max[tmp_ids]
        ids_max2 = ids_max2[tmp_ids]

        # tmp_ch_peak = np.array([ids_max[1],ids_max[2],ids_max[3],ids_max[4],ids_max[7]])

        # tmp_fit_ene,tmp_err_fit_ene =  fit_e_ch_solexs(tmp_ene_peak,tmp_ch_peak)

        # pb_guess_ch = ids_max1[2]#ids_max[3]
        pb_guess_a = spec[pb_guess_ch]

        # ni_guess_ch = ids_max2[1]#ids_max[7]
        ni_guess_a = spec[ni_guess_ch]

        # fe_guess_ch = ids_max2[0]#ids_max[4]
        fe_guess_a = spec[fe_guess_ch]

        fit_results_pb,err_fit_results_pb = fit_gaussian(ch,spec,guess=[pb_guess_a,pb_guess_ch,1],lower=np.floor(pb_guess_ch*0.985),upper=np.ceil(pb_guess_ch*1.015))

        fit_results_ni,err_fit_results_ni = fit_gaussian(ch,spec,guess=[ni_guess_a,ni_guess_ch,1],lower=np.floor(ni_guess_ch*0.98),upper=np.ceil(ni_guess_ch*1.015))

        fit_results_fe,err_fit_results_fe = fit_gaussian(ch,spec,guess=[fe_guess_a,fe_guess_ch,1],lower=np.floor(fe_guess_ch*0.98),upper=np.ceil(fe_guess_ch*1.02))

        fit_st_line = np.polyfit(np.arange(np.ceil(ids_max1[1]*1.03),int(pb_guess_ch*1.45)),np.log10(spec)[int(np.ceil(ids_max1[1]*1.03)):int(pb_guess_ch*1.45)],1)
        
        pow_spec = np.zeros(len(spec))#
        pow_spec[int(pb_guess_ch*1.108*1.03):int(pb_guess_ch*1.45)] = 10**fit_st_line[1]*10**(np.arange(int(pb_guess_ch*1.108*1.03),int(pb_guess_ch*1.45))*fit_st_line[0])

        # pb_2_guess_ch = ids_max1[0]
        pb_2_guess_a = spec[pb_2_guess_ch]

        fit_results_pb_2,err_fit_results_pb_2 = fit_gaussian(ch,spec-pow_spec,guess=[pb_2_guess_a,pb_2_guess_ch,1],lower=np.floor(pb_2_guess_ch*0.995),upper=np.ceil(pb_2_guess_ch*1.01))


        # zr_guess_ch = ids_max1[1]
        zr_guess_a = spec[zr_guess_ch] - pow_spec[zr_guess_ch]

        fit_results_zr,err_fit_results_zr = fit_gaussian(ch,spec-pow_spec,guess=[zr_guess_a,zr_guess_ch,1],lower=zr_guess_ch-2,upper=zr_guess_ch+1)



        ch_peak = [fit_results_fe[1],fit_results_ni[1],fit_results_pb[1],fit_results_pb_2[1],fit_results_zr[1]]
        ch_peak_err = [err_fit_results_fe[1],err_fit_results_ni[1],err_fit_results_pb[1],err_fit_results_pb_2[1],err_fit_results_zr[1]]
        

        fit_ene,err_fit_ene =  fit_e_ch_solexs(ene_peak,ch_peak,ch_peak_err)
        return fit_ene, err_fit_ene

    def fit_spec(self,ch,spec,gain,err_gain,pb_guess_ch,ni_guess_ch,fe_guess_ch,pb_2_guess_ch,zr_guess_ch):
        tmp_ene_peak = [12613,15775,10551,6403,7478]#[6403,3691,2622,7057]
        ene_peak = [6403,7478,10551,12613,15775]#[6403,7057,4510,4931,5414,5946,3691,4012]

        ids_max1 = argrelextrema(savgol_filter(spec,7,1)[200:],np.greater)[0]+200
        ids_max2 = argrelextrema(savgol_filter(spec,3,1)[100:200],np.greater)[0]+100


        spec_max = spec[ids_max1]

        ids_max1 = argrelextrema(savgol_filter(spec,5,1)[180:270],np.greater)[0]+180
        ids_max2 = argrelextrema(savgol_filter(spec,3,1)[100:170],np.greater)[0]+100


        spec_max = spec[ids_max1]

        tmp_ids = np.flip(np.argsort(spec_max),axis=0)
        spec_max = spec_max[tmp_ids]
        ids_max1 = ids_max1[tmp_ids]


        spec_max = spec[ids_max2]

        tmp_ids = np.flip(np.argsort(spec_max),axis=0)
        spec_max = spec_max[tmp_ids]
        ids_max2 = ids_max2[tmp_ids]

        # tmp_ch_peak = np.array([ids_max[1],ids_max[2],ids_max[3],ids_max[4],ids_max[7]])

        # tmp_fit_ene,tmp_err_fit_ene =  fit_e_ch_solexs(tmp_ene_peak,tmp_ch_peak)

        # pb_guess_ch = ids_max1[2]#ids_max[3]
        pb_guess_a = spec[pb_guess_ch]

        # ni_guess_ch = ids_max2[1]#ids_max[7]
        ni_guess_a = spec[ni_guess_ch]

        # fe_guess_ch = ids_max2[0]#ids_max[4]
        fe_guess_a = spec[fe_guess_ch]

        fit_results_pb,err_fit_results_pb = fit_gaussian(ch,spec,guess=[pb_guess_a,pb_guess_ch,1],lower=np.floor(pb_guess_ch*0.985),upper=np.ceil(pb_guess_ch*1.015))

        fit_results_ni,err_fit_results_ni = fit_gaussian(ch,spec,guess=[ni_guess_a,ni_guess_ch,1],lower=np.floor(ni_guess_ch*0.98),upper=np.ceil(ni_guess_ch*1.015))

        fit_results_fe,err_fit_results_fe = fit_gaussian(ch,spec,guess=[fe_guess_a,fe_guess_ch,1],lower=np.floor(fe_guess_ch*0.98),upper=np.ceil(fe_guess_ch*1.02))

        fit_st_line = np.polyfit(np.arange(np.ceil(ids_max1[1]*1.03),int(pb_guess_ch*1.45)),np.log10(spec)[int(np.ceil(ids_max1[1]*1.03)):int(pb_guess_ch*1.45)],1)
        
        pow_spec = np.zeros(len(spec))#
        pow_spec[int(pb_guess_ch*1.108*1.03):int(pb_guess_ch*1.45)] = 10**fit_st_line[1]*10**(np.arange(int(pb_guess_ch*1.108*1.03),int(pb_guess_ch*1.45))*fit_st_line[0])

        # pb_2_guess_ch = ids_max1[0]
        pb_2_guess_a = spec[pb_2_guess_ch]

        fit_results_pb_2,err_fit_results_pb_2 = fit_gaussian(ch,spec-pow_spec,guess=[pb_2_guess_a,pb_2_guess_ch,1],lower=np.floor(pb_2_guess_ch*0.995),upper=np.ceil(pb_2_guess_ch*1.01))


        # zr_guess_ch = ids_max1[1]
        zr_guess_a = spec[zr_guess_ch]

        fit_results_zr,err_fit_results_zr = fit_gaussian(ch,spec-pow_spec,guess=[zr_guess_a,zr_guess_ch,1],lower=zr_guess_ch-2,upper=zr_guess_ch+1)

        ch_peak = [fit_results_fe[1],fit_results_ni[1],fit_results_pb[1],fit_results_pb_2[1],fit_results_zr[1]]

        fitted_spectrum = gaussian(ch,*fit_results_fe) + gaussian(ch,*fit_results_ni) + gaussian(ch,*fit_results_pb) + gaussian(ch,*fit_results_pb_2) + gaussian(ch,*fit_results_zr) + pow_spec

        gain_arr = np.ones(len(ch_peak))*gain
        gain_arr[np.array(ch_peak)>167] = gain*2

        fwhms = np.array([fit_results_fe[2],fit_results_ni[2],fit_results_pb[2],fit_results_pb_2[2],fit_results_zr[2]])*2*np.sqrt(2*np.log(2))*gain_arr
        err_fwhms = 2*np.sqrt(2*np.log(2))*(gain_arr*np.array([err_fit_results_fe[2],err_fit_results_ni[2],err_fit_results_pb[2],err_fit_results_pb_2[2],err_fit_results_zr[2]])) + err_gain*fwhms

        return fwhms, err_fwhms, fitted_spectrum

