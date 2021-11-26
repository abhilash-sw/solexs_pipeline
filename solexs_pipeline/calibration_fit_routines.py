#####################################################
# @Author: Abhilash Sarwade
# @Date:   2020-02-14 14:51:30
# @email: sarwade@isac.gov.in
# @File Name: calibration_fit_routines.py
# @Project: solexs_pipeline

# @Last Modified time: 2021-11-26 13:56:23
#####################################################

import numpy as np
from scipy import optimize

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


def fit_fe_spectrum(ch,spec): # expects SoLEXS binning and Fe Ka in range of 60 to 120
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
