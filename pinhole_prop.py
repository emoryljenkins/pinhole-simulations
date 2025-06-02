import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.io import fits
import os
import h5py
import pickle

import poppy
from poppy.poppy_core import PlaneType

if poppy.accel_math._USE_CUPY:
    import cupy as cp
    xp = cp
    print('cupy available')
else:
    xp = np
    print('using numpy, no cupy available')

def prop(filename_in):
    file=h5py.File(filename_in, 'r')
    ex_r=file['ex.r']
    ex_i=file['ex.i']
    ey_r=file['ey.r']
    ey_i=file['ey.i']
    ex = xp.asarray(ex_r[:,:,0]+ex_i[:,:,0])
    ey = xp.asarray(ey_r[:,:,0]+ey_i[:,:,0])

    print('loaded input matrices')

    ex_pad=xp.zeros((N,N),dtype='complex_')
    ey_pad=xp.zeros((N,N),dtype='complex_')
    insize=ex.shape[1]
    ex_pad[int(N/2-insize/2):int(N/2+insize/2),int(N/2-insize/2):int(N/2+insize/2)]=ex
    ey_pad[int(N/2-insize/2):int(N/2+insize/2),int(N/2-insize/2):int(N/2+insize/2)]=ey

    print('padding complete')


    exf = xp.matmul((m/(N*N))*xp.exp(-2j*xp.pi*U*X.T), xp.matmul(ex_pad, xp.exp(-2j*xp.pi*Y*V.T)))
    del ex_pad
    print('matrix fourier transform 50% complete')
    eyf = xp.matmul((m/(N*N))*xp.exp(-2j*xp.pi*U*X.T), xp.matmul(ey_pad, xp.exp(-2j*xp.pi*Y*V.T)))
    del ey_pad
    print('matrix fourier transform 100% complete')


    oversample=2
    wfx = poppy.FresnelWavefront(x2_mft[N-1].to(u.m), wavelength=wavelength_c, npix=N, oversample=oversample)
    wfy = poppy.FresnelWavefront(x2_mft[N-1].to(u.m), wavelength=wavelength_c, npix=N, oversample=oversample)
    wfx_arr = xp.zeros((N*oversample,N*oversample),dtype='complex')
    wfy_arr = xp.zeros((N*oversample,N*oversample),dtype='complex')
    wfx_arr[int((oversample-1)*N/2):int((oversample+1)*N/2),int((oversample-1)*N/2):int((oversample+1)*N/2)]=exf
    wfy_arr[int((oversample-1)*N/2):int((oversample+1)*N/2),int((oversample-1)*N/2):int((oversample+1)*N/2)]=eyf
    del exf
    del eyf
    wfx.wavefront=wfx_arr
    wfy.wavefront=wfy_arr
    del wfx_arr
    del wfy_arr
    wfx*=oap_ap
    wfy*=oap_ap

    print('created fresnelwavefront objects and applied aperture')

    coeffs = poppy.zernike.decompose_opd(xp.unwrap(xp.angle(wfx.wavefront))*wavelength_c.value/(xp.pi), aperture=xp.abs(wfx.wavefront), nterms=4)
    if poppy.accel_math._USE_CUPY:
        anti_coeffs = [0, -coeffs[1].get(), -coeffs[2].get(), -coeffs[3].get()]
    else:
        anti_coeffs = [0, -coeffs[1], -coeffs[2], -coeffs[3]]
    PTTD_remove = poppy.ZernikeWFE(radius=oap_diam/2, coefficients=anti_coeffs, aperture_stop=False)
    wfx*=PTTD_remove
    wfy*=PTTD_remove
    del PTTD_remove

    print('removed tip/tilt and defocus terms')

    wfx.propagate_fresnel(z_fsm)
    wfy.propagate_fresnel(z_fsm)

    print('fresnel propagated to pupil plane')

    M=wfx.wavefront.shape[1]
    unpadx=wfx.wavefront[int((M-N/crop_factor)/oversample):int((M+N/crop_factor)/oversample), int((M-N/crop_factor)/oversample):int((M+N/crop_factor)/oversample)]
    unpady=wfy.wavefront[int((M-N/crop_factor)/oversample):int((M+N/crop_factor)/oversample), int((M-N/crop_factor)/oversample):int((M+N/crop_factor)/oversample)]
    if poppy.accel_math._USE_CUPY:
        fits_array = np.stack([np.real(unpadx.get()), np.imag(unpadx.get()), np.real(unpady.get()), np.imag(unpady.get())], axis=-1)
    else:
        fits_array = np.stack([np.real(unpadx), np.imag(unpadx), np.real(unpady), np.imag(unpady)], axis=-1)
    

    del unpadx
    del unpady

    return fits_array

wavelength_c = 632.8e-9*u.m
pupil_diam = 8*u.mm
oap_diam = 0.0254*u.m

N=8192
z_OAP0=146.8*u.mm
z_fsm=196.8*u.mm

X=xp.zeros([N,1])
X[:,0]=(xp.linspace(0, N, N)-N/2)/(N)
Y=X
m=32
U=X*m
V=U

res=80 # pixels per micron from the meep sim
width=N*u.um/res #size of padded pinhole array in length units
x2=(np.linspace(-N/2,N/2-1,N)/width)*(wavelength_c*z_OAP0) #spatial vector after fraunhofer propagation given fft
x2_mft=x2*m/N #scaled spatial vector after fraunhofer propagation. scaled according to matrix fourier transform vs fft

oap_ap=poppy.CircularAperture(radius=oap_diam/2)

width=(x2_mft[N-1]-x2_mft[0]).to(u.mm)

crop_factor=4
arr_width = width/crop_factor

diameter = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
flatness = [-0.1, 0.0, 0.1]
angle = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]
numwiggles = [10.0]
wigglestrength = [0.0, 0.05]

for d in diameter:
    for f in flatness:
        for a in angle:
            for n in numwiggles:
                for w in wigglestrength:
                    print(f)
                    print(str(f))
                    print(str(a))
                    print('Beginning '+str(d)+'um '+str(f)+' flatness '+str(a)+' degrees '+str(n)+' '+str(w*100)+'% wiggles pinhole propagation')
                    filename='pinhole-'+str(d)+'um_f='+str(f)+'_angle='+str(a)+'_'+str(n)+'_wiggles_'+str(w)+'_strong.h5'
                    fits_array = prop(filename)
                    hdu = fits.PrimaryHDU(fits_array)
                    hdr = hdu.header
                    hdr['LAYER_0'] = 'real x'
                    hdr['LAYER_1'] = 'imag x'
                    hdr['LAYER_2'] = 'real y'
                    hdr['LAYER_3'] = 'imag y'
                    hdr['arrwidth'] = (arr_width.value, 'width of array in mm')
                    hdr['pinhole'] = (d, 'diameter of pinhole in um')
                    hdr['flatness'] = (f, 'flatness of pinhole')
                    hdr['angle'] = (a, 'angle of rotation in degrees')
                    hdr['numwiggl'] = (n, 'number of cycles for wiggles')
                    hdr['wigstren'] = (w, 'fractional size of wiggles')
                    hdu.writeto('pinhole_pupil_plane_arr_'+str(d)+'um_f='+str(f)+'_angle='+str(a)+'_'+str(n)+'_wiggles_'+str(w)+'_strong.fits')
                    del fits_array
                    del hdu

print('mission complete')                    