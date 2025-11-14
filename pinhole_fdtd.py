import numpy as np
import meep as mp
from meep.materials import Al
from meep.materials import Si3N4
import pickle
import os
import pathlib

vary_domain_size=True
RealMats=True
defocus=False
overwrite=False

def pinhole(radius=2.5, flattening=0.0, angle=0.0, cone_angle=0.0, resolution=80, wavelength=0.6328,
             beam_diameter=6.0, roughness=0.0, folder='', polarization='x', time=4.0, **kwargs):
    if defocus:
        defocus_str = '_defocus.pickle'
    else:
        defocus_str = '.pickle'

    diam_str = '_'+str(beam_diameter)+'um_beam'

    if roughness is pinhole.__defaults__[9]:
        if vary_domain_size:
            if radius*2>=7.5:
                xwidth=10
                ywidth=10
                input_filename='/pinhole_in_10um'+diam_str+defocus_str
            elif radius*2>=5.5:
                xwidth=8
                ywidth=8
                input_filename='/pinhole_in_8um'+diam_str+defocus_str
            elif radius*2>=3.5:
                xwidth=6
                ywidth=6
                input_filename='/pinhole_in_6um'+diam_str+defocus_str
            else:
                xwidth=4
                ywidth=4
                input_filename='/pinhole_in_4um'+diam_str+defocus_str
        else:
            xwidth=10
            ywidth=10
            input_filename='/pinhole_in_10um'+diam_str+defocus_str
    else:
        if vary_domain_size:
            if radius*2>=5.5:
                xwidth=10
                ywidth=10
                input_filename='/pinhole_in_10um'+diam_str+defocus_str
            elif radius*2>=3.5:
                xwidth=8
                ywidth=8
                input_filename='/pinhole_in_8um'+diam_str+defocus_str
            elif radius*2>=1.5:
                xwidth=6
                ywidth=6
                input_filename='/pinhole_in_6um'+diam_str+defocus_str
            else:
                xwidth=4
                ywidth=4
                input_filename='/pinhole_in_4um'+diam_str+defocus_str
        else:
            xwidth=10
            ywidth=10
            input_filename='/pinhole_in_10um'+diam_str+defocus_str

    zlength=3
    cell = mp.Vector3(xwidth,ywidth,zlength)
    theta=angle*np.pi/180
    cone_theta=cone_angle*np.pi/180
    lam=wavelength

    Al_epsr=np.real(Al.epsilon(1/0.6328)[0,0])
    Al_epsi=np.imag(Al.epsilon(1/0.6328)[0,0])
    Al_D=2*np.pi*Al_epsi/(lam*Al_epsr)
    fakeAl = mp.Medium(epsilon=Al_epsr, D_conductivity=Al_D)

    SiN_epsr=np.real(Si3N4.epsilon(1/0.6328)[0,0])
    SiN_epsi=np.imag(Si3N4.epsilon(1/0.6328)[0,0])
    SiN_D=2*np.pi*SiN_epsi/(lam*SiN_epsr)
    fakeSiN = mp.Medium(epsilon=SiN_epsr, D_conductivity=SiN_D)

    SiNthiccness=1
    Althiccness=0.2
    substrate_loc=0.2

    if RealMats:
        coating=Al
        membrane=Si3N4
    else:
        coating=fakeAl
        membrane=fakeSiN
    extra_mats = [coating, membrane, mp.vacuum]

    xrad=radius
    yrad=xrad*(1-flattening)

    with open('bumps.pickle', 'rb') as f:
            bumps_arr = pickle.load(f)
            res = pickle.load(f)
            circumference = pickle.load(f)
    def bumps(phi, z):
        z_pix = int(np.floor((z+Althiccness/2)*res))-1
        phi_pix = int(np.round(phi*circumference*res/(2*np.pi)))
        bump = bumps_arr[phi_pix, z_pix]
        return bump

    def AlPinhole(vector):
        r=(vector.x**2 + vector.y**2)**(0.5)
        phi=np.arctan2(vector.y, vector.x)-theta
        zval=vector.z
        if cone_angle is pinhole.__defaults__[3]:
            cone_offset=0
        else:
            cone_offset=np.tan(cone_theta)*(zval-0.5*(SiNthiccness+Althiccness))
        if roughness==0.0:
            bumpterm=0
        else:
            bumpterm = roughness*bumps(phi, zval)
        ellipse=(xrad*yrad/((yrad*np.cos(phi))**2+(xrad*np.sin(phi))**2)**(0.5))+cone_offset+bumpterm
        if r > ellipse:
            if zval > 0.6:
                return coating
            return membrane
        return mp.vacuum


    AlPinhole.do_averaging = False

    geometry = [mp.Block(center=mp.Vector3(0,0,substrate_loc),
                    size=mp.Vector3(xwidth,ywidth,SiNthiccness+Althiccness),
                    material=AlPinhole)]

    if angle is pinhole.__defaults__[2]:
        angle_str=''
    else:
        angle_str='_angle='+str(angle)
    if cone_angle is pinhole.__defaults__[3]:
        cone_str=''
    else:
        cone_str='_cone='+str(cone_angle)
    if roughness is pinhole.__defaults__[9]:
        rough_str=''
    else:
        rough_str='_roughness='+str(roughness)
    res_str='_res='+str(resolution)

    pol_str='_'+polarization+'_pol'
    
    filename = str(2*xrad)+"um_f="+str(flattening)+angle_str+cone_str+res_str+rough_str+pol_str


    cwd = pathlib.Path(os.getcwd())
    with open(str(cwd)+input_filename, "rb") as f:
                source_arr = pickle.load(f)

    if polarization == 'x':
        pol=mp.Ex
    elif polarization == 'y':
        pol=mp.Ey
    else:
        pol=None
    sources = [
        mp.Source(
            src=mp.ContinuousSource(wavelength=lam, is_integrated=True),
            component=pol,
            size=mp.Vector3(xwidth,ywidth,0),
            center=mp.Vector3(0,0,-0.8),
            amp_data=source_arr
        )
    ]

    pml_layers = [mp.Absorber(0.5)]

    sim = mp.Simulation(cell_size=cell,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        extra_materials=extra_mats,
                        sources=sources,
                        resolution=resolution,
                        filename_prefix=None,
                        force_complex_fields=True)
    
    if folder is not pinhole.__defaults__[8]:
         sim.use_output_directory(folder)
         folderfile=folder+'/pinhole-'+filename+'.h5'
    else:
        folderfile='pinhole-'+filename+'.h5'
    print(folderfile)
    if not overwrite:
        print('not overwriting files')
        if os.path.isfile(folderfile):
            print("file already exists")
            return

    sim.run(mp.in_volume(mp.Volume(mp.Vector3(0,0,substrate_loc+0.5*(SiNthiccness+Althiccness)), size=mp.Vector3(xwidth,ywidth,0)), mp.to_appended(filename, mp.at_every(time, mp.output_efield_x), mp.at_every(time, mp.output_efield_y))),
        until=time)

import argparse
import inspect
def funopt(fun, argv=None):
    parser = argparse.ArgumentParser()

    if hasattr(inspect, 'getfullargspec'):
        spec = inspect.getfullargspec(fun)
    else:
        spec = inspect.getargspec(fun)

    num_defaults = len(spec.defaults) if spec.defaults is not None else 0
    for i in range(len(spec.args)):
        if i < len(spec.args) - num_defaults:
            parser.add_argument(spec.args[i])
        elif spec.defaults[i - len(spec.args)] is False:
            parser.add_argument('--' + spec.args[i], 
                                default=False, action='store_true')
        else:
            default = spec.defaults[i - len(spec.args)]
            parser.add_argument('--' + spec.args[i],
                                default=default,
                                type=type(default))
    if spec.varargs is not None:
            parser.add_argument(spec.varargs,
                                nargs='*')

    kwargs = vars(parser.parse_args(argv))
    args = []
    for arg in spec.args:
        args += [kwargs[arg]]
    if spec.varargs is not None:
        args += kwargs[spec.varargs]

    fun(*args)


funopt(pinhole)
