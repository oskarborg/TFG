import matplotlib.pyplot as plt
import numpy as np

import yt

h0 = 1*0.01*1.6916

def _conffield(field,data):
    return(data["chombo", "phi"]*data["chombo","chi"]**(-1/2)/1.6916)

def fft_comp(ds, iu1,level, low, delta):
    cube = ds.covering_grid(level, left_edge=low, dims=delta, fields=[iu1], num_ghost_zones = 1)

    u = cube[iu1].d

    nx, ny, nz = u.shape

    # do the FFTs -- note that since our data is real, there will be
    # too much information here.  fftn puts the positive freq terms in
    # the first half of the axes -- that's what we keep.  Our
    # normalization has an '8' to account for this clipping to one
    # octant.
    # ru = np.fft.fftn(u*(avgchi)**-2)  [
    #     0: nx // 2 + 1, 0: ny // 2 + 1, 0: nz // 2 + 1
    # ]
    ru = np.fft.fftn(u)[
        0: nx // 2 + 1, 0: ny // 2 + 1, 0: nz // 2 + 1
    ]
    ru = 8.0 * ru / (nx * ny * nz)

    return np.abs(ru) ** 2 /(2*np.pi)**3

def doit(ds):
    max_level = ds.index.max_level

    low = ds.domain_left_edge
    dims = ds.domain_dimensions
    
    nx, ny, nz = dims

    Kk = np.zeros((nx // 2+1 , ny // 2 +1, nz // 2+1))

    Kk += 0.5 * fft_comp(
            ds, ("chombo", "conffield"),max_level, low, dims)
    

    # wavenumbers
    L = (ds.domain_right_edge - ds.domain_left_edge).d

    kx = 2* np.pi * np.fft.rfftfreq(nx) * nx / (L[0]*0.01*1.691595167014911)
    ky = 2* np.pi *np.fft.rfftfreq(ny) * ny / (L[1]*0.01*1.691595167014911)
    kz = 2* np.pi *np.fft.rfftfreq(nz) * nz / (L[2]*0.01*1.691595167014911)

    # physical limits to the wavenumbers
    kmin = np.min(2* np.pi *1.0 / (L*0.01*1.691595167014911))
    kmax = np.min(2* np.pi *0.5 * dims / (L*0.01*1.691595167014911))

    kbins = np.arange(kmin, kmax, kmin)
    N = len(kbins)

    # bin the Fourier KE into radial kbins
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    whichbin = np.digitize(k.flatten(), kbins)
    ncount = np.bincount(whichbin, minlength=len(kbins))

    E_spectrum = np.zeros(len(ncount) - 1)
    for n in range(1, len(ncount)):
        E_spectrum[n - 1] = np.sum(Kk.flat[whichbin == n])

    k = 0.5 * (kbins[0 : N - 1] + kbins[1:N])
    E_spectrum = E_spectrum[0:N-1]

    index = np.argmax(E_spectrum)
    kmax = k[index]

    avgchi = ad.quantities.weighted_average_quantity(("chombo", "chi"), "ones")
    scale_factor = 1/avgchi
    conf_time=(scale_factor-1)*h0
    print(k)    
    plt.plot(k, E_spectrum, label = np.floor(float(conf_time)))

    plt.xlabel(r"Wave number")
    plt.ylabel(r"Power spectrum")
    plt.legend(title = 'Conformal time', loc='upper right', fontsize='small')
    plt.yscale("log")
    plt.xlim(kmin,6)
    plt.ylim(10**-10,10**-3)

    plt.savefig("spectrum.png")

# Enable parallelism in the script (assuming it was called with
# `mpirun -np <n_procs>` )
yt.enable_parallelism()

ts = yt.DatasetSeries([
    "/home/oskar/TFG/CuarticPotential_10-4_N64_L18/hdf5/ScalarFieldp_000000.3d.hdf5",
    "/home/oskar/TFG/CuarticPotential_10-4_N64_L18/hdf5/ScalarFieldp_220000.3d.hdf5",
    "/home/oskar/TFG/CuarticPotential_10-4_N64_L18/hdf5/ScalarFieldp_317000.3d.hdf5",
    "/home/oskar/TFG/CuarticPotential_10-4_N64_L18/hdf5/ScalarFieldp_1038000.3d.hdf5",
    "/home/oskar/TFG/CuarticPotential_10-4_N64_L18/hdf5/ScalarFieldp_2335000.3d.hdf5",
    #"/home/oskar/TFG/ScalarField2/hdf5/ScalarFieldp*"
    #"/home/oskar/Desktop/TFG/Experimental_tools-main/CTTK/InitialCond*"
])

storage = {}
# The serial equivalent of piter() here is just "for ds in ts:" .
i=0
Num_plots = 3
Num_plots = np.floor((len(ts)-1)/Num_plots)
for store, ds in ts.piter(storage=storage): 
    data = ds.all_data()
    if (i%Num_plots == (Num_plots-1)):
    #if (True):
        
        ad = ds.all_data()
        ds.add_field(
            ("chombo", "conffield"),
            units="dimensionless",
            function=_conffield,
            sampling_type="local",  
        )
    
        grad_fields = ds.add_gradient_fields(("chombo", "conffield"))
        doit(ds)
    i=i+1

