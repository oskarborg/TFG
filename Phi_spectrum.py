import matplotlib.pyplot as plt
import numpy as np

import yt

h0 = 1.1547*0.01/1.6

def fft_comp(ds, iu1,level, low, delta):
    cube = ds.covering_grid(level, left_edge=low, dims=delta, fields=[iu1], num_ghost_zones = 1)

    u = cube[iu1].d

    nx, ny, nz = u.shape

    # do the FFTs -- note that since our data is real, there will be
    # too much information here.  fftn puts the positive freq terms in
    # the first half of the axes -- that's what we keep.  Our
    # normalization has an '8' to account for this clipping to one
    # octant.
    # ru = np.fft.fftn(u*(avgchi)**-2)[
    #     0: nx // 2 + 1, 0: ny // 2 + 1, 0: nz // 2 + 1
    # ]
    ru = np.fft.fftn(u)[
        0: nx // 2 + 1, 0: ny // 2 + 1, 0: nz // 2 + 1
    ]
    ru = 8.0 * ru / (nx * ny * nz)

    return np.abs(ru) ** 2#/(18**3)

def doit(ds):
    # a FFT operates on uniformly gridded data.  We'll use the yt
    # covering grid for this.
    max_level = ds.index.max_level

    low = ds.domain_left_edge
    dims = ds.domain_dimensions
    
    nx, ny, nz = dims

    Kk = np.zeros((nx // 2+1 , ny // 2 +1, nz // 2+1))

    Kk += 0.5 * fft_comp(
            ds, ("chombo", "phi"),max_level, low, dims)
    

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
    print("Kk.shape:", Kk.shape)
    print("whichbin.shape:", whichbin.shape)
    print("Kk.flatten().shape:", Kk.flatten().shape)
    print("whichbin == n shape for n=1:", (whichbin == 1).shape)
    for n in range(1, len(ncount)):
        E_spectrum[n - 1] = np.sum(Kk.flat[whichbin == n])

    k = 0.5 * (kbins[0 : N - 1] + kbins[1:N])
    E_spectrum = E_spectrum[1:N]

    index = np.argmax(E_spectrum)
    kmax = k[index]

    avgchi = ad.quantities.weighted_average_quantity(("chombo", "chi"), "ones")
    scale_factor = 1/avgchi
    conf_time=(scale_factor-1)*h0
    
    #plt.plot(k, E_spectrum, label = np.floor(Conftime))
    plt.plot(k, E_spectrum, label = np.floor(conf_time))

    plt.xlabel(r"$k$")
    plt.ylabel(r"$E(k)dk$")
    plt.legend(title = 'Power spectrum', loc='upper right', fontsize='small')
    plt.yscale("log")
    plt.ylim(10**-8,10**-1)

    plt.savefig("spectrum.png")
    
def _pertenergy(field,data):
    dd = data.ds.r[:,:,:]
    physical_cell_volume =  dd['dx']**3*dd['chi']**(-1.5)    # physical cell volume taking into account the conformal factor "chi". 
    total_volume = np.sum(physical_cell_volume)
    energy = dd["phi"]
    energy = energy.flatten()
    print(energy.shape)
    avgenergy = np.sum(energy * physical_cell_volume)/total_volume
    return ( (data["chombo" , "phi"]-avgenergy)*data["chombo", "chi"]**(-1))

# Enable parallelism in the script (assuming it was called with
# `mpirun -np <n_procs>` )
yt.enable_parallelism()

# By using wildcards such as ? and * with the load command, we can load up a
# Time Series containing all of these datasets simultaneously.
# The "entropy" field that we will use below depends on the electron number
# density, which is not in these datasets by default, so we assume full
# ionization using the "default_species_fields" kwarg.
ts = yt.load(
    "/home/oskar/TFG/CuarticPotential_10-4_N32_L18/hdf5/ScalarFieldp*"
    #"/home/oskar/TFG/ScalarField2/hdf5/ScalarFieldp*"
    #"/home/oskar/Desktop/TFG/Experimental_tools-main/CTTK/InitialCond*"
)

storage = {}

# By using the piter() function, we can iterate on every dataset in
# the TimeSeries object.  By using the storage keyword, we can populate
# a dictionary where the dataset is the key, and sto.result is the value
# for later use when the loop is complete.

# The serial equivalent of piter() here is just "for ds in ts:" .
i=0
Num_plots = 5
Num_plots = np.floor((len(ts)-1)/Num_plots)
for store, ds in ts.piter(storage=storage): 
    data = ds.all_data()
    if (i%Num_plots == 0):
    #if (True):
        
        ad = ds.all_data()
    
        grad_fields = ds.add_gradient_fields(("chombo", "phi"))
        # ds.add_field(
        #     ("chombo", "pert"),
        #     function=_pertenergy,
        #     sampling_type="local",
        #     )
        doit(ds)
        # Create a sphere of radius 100 kpc at the center of the dataset volume
        # Store the current time and sphere entropy for this dataset in our
        # storage dictionary as a tuple
    i=i+1

