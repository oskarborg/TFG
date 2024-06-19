

import matplotlib.pyplot as plt
import numpy as np

import yt

"""
Make a power spectrum. 
We aim to compute:

                      1  ^*         ^*
     E(k) = integral  -  phi(k) . phi(k) dS
                      2


(Note: sometimes we normalize by 1/volume to get a spectral
energy density spectrum).


"""
def _phi_with_ghost_zones(field, data):
    num_ghost_zones = 1  # Adjust this value as needed
    cg = data.ds.covering_grid(
        level=0,
        left_edge=data.ds.domain_left_edge,
        dims=data.ds.domain_dimensions,
        num_ghost_zones=num_ghost_zones
    )
    return cg[("chombo", "phi")]

def _energy(field,data):
    num_ghost_zones = 0  # Adjust this value as needed
    cg = data.ds.covering_grid(
        level=0,
        left_edge=data.ds.domain_left_edge,
        dims=data.ds.domain_dimensions,
        num_ghost_zones=num_ghost_zones
    )
    hx = cg[("chombo", "phi_gradient_x")]
    hy = cg[("chombo", "phi_gradient_y")]
    hz = cg[("chombo", "phi_gradient_z")]
    energy = 0.25*10**(-4)*(cg["chombo", "phi"])*(cg["chombo", "phi"])*(cg["chombo", "phi"])*(cg["chombo", "phi"])
    energy += 0.5*(cg["chombo", "Pi"])*(cg["chombo", "Pi"])
    energy = 0.5*(hx*hx+hy*hy+hz*hz)*(cg["chombo", "chi"])**-4*data.ds.length_unit**2
    return(energy)

def _pertfield(field, data):
    dd = data.ds.r[:,:,:]
    physical_cell_volume =  dd['dx']**3*dd['chi']**(-1.5)    # physical cell volume taking into account the conformal factor "chi". 
    total_volume = np.sum(physical_cell_volume)
    energy = dd["energy"]
    energy = energy.flatten()
    print(energy.shape)
    avgenergy = np.sum(energy * physical_cell_volume)/total_volume
    return ( (data["chombo" , "energy"]-avgenergy)/avgenergy )

def doit(ds):
    num_ghost_zones = 1  # Adjust this value as needed
    data = ds.all_data()
    cg = data.ds.covering_grid(
    level=0,
    left_edge=data.ds.domain_left_edge,
    dims=data.ds.domain_dimensions,
    num_ghost_zones=num_ghost_zones
)
    cg.ds.add_field(
    ("chombo", "phig"),
    function=_phi_with_ghost_zones,
    units="auto",  # Use appropriate units
    sampling_type="cell",
    force_override = True
)
    # Ensure the derived field is accessed to create the ghost zones
    data = ds.all_data()
    phi_with_ghost_zones = cg[("chombo", "phig")]
    grad_phields = cg.ds.add_gradient_fields(("chombo", "phi"))
    cg.add_field(
    ("chombo", "pertenergy"),
    function=_pertenergy,
    sampling_type="local",
)
    
    cg.ds.add_field(
    ("chombo", "energy"), 
    function=_energy,
    sampling_type="local",
)
    
    cg.ds.add_field(
     ("chombo", "pert"), 
     function=_pertfield,
     sampling_type="local",
 )
    max_level = ds.index.max_level

    low = ds.domain_left_edge
    dims = ds.domain_dimensions

    nx, ny, nz = dims

    Kk = np.zeros((nx // 2 + 1, ny // 2 + 1, nz // 2 + 1))

    Kk += 0.5 * (
        fft_comp(ds, ("chombo", "pert"), max_level, low, dims)
    )
    # wavenumbers
    L = (ds.domain_right_edge - ds.domain_left_edge).d


    kx = np.fft.rfftfreq(nx) * nx / L[0]
    ky = np.fft.rfftfreq(ny) * ny / L[1]
    kz = np.fft.rfftfreq(nz) * nz / L[2]

    # # physical limits to the wavenumbers
    kmin = np.min(1.0 / L)
    kmax = np.min(0.5 * dims / L)
    kbins = np.arange(kmin, kmax, kmin)
    N = len(kbins)

    # bin the Fourier KE into radial kbins
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.sqrt(kx3d**2 + ky3d**2 + kz3d**2)

    whichbin = np.digitize(k.flat, kbins)
    ncount = np.bincount(whichbin)

    E_spectrum = np.zeros(len(ncount) - 1)

    for n in range(1, len(ncount)):
        E_spectrum[n - 1] = np.sum(Kk.flat[whichbin == n])

    k = 0.5 * (kbins[1: N - 1] + kbins[2:N])
    E_spectrum = E_spectrum[2:N]

    index = np.argmax(E_spectrum)
    kmax = k[index]

    plt.plot(k, E_spectrum, label = np.floor(ds.current_time.in_units("s")))
    plt.xlabel(r"$k$")
    plt.ylabel(r"$E(k)dk$")
    plt.legend(title = 'Power spectrum', loc='upper right', fontsize='small')
    plt.yscale("log")

    plt.savefig("spectrum.png")


def fft_comp(ds, iu1, level, low, delta):
    cube = ds.covering_grid(level, left_edge=low, dims=delta, fields=[iu1])

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
    
    
    ru = np.fft.fftn((u))[
        0: nx // 2 + 1, 0: ny // 2 + 1, 0: nz // 2 + 1
    ]
    ru = 8.0 * ru / (nx * ny * nz)

    return np.abs(ru) ** 2


# ds = yt.load("/home/oskar/Desktop/TFG/Experimental_tools-main/CTTK/InitialConditionsFinal.3d.hdf5")
#ds = yt.load("/home/oskar/TFG/ScalarField2/hdf5/ScalarFieldp_002600.3d.hdf5")
# ds = yt.load("/home/oskar/TFG/REHEATING/hdf5/ScalarFieldp_000000.3d.hdf5")

ts = yt.load("/home/oskar/TFG/CuarticPotN32/hdf5/ScalarFieldp*")
#ts = yt.load("/home/oskar/TFG/REHEATING/hdf5/ScalarFieldp*")

for i in range(0, len(ts)-1, int((len(ts)-1)/6)):
    doit(ts[i])
