import matplotlib.pyplot as plt
import numpy as np

import yt
h0 = 1.1547*0.01/1.6

def _potenergy(field,data):
    return(0.25*10**(-4)*(data["chombo", "phi"])**4)
def _kinenergy(field,data):
        return(0.5*(data["chombo", "Pi"])**2)
def _pertenergy(field,data):
        hx = data["chombo", "phi_gradient_x"]
        hy = data["chombo", "phi_gradient_y"]
        hz = data["chombo", "phi_gradient_z"]
        return(0.5*(hx*hx+hy*hy+hz*hz)*data.ds.length_unit**2)

# Enable parallelism in the script (assuming it was called with
# `mpirun -np <n_procs>` )
yt.enable_parallelism()

# By using wildcards such as ? and * with the load command, we can load up a
# Time Series containing all of these datasets simultaneously.
# The "entropy" field that we will use below depends on the electron number
# density, which is not in these datasets by default, so we assume full
# ionization using the "default_species_fields" kwarg.
ts = yt.load(
    "/home/oskar/TFG/CuarticPotential_10-4_N32_L18/hdf5/ScalarFieldp_??????.3d.hdf5"
    #"/home/oskar/TFG/ScalarField2/hdf5/ScalarFieldp*"
)

storage = {}

# By using the piter() function, we can iterate on every dataset in
# the TimeSeries object.  By using the storage keyword, we can populate
# a dictionary where the dataset is the key, and sto.result is the value
# for later use when the loop is complete.

# The serial equivalent of piter() here is just "for ds in ts:" .

for store, ds in ts.piter(storage=storage):
    ad = ds.all_data()
    
    grad_fields = ds.add_gradient_fields(("chombo", "phi"))
    
    ds.add_field(
    ("chombo", "potenergy"),
    function=_potenergy,
    sampling_type="local",
)
    ds.add_field(
    ("chombo", "kinenergy"),
    function=_kinenergy,
    sampling_type="local",
)
    ds.add_field(
    ("chombo", "pertenergy"),
    function=_pertenergy,
    sampling_type="local",
)
    
    avgchi = ad.quantities.weighted_average_quantity(("chombo", "chi"), "ones")
    avgkin = ad.quantities.weighted_average_quantity(("chombo", "kinenergy"), "ones")*avgchi**-2
    avgpot = ad.quantities.weighted_average_quantity(("chombo", "potenergy"), "ones")*avgchi**-2
    avgpert = ad.quantities.weighted_average_quantity(("chombo", "pertenergy"), "ones")#*avgchi**-2
    scale_factor = 1/avgchi
    conf_time=(scale_factor-1)*h0
    # Create a sphere of radius 100 kpc at the center of the dataset volume
    # Store the current time and sphere entropy for this dataset in our
    # storage dictionary as a tuple
    store.result = (conf_time, avgkin, avgpot, avgpert, avgkin+avgpot+avgpert)

# Convert the storage dictionary values to a Nx2 array, so the can be easily
# plotted
arr = np.array(list(storage.values()))
sorting_indices = np.argsort(arr[:, 0])
arr = arr[sorting_indices]
# Plot up the results: time versus entropy
plt.plot(arr[:, 0], arr[:, 1], label = "Kinetic Energy")
plt.plot(arr[:, 0], arr[:, 2], label = "Potential Energy")
plt.plot(arr[:, 0], arr[:, 3], label = "Energy in Perturbations")
plt.plot(arr[:, 0], arr[:, 4], label = "Total Energy")
plt.xlabel("Time")
plt.legend(title = 'Energy content', loc='lower right', fontsize='small')
plt.ylabel("Energy content")
plt.savefig("time_versus_entropy.png")
plt.yscale("log")