import matplotlib.pyplot as plt
import numpy as np

import yt

h0 = 1.1547*0.01/1.6

def _conffield(field,data):
    return(data["chombo", "phi"])#*data["chombo","chi"]**-1)

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
)

storage = {}

# By using the piter() function, we can iterate on every dataset in
# the TimeSeries object.  By using the storage keyword, we can populate
# a dictionary where the dataset is the key, and sto.result is the value
# for later use when the loop is complete.

# The serial equivalent of piter() here is just "for ds in ts:" .

for store, ds in ts.piter(storage=storage):
    ad = ds.all_data()
    # ds.add_field(
    #     ("chombo", "conffield"),
    #     function=_conffield,
    #     sampling_type="local",
    # )
    avgchi = ad.quantities.weighted_average_quantity(("chombo", "chi"), "ones")
    scale_factor = 1/avgchi
    conf_time=(scale_factor-1)*h0
    # Create a sphere of radius 100 kpc at the center of the dataset volume
    # var = ad.quantities.weighted_standard_deviation(("chombo", "phi"), "ones")  
    # Store the current time and sphere entropy for this dataset in our
    # storage dictionary as a tuple
    avgconffield = ad.quantities.weighted_standard_deviation(("chombo", "phi"), "ones")
    store.result = (conf_time, avgconffield[0])#avgchi -1 o -2?
    
    # Convert the storage dictionary values to a Nx2 array, so the can be easily
# plotted
arr = np.array(list(storage.values()))
sorting_indices = np.argsort(arr[:, 0])
arr = arr[sorting_indices]
# Plot up the results: time versus entropy
plt.plot(arr[:, 0], arr[:, 1])
plt.xlabel("Time")
plt.ylabel("Variance of the conformal field")
plt.savefig("Variance.png")
