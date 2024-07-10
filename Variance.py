import matplotlib.pyplot as plt
import numpy as np

import yt

h0 = 1*0.01*1.6916

def _conffield(field,data):
    return(data["chombo", "phi"]*data["chombo","chi"]**(-1/2)/1.6916)

# Enable parallelism in the script (assuming it was called with
# `mpirun -np <n_procs>` )
yt.enable_parallelism()

# Load the data
ts = yt.load(
        "/home/oskar/TFG/CuarticPotential_10-4_N64_L18/hdf5/ScalarFieldp*"
        #"/home/oskar/TFG/ScalarField2/hdf5/ScalarFieldp*"
)

storage = {}

# The serial equivalent of piter() here is just "for ds in ts:" .

for store, ds in ts.piter(storage=storage):
    ad = ds.all_data()
    ds.add_field(
        ("chombo", "conffield"),
        function=_conffield,
        sampling_type="local",
    )
    avgchi = ad.quantities.weighted_average_quantity(("chombo", "chi"), "ones")
    scale_factor = 1/avgchi
    conf_time=(scale_factor-1)*h0
    avgconffield = ad.quantities.weighted_standard_deviation(("chombo", "conffield"), "ones")
    store.result = (conf_time, avgconffield[0]**2)#avgchi -1 o -2?
    
arr = np.array(list(storage.values()))
sorting_indices = np.argsort(arr[:, 0])
arr = arr[sorting_indices]
# Plot up the results: time versus variance
plt.plot(arr[:, 0], arr[:, 1])
plt.xlabel("Time")
plt.ylabel("Variance of the conformal field")
plt.savefig("Variance.png")
