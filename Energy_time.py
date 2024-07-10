import matplotlib.pyplot as plt
import numpy as np

import yt
h0 = 1*0.01*1.6916
def _conffield(field,data):
    return(data["chombo", "phi"]*data["chombo","chi"]**(-1/2)/1.6916)

def _potenergy(field,data):
    return(0.25*(data["chombo", "conffield"])**4)
def _kinenergy(field,data):
        return(0.5*(data["chombo", "Pi"])**2*data["chombo", "chi"]**-2/(1.6916)**4/0.0001)
def _pertenergy(field,data):
        hx = data["chombo", "conffield_gradient_x"]
        hy = data["chombo", "conffield_gradient_y"]
        hz = data["chombo", "conffield_gradient_z"]
        return(0.5*(hx*hx+hy*hy+hz*hz)*data.ds.length_unit**2/0.0001/1.6916**2)

# Enable parallelism in the script (assuming it was called with
# `mpirun -np <n_procs>` )
yt.enable_parallelism()

ts = yt.load(
    "/home/oskar/TFG/CuarticPotential_10-4_N64_L18/hdf5/ScalarFieldp*"
    #"/home/oskar/TFG/ScalarField2/hdf5/ScalarFieldp*"
)

storage = {}

for store, ds in ts.piter(storage=storage):
    ad = ds.all_data()
    
    ds.add_field(
        ("chombo", "conffield"),
        function=_conffield,
        sampling_type="local",
        units="dimensionless"
    )
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
    grad_fields = ds.add_gradient_fields(("chombo", "conffield"))
    ds.add_field(
    ("chombo", "pertenergy"),
    function=_pertenergy,
    sampling_type="local",
)
    
    avgchi = ad.quantities.weighted_average_quantity(("chombo", "chi"), "ones")
    avgkin = ad.quantities.weighted_average_quantity(("chombo", "kinenergy"), "ones")
    avgpot = ad.quantities.weighted_average_quantity(("chombo", "potenergy"), "ones")
    avgpert = ad.quantities.weighted_average_quantity(("chombo", "pertenergy"), "ones")
    scale_factor = 1/avgchi
    conf_time=(scale_factor-1)*h0
    store.result = (conf_time, avgkin, avgpot, avgpert, avgkin+avgpot+avgpert)

# Convert the storage dictionary values to a Nx2 array, so the can be easily
# plotted
arr = np.array(list(storage.values()))
sorting_indices = np.argsort(arr[:, 0])
arr = arr[sorting_indices]
# Plot up the results: time versus energy
plt.plot(arr[:, 0], arr[:, 2], label = "Potential Energy")
plt.plot(arr[:, 0], arr[:, 1], label = "Kinetic Energy")
plt.plot(arr[:, 0], arr[:, 3], label = "Energy in Perturbations")
plt.plot(arr[:, 0], arr[:, 4], label = "Total Energy")
plt.yscale("log")
plt.xlabel("Time")
plt.legend(title = 'Energy content', loc='lower right', fontsize='small')
plt.ylabel("Energy content")
plt.savefig("Energy_time.png")
