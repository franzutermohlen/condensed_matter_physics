import numpy as np
from scipy import sparse
# import multiprocessing as mp
# from multiprocessing import Pool
import os
import pickle
import time
import sys
import functions as jkg

# This code uses the Python package Primme when diagonalization_method is set to 'lanczos'
# Instructions on downloading this package can be found here: https://pypi.org/project/primme/
# (Note: This code works correctly with Primme 3.0.3; other versions have not been tested)

# ============================================= Inputs =============================================
# Spin moment at each site
spin = spin0

# Length of system along the two lattice directions
Nx = Nx0
Ny = Ny0

# Coupling constants (units: meV)
# # Heisenberg
# J = -0.212
# # Kitaev
# K = -5.190
# # Gamma
# G = -0.0675

# Coupling constants (units: none)
# Heisenberg
J = J0
# Kitaev
K = K0
# Gamma
G = G0

# External magnetic field magnitude, direction (in local x,y,z coordinates), and units ("none" or "tesla")
# (Note: only use "tesla" if the coupling constants are in units of meV)
h = h_magnitude0
h_units = "none"
h_direction = h_direction0
# h_direction = [1,1,1]

# Boundary conditions; can be either 'periodic', 'antiperiodic', or 'open'
boundary_conditions = 'boundary_conditions0'

# Diagonalization method; can be either 'lanczos' or 'exact'
diagonalization_method = 'method0'

# Maximum number of lowest energy eigenvalues/eigenvectors to find
max_num_evals_to_find = num_evals0

# Required accuracy for eigenvalues
tolerance = 1e-6

# Basis used to represent the spin matrices; can be either 'z' or 'e3'
basis = 'basis0'
# ==================================================================================================


# Start timing this calculation
start_time = time.time()

# Initialize the dictionary in which we will store important data
data_dict = {}

# Add important inputs to data_dict
data_dict['spin'] = spin
data_dict['Nx'] = Nx
data_dict['Ny'] = Ny
data_dict['J'] = J
data_dict['K'] = K
data_dict['G'] = G
data_dict['h'] = h
data_dict['h_units'] = h_units
data_dict['h_direction'] = h_direction
data_dict['boundary_conditions'] = boundary_conditions
data_dict['diagonalization_method'] = diagonalization_method
if diagonalization_method == 'lanczos':
    data_dict['max_num_evals_to_find'] = max_num_evals_to_find
data_dict['tolerance'] = tolerance

data_dict['basis'] = basis

# Dimension of the spin Hilbert space for a single site
dim = int(2*spin + 1)

# Number of sites
num_sites = 2 * Nx * Ny

# Size of Hamiltonian (= dimension of the Hilbert space for the whole system)
ham_size = dim ** num_sites

# If the magnetic field is in teslas, convert it to meV
if h_units == "tesla":
    # Save original value in teslas
    h_tesla = h
    # g-factor
    g_factor = 2
    # Bohr magneton (units: meV/T)
    mu_B = 6 * 1e-5 * 1000
    # Conversion factor from tesla to meV
    tesla_to_meV = g_factor * mu_B
    # Convert magnetic field magnitude to units of meV
    h = h_tesla * tesla_to_meV
    # Add it to data_dict
    data_dict['h_tesla'] = h_tesla

print("Constructing Hamiltonian...")

# Construct Hamiltonian
ham = jkg.JKG_hamiltonian(J, K, G, h, h_direction, spin, Nx, Ny, boundary_conditions, basis)

print("Done.\n")

end_time_ham = time.time()
ham_time = round(end_time_ham - start_time, 8)
print("(Time taken to construct Hamiltonian: %s seconds)\n" % ham_time)

# print(ham.toarray())
# print(ham.nonzero())

# Number of lowest energy eigenvalues/eigenvectors to find
if diagonalization_method == 'lanczos':
    # (Note: If 0.8 * ham_size < max_num_evals_to_find, then set num_evals = 0.8 * ham_size
    #        so that we compute at most 80% of the full spectrum using Lanczos)
    num_evals = min(max_num_evals_to_find, round(0.8*ham_size/2)*2)
elif diagonalization_method == 'exact':
    num_evals = ham_size

# Add it to data_dict
data_dict['num_evals'] = num_evals

if diagonalization_method == 'lanczos':
    print("Finding " + str(num_evals) + " lowest eigenvalues and their eigenvectors...")
elif diagonalization_method == 'exact':
    print("Finding all eigenvalues and their eigenvectors...")

# Find energy eigenvalues and eigenvectors
if diagonalization_method == 'lanczos':
    import primme
    evals, evecs = primme.eigsh(ham, k=num_evals, tol=tolerance, which='SA')
elif diagonalization_method == 'exact':
    from numpy import linalg as LA
    evals, evecs = LA.eig(ham.todense())
    # Modify evals and evecs to have the same format as when we do Lanczos with Primme
    evals = evals.real
    evecs = np.array(evecs)
    # Sort eigenvalues and (reorder the corresponding eigenvectors)
    evecs = evecs[:,np.argsort(evals)]
    evals = np.sort(evals)

print("Done.\n")

# Free up memory stored in Hamiltonian
ham = None

end_time_eigenstates = time.time()
eigenstates_time = round(end_time_eigenstates - end_time_ham, 8)
print("(Time taken to obtain these eigenstates: %s seconds)\n" % eigenstates_time)

# Get rid of infinitesimal differences between the energy eigenvalues
evals = jkg.chop_differences(evals, 1e-10)

# Add evals to data_dict
data_dict['evals'] = evals

# Energy and degeneracy of the ground state and first excited state energy
gs_energy = evals[0]
gs_degeneracy = np.sum(evals == gs_energy)

fexc_energy = evals[gs_degeneracy]
fexc_degeneracy = np.sum(evals == fexc_energy)

# Add these to data_dict
data_dict['gs_energy'] = gs_energy
data_dict['gs_degeneracy'] = gs_degeneracy
data_dict['fexc_energy'] = fexc_energy
data_dict['fexc_degeneracy'] = fexc_degeneracy

# print(fexc_energy - gs_energy)

print("Ground state degeneracy:       ",gs_degeneracy)
print("First excited state degeneracy:",fexc_degeneracy)

print("\nCalculating all single spin matrix elements <0|S_i^alpha|m> between the ground state |0> and all other states |m>...")

spin_matrix_elements_0m = jkg.spin_matrix_elements_between_a_bra_and_many_kets(evecs[:,0], evecs[:,0:], Nx, Ny, basis)

# Add to data_dict
data_dict['spin_matrix_elements_0m'] = spin_matrix_elements_0m

print("Done.\n")

print("Calculating all spin-spin correlations <0|S_i^alpha S_j^beta|m> between the ground state |0> and all other states |m>...")
spin_spin_correlations_0m = jkg.spin_spin_correlations_between_a_bra_and_many_kets(evecs[:,0], evecs[:,0:], Nx, Ny, basis)

# Add to data_dict
data_dict['spin_spin_correlations_0m'] = spin_spin_correlations_0m

print("Done.\n")

# Free up memory stored in the eigenvectors
evecs = None

end_time_total = time.time()
runtime = round(end_time_total - start_time, 8)
print("Total runtime: %s seconds" % runtime)

# Add runtime to data_dict
data_dict['runtime'] = runtime

# ============================= Save data ==============================
print("\nSaving data...")

file_extension = 'pickle'

# Get current directory
current_directory = os.getcwd()

# Local path and filename
local_path = 'local_path0'
filename = jkg.data_filename(num_sites, spin, J, K, G, h, h_direction, file_extension, num_decimal_places_h0, num_evals, boundary_conditions, basis, diagonalization_method)
file = current_directory + '/' + local_path + '/' + filename

# Save data
with open(file, 'wb') as f:
    pickle.dump(data_dict, f)
# ======================================================================

print("All done!")

# sys.exit("Reached breakpoint")
