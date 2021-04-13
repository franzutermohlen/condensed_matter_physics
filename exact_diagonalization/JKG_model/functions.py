# This file contains all of the functions used to perform an exact diagonalization simulation
# of the Heisenberg-Kitaev-Gamma (JKG) model on a honeycomb lattice.
#
# Copyright © 2021 Franz Utermohlen. All rights reserved.

import numpy as np
from scipy import sparse
from scipy.signal import find_peaks

# Note: The '@' operator in these functions only works in Python 3.5 or later.

# =================================================== General useful functions ===================================================

def normalize_numpy_array_by_row(numpy_array, threshold = 1e-10):
    """Function that normalizes a 1D or 2D NumPy array row-by-row (so that each row has a maximum magnitude of 1).
    If the number with highest magnitude in a row is very small (< threshold), then set all values in the row to 0,
    since these nonzero values are likely due to numerical error.
    Input:
        numpy_array (1D or 2D NumPy array of floats)
        threshold (float; if the number with highest magnitude in a row is smaller than threshold, then set all values in the row to 0)
    Outputs:
        numpy_array_normalized (normalized version of the input array)
        normalization_factor (multiplicative factor(s) by which each row in numpy_array was multiplied)"""

    # Make a copy of numpy_array (so that this function doesn't overwrite the original numpy_array that was input)
    numpy_array_normalized = numpy_array.copy()

    # Shape of the NumPy array
    array_shape = numpy_array_normalized.shape

    # Array dimension
    dim = len(array_shape)

    # Check if the array has a trivial dimension (and if it does, fix the array shape)
    if (dim == 2) and (min(array_shape) == 1):
        dim = 1
        array_shape = max(array_shape)
        numpy_array_normalized = numpy_array_normalized.reshape(array_shape)

    # Get the number of rows in the array
    if dim == 1:
        num_rows = 1
    elif dim == 2:
        num_rows = array_shape[0]
    else:
        print("Error: Incorrect input array dimension; must be a 1D or 2D NumPy array.")
        return -1

    # Normalize the array
    if num_rows == 1:
        max_val = max(np.abs(numpy_array_normalized))
        if max_val >= threshold:
            normalization_factor = 1./max_val
        else:
            normalization_factor = 0
        numpy_array_normalized *= normalization_factor
    else:
        normalization_factor = np.full(num_rows, 0.)
        # Loop over rows
        for row in range(num_rows):
            max_val = max(np.abs(numpy_array_normalized[row]))
            if max_val >= threshold:
                normalization_factor[row] = 1./max_val
            else:
                normalization_factor[row] = 0
            numpy_array_normalized[row] *= normalization_factor[row]

    return numpy_array_normalized, normalization_factor

def chop(expression, threshold = 1e-10):
    """Function that replaces a number by 0 if its magnitude is very small (<= threshold). If the number is complex,
    the real and imaginary components are each looked at separately.
    Input:
        expression (number (float or complex), list of numbers, or NumPy N-dimensional array of numbers)
        threshold (float)
    Outputs:
        expression_modified (modified version of expression;
                             if a list was input, it gets converted to a NumPy array)
    Note: This function is named "chop" because it is modeled loosely after Mathematica's "Chop" function."""

    # Turn expression into a NumPy array
    expression_modified = np.atleast_1d(expression)

    # Number of elements in expression_modified
    n_elements = np.size(expression)

    # Check if any element in expression has a non-zero imaginary component
    if np.max(np.abs(np.imag(expression_modified))) > 0:
        contains_imaginary = True
    else:
        contains_imaginary = False

    if contains_imaginary == False:

        # Indices of the elements in expression_modified whose magnitudes are <= threshold
        chop_indices = (np.abs(expression_modified) <= threshold).nonzero()

        # Set these elements to 0
        expression_modified[chop_indices] = 0

    elif contains_imaginary == True:

        # Split expression_modified into its real and imaginary components
        expression_modified_real = expression_modified.real
        expression_modified_imag = expression_modified.imag

        # Indices of the real and imaginary elements in expression_modified_real and expression_modified_imag
        # whose magnitudes are <= threshold
        chop_indices_real = (np.abs(expression_modified_real) <= threshold).nonzero()
        chop_indices_imag = (np.abs(expression_modified_imag) <= threshold).nonzero()

        # Combine the real and imaginary components back into expression_modified
        expression_modified = expression_modified_real + 1j*expression_modified_imag

    # If expression is a single number, convert expression_modified from array format to number format
    if n_elements == 1:
        expression_modified = expression_modified.item()

    return expression_modified

def chop_differences(list_of_numbers, threshold = 1e-10):
    """Function that gets rid of infinitesimal differences (<= threshold) between the values of the elements in a list.
    Input:
        list_of_numbers (list of numbers (float or complex) or 1D NumPy array of numbers)
        threshold (float; if the magnitude of the difference of any two elements is smaller than threshold, then
                          set one of them equal to the other one)
    Outputs:
        list_of_numbers_modified (modified version of list_of_numbers)
    Note 1: If two or more elements in list_of_numbers are less than threshold apart, the value of the one with
            the lowest index in the list is assigned to the rest of them.
    Note 2: This function is named "chop_differences" because it uses the "chop" function, which is modeled loosely after
            Mathematica's "Chop" function."""

    # Make a copy of list_of_numbers (so that this function doesn't overwrite the original list_of_numbers that was input)
    list_of_numbers_modified = list_of_numbers.copy()

    for i in range(len(list_of_numbers_modified)):
        list_of_numbers_modified = chop(list_of_numbers_modified - list_of_numbers_modified[i], threshold) + list_of_numbers_modified[i]

    return list_of_numbers_modified

def string_to_float(string):
    """Function that converts a string to a float.
    Input:
        string (string representing a number)
    Output:
        value (float represented by the input string)
    Note: This function can handle strings representing fractions (e.g., '1/2' gets converted to 0.5)."""

    try:
        value = float(string)
    except ValueError:
        numerator, denominator = string.split('/')
        value = float(numerator) / float(denominator)

    return value

# ================================================================================================================================



# ================================================= Functions for the JKG model ==================================================

def kronecker_product(matrix_list):
    """Function that returns the Kronecker product of the matrices input in matrix_list, where
    matrix_list is a 1D NumPy array of the form matrix_list = np.array([mat_1, mat_2, mat_3, ...])
    and mat_m (m=1,2,3,...) are sparse matrices."""

    product = sparse.kron(matrix_list[0],matrix_list[1], 'csr')

    if len(matrix_list) > 2:
        for mat_index in range(2, len(matrix_list)):
            product = sparse.kron(product, matrix_list[mat_index], 'csr')

    return product

def lorentzian(x, x0, eta):
    """Gives the value of the Lorentzian distribution function
        L(x; x0, eta) = eta / [(x-x0)^2 + eta^2]
    at a position x, where
        x0 is the mean of the Lorentzian distribution,
        eta is the half-width at half-maximum.

    When eta is very small, this gives the Dirac delta function:
        L(x; x0, eta -> 0+) = delta(x-x0)"""

    value = (1./np.pi) * eta / ((x - x0)**2 + eta**2)

    return value

def normalize_and_find_peaks_and_get_gap(spectral_data_x_values, spectral_data_y_values, prominence=0.01, threshold = 1e-10, y_values_can_be_negative=False):
    """Function that normalizes spectral data (so that the maximum magnitude of the y values is 1),
    finds the x values of the spectral peaks, and gets the size of the gap in the spectrum (i.e., the x value of the first peak).
    Inputs:
        spectral_data_x_values, spectral_data_y_values (1D NumPy arrays of equal length specifying the x and y values of the spectral data)
        prominence (float specifying the minimum prominence necessary to classify something as a peak)
        threshold (float; if the number in spectral_data_y_values with highest magnitude is smaller than threshold, then normalize spectral_data_y_values to 0)
        y_values_can_be_negative (Boolean specifying whether to consider the scenario where the y values can be negative (e.g., negative peaks))
    Outputs:
        spectral_data_y_values_normalized (version of the spectral_data_y_values array normalized to having a peak of magnitude 1)
        peak_indices (1D NumPy array containing the indices corresponding to the peaks in spectral_data_y_values)
        peak_x_values (1D NumPy array containing the x values of the peaks)
        spectral_gap (float specifying the value of the gap in the spectrum, i.e. the x value of the first peak)"""

    # Normalize spectral data
    spectral_data_y_values_normalized, _ = normalize_numpy_array_by_row(spectral_data_y_values, threshold=threshold)

    # Find the x values of the peaks
    peak_indices, _ = find_peaks(spectral_data_y_values_normalized, prominence=prominence)
    peak_x_values = spectral_data_x_values[peak_indices]

    if y_values_can_be_negative == True:
        # Find the x values of the negative peaks
        negative_peak_indices, _ = find_peaks(-spectral_data_y_values_normalized, prominence=0.01)
        negative_peak_x_values = spectral_data_x_values[negative_peak_indices]

        if len(negative_peak_x_values) > 0:
            # Make a mask of the "nontrivial" negative peaks, defined as those that occur as a negative y value
            negative_peak_indices_nontrivial = (chop(spectral_data_y_values[negative_peak_indices]) < 0)

            # Only keep the "nontrivial" negative peaks
            negative_peak_indices = negative_peak_indices[negative_peak_indices_nontrivial].flatten()
            negative_peak_x_values = negative_peak_x_values[negative_peak_indices_nontrivial].flatten()

            # Combine the arrays with positive peaks and negative peaks
            peak_indices = np.concatenate((peak_indices,negative_peak_indices))
            peak_x_values = np.concatenate((peak_x_values,negative_peak_x_values))

            if len(peak_x_values) > 1:
                # Sort peaks from lowest to highest x values
                peak_sort_indices = np.argsort(peak_x_values)
                peak_x_values = peak_x_values[peak_sort_indices]
                peak_indices = peak_indices[peak_sort_indices]

    # Get the value of the gap in the spectrum (the x value of the first peak)
    if len(peak_x_values) > 0:
        spectral_gap = peak_x_values[0]
    else:
        spectral_gap = None

    return spectral_data_y_values_normalized, peak_indices, peak_x_values, spectral_gap

def momentum_vectors_probing_a_rectangular_region_uniformly(kx_min, kx_max, ky_min, ky_max, num_momentum_vectors_along_an_axis_direction = 50):
    """Function that constructs momentum space vectors in e1,e2 coordinates spaced uniformly throughout the rectangular region
    in momentum space bounded by [kx_min,kx_max] and [ky_min,ky_max].
    Input:
        kx_min, kx_max (floats specifying the minimum and maximum values of the x-component of the momentum (kx) to probe)
        ky_min, ky_max (floats specifying the minimum and maximum values of the y-component of the momentum (ky) to probe)
        num_momentum_vectors_along_a_lattice_direction (integer specifying the number of momentum space vectors to construct
                                                        along a lattice direction)
    Output:
        momentum_vectors (2D NumPy array of shape N_k x 2, where N_k = (num_momentum_vectors_along_a_lattice_direction+1)^2 is
                          the total number of momentum space vectors created;
                          momentum_vectors[:,0] contains the e1 coordinates of the vectors, and
                          momentum_vectors[:,1] contains the e2 coordinates of the vectors)"""

    # Total number of momentum space vectors to probe
    num_momentum_vectors_total = (num_momentum_vectors_along_an_axis_direction+1) ** 2

    # Range of kx and ky values
    delta_kx = kx_max - kx_min
    delta_ky = ky_max - ky_min

    # e1 and e2 unit vectors
    e1 = np.array([1,0])
    e2 = np.array([0,1])

    # Initialize NumPy array that will contain the momentum space vectors
    momentum_vectors = np.full((num_momentum_vectors_total,2), 0.)

    # Loop over the two axis directions
    index = 0
    for i in range(num_momentum_vectors_along_an_axis_direction+1):
        for j in range(num_momentum_vectors_along_an_axis_direction+1):
            momentum_vectors[index] = (kx_min + (i*delta_kx/num_momentum_vectors_along_an_axis_direction)) * e1 + \
                                      (ky_min + (j*delta_ky/num_momentum_vectors_along_an_axis_direction)) * e2

            # Update index
            index += 1

    return momentum_vectors

def momentum_vectors_probing_the_first_Brillouin_zone_uniformly(num_momentum_vectors_along_a_lattice_direction = 50):
    """Function that constructs momentum space vectors in e1,e2 coordinates spaced uniformly throughout the first Brillouin zone
    of the system's honeycomb lattice.
    Input:
        num_momentum_vectors_along_a_lattice_direction (integer specifying the number of momentum space vectors to construct
                                                        along a lattice direction)
    Output:
        momentum_vectors (2D NumPy array of shape N_k x 2, where N_k = num_momentum_vectors_along_a_lattice_direction^2 is
                          the total number of momentum space vectors created;
                          momentum_vectors[:,0] contains the e1 coordinates of the vectors, and
                          momentum_vectors[:,1] contains the e2 coordinates of the vectors)"""

    # Total number of momentum space vectors to probe
    num_momentum_vectors_total = num_momentum_vectors_along_a_lattice_direction ** 2

    # Construct relevant lattice and reciprocal lattice vectors in e1,e2 coordinates
    # (Note: We will be neglecting the e3 component, which is trivially 0 for all of them)
    # # Nearest-neighbor vectors (going from sublattice A sites to neighboring sublattice B sites)
    # dx = 1/2 * np.array([-np.sqrt(3), -1])
    # dy = 1/2 * np.array([ np.sqrt(3), -1])
    # dz = np.array([0, 1])
    # # Lattice vectors
    # a1 = dy - dx
    # a2 = dz - dx
    # Momentum space (reciprocal space) lattice vectors
    b1 = 2*np.pi/3 * np.array([np.sqrt(3),-1])
    b2 = 4*np.pi/3 * np.array([0,1])
    # # Reciprocal space high-symmetry points
    # Gamma_point = np.array([0,0])
    # K_point = 4*np.pi/(3*np.sqrt(3)) * np.array([1,0])
    # M_point = np.pi/3 * np.array([np.sqrt(3),1])

    # Initialize NumPy array that will contain the momentum space vectors
    momentum_vectors = np.full((num_momentum_vectors_total,2), 0.)

    # Loop over the two lattice directions
    index = 0
    for i in range(num_momentum_vectors_along_a_lattice_direction):
        for j in range(num_momentum_vectors_along_a_lattice_direction):
            momentum_vectors[index] = (i / num_momentum_vectors_along_a_lattice_direction) * b1 + \
                                      (j / num_momentum_vectors_along_a_lattice_direction) * b2

            # Update index
            index += 1

    return momentum_vectors

def e1e2e3_unit_vectors_in_terms_of_xyz_coordinates():
    """Function that returns the lab coordinate unit vectors e1,e2,e3 in terms of their crystal coordinates x,y,z, given by
        e1 = 1/sqrt(6) * (- x - y + 2z)
        e2 = 1/sqrt(2) * (x - y)
        e3 = 1/sqrt(3) * (x + y + z)
    where e1,e2 lie in the honeycomb plane and e3 is perpendicular to the honeycomb plane.
    Input:
        None
    Output:
        e1, e2, e3 (3-element 1D NumPy arrays containing the crystal coordinates x,y,z of each unit vector)"""

    # Define matrix that rotates the crystal coordinates x,y,z into the lab coordinates e1,e2,e3:
    # [e1,e2,e3] = rotation_matrix.[x,y,z]
    rotation_matrix = np.array([[-1/np.sqrt(6), -1/np.sqrt(6), 2/np.sqrt(6)],
                                [ 1/np.sqrt(2), -1/np.sqrt(2), 0           ],
                                [ 1/np.sqrt(3),  1/np.sqrt(3), 1/np.sqrt(3)]])

    # Compute the e1,e2,e3 lab coordinate unit vectors in terms of the crystal coordinates x,y,z
    [e1, e2, e3] = rotation_matrix @ np.identity(3)

    return e1, e2, e3

def xyz_unit_vectors_in_terms_of_e1e2e3_coordinates():
    """Function that returns the crystal coordinate unit vectors x,y,z in terms of their lab coordinates e1,e2,e3, given by
        x = (- e1/sqrt(6) + e2/sqrt(2) + e3/sqrt(3))
        y = (- e1/sqrt(6) - e2/sqrt(2) + e3/sqrt(3))
        z = (e1*sqrt(2/3) + e3/sqrt(3))
    where e1,e2 lie in the honeycomb plane and e3 is perpendicular to the honeycomb plane.
    Input:
        None
    Output:
        x, y, z (3-element 1D NumPy arrays containing the lab coordinates e1,e2,e3 of each unit vector)"""

    # Define matrix that rotates the crystal coordinates x,y,z into the lab coordinates e1,e2,e3:
    # [e1,e2,e3] = rotation_matrix @ [x,y,z], so [x,y,z] = rotation_matrix.T @ [e1,e2,e3]
    rotation_matrix = np.array([[-1/np.sqrt(6), -1/np.sqrt(6), 2/np.sqrt(6)],
                                [ 1/np.sqrt(2), -1/np.sqrt(2), 0           ],
                                [ 1/np.sqrt(3),  1/np.sqrt(3), 1/np.sqrt(3)]])

    # Compute the crystal coordinate x,y,z in terms of the lab coordinates e1,e2,e3
    [x, y, z] = rotation_matrix.T @ np.identity(3)

    return x, y, z

def spin_matrices(spin, basis = 'z'):
    """Function that constructs the following matrices for a particle with the specified spin quantum number:
        S^x, S^y, S^z:      spin matrices along the crystal coordinates x,y,z,
        S^{+_z}, S^{-_z}:   raising and lowering spin matrices along the z axis,
        S^e1, S^e2, S^e3:   spin matrices along the lab coordinates e1,e2,e3,
        S^{+_e3}, S^{-_e3}: raising and lowering spin matrices along the e3 axis,
        I:                  identity matrix,
    for a particle with the specified spin quantum number.
    Inputs:
        spin (spin quantum number for which to construct the spin matrices; can be either 1/2, 1, 3/2, or 2)
        basis (string specifying which basis is being used to represent the spin matrices; can be either 'z' or 'e3'.
               For example, for spin = 1/2:
               if basis = 'z', then Sx,Sy,Sz will be the Pauli spin matrices sigma_1,sigma_2,sigma_3.
               if basis = 'e3', then Se1,Se2,Se3 will be the Pauli spin matrices sigma_1,sigma_2,sigma_3.)
    Outputs:
        Sx, Sy, Sz, Sp_z, Sm_z, Se1, Se2, Se3, Sp_e3, Sm_e3, identity (spin matrices and identity matrix
                                                                       as SciPy csr (sparse) matrices;
                                                                       Sp_z,Sm_z are spin raising and lowering matrices along the z axis, and
                                                                       Sp_e3,Sm_e3 are spin raising and lowering matrices along the e3 axis)"""

    if basis == 'z':

        # Define spin matrices (in dense form)
        if spin == 1/2:
            Sx_dense = 1/2 * np.array([[0, 1],
                                       [1, 0]])
            Sy_dense = 1/2 * np.array([[0 , -1j],
                                       [1j,  0 ]])
            Sz_dense = 1/2 * np.array([[1,  0],
                                       [0, -1]])
        elif spin == 1:
            Sx_dense = 1/np.sqrt(2) * np.array([[0, 1, 0],
                                                [1, 0, 1],
                                                [0, 1, 0]])
            Sy_dense = 1/np.sqrt(2) * np.array([[0 , -1j,  0 ],
                                                [1j,  0 , -1j],
                                                [0 ,  1j,  0 ]])
            Sz_dense = np.array([[1, 0,  0],
                                 [0, 0,  0],
                                 [0, 0, -1]])
        elif spin == 3/2:
            Sx_dense = 1/2 * np.array([[        0 , np.sqrt(3),         0 ,         0 ],
                                       [np.sqrt(3),         0 ,         2 ,         0 ],
                                       [        0 ,         2 ,         0 , np.sqrt(3)],
                                       [        0 ,         0 , np.sqrt(3),         0 ]])
            Sy_dense = (1/(2j)) * np.array([[         0 , np.sqrt(3),          0 ,         0 ],
                                            [-np.sqrt(3),         0 ,          2 ,         0 ],
                                            [         0 ,        -2 ,          0 , np.sqrt(3)],
                                            [         0 ,         0 , -np.sqrt(3),         0 ]])
            Sz_dense = 1/2 * np.array([[3, 0,  0,  0],
                                       [0, 1,  0,  0],
                                       [0, 0, -1,  0],
                                       [0, 0,  0, -3]])
        elif spin == 2:
            Sx_dense = 1/2 * np.array([[0,         2 ,         0 ,         0 , 0],
                                       [2,         0 , np.sqrt(6),         0 , 0],
                                       [0, np.sqrt(6),         0 , np.sqrt(6), 0],
                                       [0,         0 , np.sqrt(6),         0 , 2],
                                       [0,         0 ,         0 ,         2 , 0]])
            Sy_dense = 1/2 * np.array([[0 ,        -2j   ,          0    ,          0    ,  0 ],
                                       [2j,         0    , -np.sqrt(6)*1j,          0    ,  0 ],
                                       [0 , np.sqrt(6)*1j,          0    , -np.sqrt(6)*1j,  0 ],
                                       [0 ,         0    ,  np.sqrt(6)*1j,          0    , -2j],
                                       [0 ,         0    ,          0    ,          2j   ,  0 ]])
            Sz_dense = np.array([[2, 0, 0,  0,  0],
                                 [0, 1, 0,  0,  0],
                                 [0, 0, 0,  0,  0],
                                 [0, 0, 0, -1,  0],
                                 [0, 0, 0,  0, -2]])

        # Sparse versions of the spin matrices
        Sx = sparse.csr_matrix(Sx_dense)
        Sy = sparse.csr_matrix(Sy_dense)
        Sz = sparse.csr_matrix(Sz_dense)

        # Lab coordinate unit vectors e1,e2,e3 in terms of their crystal coordinates x,y,z
        e1, e2, e3 = e1e2e3_unit_vectors_in_terms_of_xyz_coordinates()

        # Spin matrices along the e1,e2,e3 directions
        Se1 = e1[0]*Sx + e1[1]*Sy + e1[2]*Sz
        Se2 = e2[0]*Sx + e2[1]*Sy + e2[2]*Sz
        Se3 = e3[0]*Sx + e3[1]*Sy + e3[2]*Sz

        # Note: In calculating Se1 above (for example), we are essentially doing
        # Se1 = e1 @ [Sx,Sy,Sz]
        # However, this only works in some versions of Python.

    elif basis == 'e3':

        # Define spin matrices (in dense form)
        if spin == 1/2:
            Se1_dense = 1/2 * np.array([[0, 1],
                                        [1, 0]])
            Se2_dense = 1/2 * np.array([[0 , -1j],
                                        [1j,  0 ]])
            Se3_dense = 1/2 * np.array([[1,  0],
                                        [0, -1]])
        elif spin == 1:
            Se1_dense = 1/np.sqrt(2) * np.array([[0, 1, 0],
                                                 [1, 0, 1],
                                                 [0, 1, 0]])
            Se2_dense = 1/np.sqrt(2) * np.array([[0 , -1j,  0 ],
                                                 [1j,  0 , -1j],
                                                 [0 ,  1j,  0 ]])
            Se3_dense = np.array([[1, 0,  0],
                                  [0, 0,  0],
                                  [0, 0, -1]])
        elif spin == 3/2:
            Se1_dense = 1/2 * np.array([[        0 , np.sqrt(3),         0 ,         0 ],
                                        [np.sqrt(3),         0 ,         2 ,         0 ],
                                        [        0 ,         2 ,         0 , np.sqrt(3)],
                                        [        0 ,         0 , np.sqrt(3),         0 ]])
            Se2_dense = (1/(2j)) * np.array([[         0 , np.sqrt(3),          0 ,         0 ],
                                             [-np.sqrt(3),         0 ,          2 ,         0 ],
                                             [         0 ,        -2 ,          0 , np.sqrt(3)],
                                             [         0 ,         0 , -np.sqrt(3),         0 ]])
            Se3_dense = 1/2 * np.array([[3, 0,  0,  0],
                                        [0, 1,  0,  0],
                                        [0, 0, -1,  0],
                                        [0, 0,  0, -3]])
        elif spin == 2:
            Se1_dense = 1/2 * np.array([[0,         2 ,         0 ,         0 , 0],
                                        [2,         0 , np.sqrt(6),         0 , 0],
                                        [0, np.sqrt(6),         0 , np.sqrt(6), 0],
                                        [0,         0 , np.sqrt(6),         0 , 2],
                                        [0,         0 ,         0 ,         2 , 0]])
            Se2_dense = 1/2 * np.array([[0 ,        -2j   ,          0    ,          0    ,  0 ],
                                        [2j,         0    , -np.sqrt(6)*1j,          0    ,  0 ],
                                        [0 , np.sqrt(6)*1j,          0    , -np.sqrt(6)*1j,  0 ],
                                        [0 ,         0    ,  np.sqrt(6)*1j,          0    , -2j],
                                        [0 ,         0    ,          0    ,          2j   ,  0 ]])
            Se3_dense = np.array([[2, 0, 0,  0,  0],
                                  [0, 1, 0,  0,  0],
                                  [0, 0, 0,  0,  0],
                                  [0, 0, 0, -1,  0],
                                  [0, 0, 0,  0, -2]])

        # Sparse versions of the spin matrices
        Se1 = sparse.csr_matrix(Se1_dense)
        Se2 = sparse.csr_matrix(Se2_dense)
        Se3 = sparse.csr_matrix(Se3_dense)

        # x,y,z coordinate unit vectors in terms of their e1,e2,e3 coordinates
        x, y, z = xyz_unit_vectors_in_terms_of_e1e2e3_coordinates()

        # Spin matrices along the x,y,z directions
        Sx = x[0]*Se1 + x[1]*Se2 + x[2]*Se3
        Sy = y[0]*Se1 + y[1]*Se2 + y[2]*Se3
        Sz = z[0]*Se1 + z[1]*Se2 + z[2]*Se3

        # Note: In calculating Sx above (for example), we are essentially doing
        # Sx = x @ [Se1,Se2,Se3]
        # However, this only works in some versions of Python.

    # Raising and lowering spin operators S+ and S- along the z axis
    Sp_z = Sx + 1j*Sy
    Sm_z = Sx - 1j*Sy

    # Raising and lowering spin operators S+ and S- along the e3 axis
    Sp_e3 = Se1 + 1j*Se2
    Sm_e3 = Se1 - 1j*Se2

    # Size of the spin Hilbert space for a single spin
    dim = int(2*spin + 1)

    # Identity matrix and its sparse version
    identity_dense = np.identity(dim)
    identity = sparse.csr_matrix(identity_dense)

    return Sx, Sy, Sz, Sp_z, Sm_z, Se1, Se2, Se3, Sp_e3, Sm_e3, identity

def get_site_index(Nx, i, j, sublattice):
    """Function that returns the unique site index corresponding to the
    (i,j,sublattice) site in an (Nx * Ny * 2) honeycomb lattice.
    i goes from 0 to Nx-1; j goes from 0 to Ny-1; sublattice is either 0 or 1.
    Each site index is a number from 0 to num_sites-1, where num_sites = 2 * Nx * Ny.
    Sublattice A/B sites are indexed by an even/odd number."""

    # If the specified i index is < 0 or > Nx-1, get the corresponding index between 0 and Nx-1
    i = i % Nx

    # Get site index
    site_index = 2 * (i + j*Nx) + sublattice

    return site_index

def JKG_hamiltonian(J, K, G, h, h_direction, spin, Nx, Ny, boundary_conditions = 'periodic', basis = 'z'):
    """Function that constructs the JKG (Heisenberg + Kitaev + symmetric off-diagonal) Hamiltonian on a honeycomb lattice
    in an external magnetic field, namely
        H_JKG = Sum_{<ij> in alpha beta (gamma)} [ J S_i.S_j + K S_i^gamma S_j^gamma + G (S_i^alpha S_j^beta + S_i^beta S_j^alpha) ] - Sum_i h.S_i
    where Sum_{<ij> in alpha beta (gamma)} denotes a sum over neighboring sites interacting through a gamma-bond
    (gamma=x,y,z, and alpha,beta are the other two directions), and
    S_i.S_j denotes a dot product between the spins on sites i and j.
    Inputs:
        J (Heisenberg coupling)
        K (Kitaev coupling)
        G (Gamma coupling)
        h (magnitude of external magnetic field;
           if negative, it also reverses the direction of the magnetic field (h_direction))
        h_direction (direction of external magnetic field as a list)
        spin (spin quantum number at each site; can be either 1/2, 1, 3/2, or 2)
        Nx, Ny (length of system along the two lattice directions)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
        basis (string specifying which basis is being used to represent the spin matrices; can be either 'z' or 'e3'.
               For example, for spin = 1/2:
               if basis = 'z', then Sx,Sy,Sz will be the Pauli spin matrices sigma_1,sigma_2,sigma_3.
               if basis = 'e3', then Se1,Se2,Se3 will be the Pauli spin matrices sigma_1,sigma_2,sigma_3.)
    Output:
        hamiltonian (SciPy csr (sparse) matrix of size D = dim^num_sites, where dim = 2*spin + 1. and num_sites = 2 Nx Ny)"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Make h_direction a NumPy array and normalize it
    h_direction = np.array(h_direction)
    h_direction = h_direction / np.linalg.norm(h_direction)

    # Construct the spin matrices and the identity matrix
    Sx, Sy, Sz, _, _, _, _, _, _, _, identity = spin_matrices(spin, basis)

    # Make a 1D NumPy array of the three spin matrices
    S = np.array([Sx, Sy, Sz])

    # Make a 1D NumPy array of num_sites dim-by-dim sparse identity matrices, where dim = 2*spin + 1.
    # ((For example, for spin=1/2 and num_sites=4, matrix_list is a NumPy array of four 2x2 sparse identity matrices.))
    matrix_list = np.full(num_sites, identity)

    # Initialize Hamiltonian matrix
    hamiltonian = 0 * kronecker_product(matrix_list)

    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site index of the sublattice B site in this unit cell
            site_index = get_site_index(Nx, i, j, 1)

            # ============ Add bond interaction terms to the Hamiltonian ============
            # Loop over the 3 bonds for the B site of this unit cell
            # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
            for bond_type in range(3):

                # Get the other two bond types (e.g., if bond_type = 0, then bond_type_other_1 = 1 and bond_type_other_2 = 0)
                bond_type_other_1 = (bond_type + 1) % 3
                bond_type_other_2 = (bond_type + 2) % 3

                # Initialize Boolean that specifies whether or not this bond is a boundary bond
                # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
                boundary_bond = False

                # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
                if bond_type == 0:
                    i_neighbor = i
                    j_neighbor = (j + 1) % Ny
                    if j_neighbor != (j + 1):
                        boundary_bond = True
                elif bond_type == 1:
                    i_neighbor = (i - 1) % Nx
                    j_neighbor = (j + 1) % Ny
                    if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                        boundary_bond = True
                elif bond_type == 2:
                    i_neighbor = i
                    j_neighbor = j

                # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
                # and this is a boundary bond
                if (boundary_conditions == 'open') and (boundary_bond == True):
                    continue

                # Factor by which to multiply each spin-spin interaction in this bond in order to implement antiperiodic boundary conditions,
                # if specified; this will be -1 for antiperiodic boundary bonds, and +1 for all other bonds
                if (boundary_conditions == 'antiperiodic') and (boundary_bond == True):
                    bond_sign = -1
                else:
                    bond_sign = 1

                # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
                site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

                # Add the interaction terms for this bond to the Hamiltonian
                # Heisenberg
                if J != 0:
                    matrix_list[site_index] = S[0]
                    matrix_list[site_index_neighbor] = S[0]
                    hamiltonian += J * kronecker_product(matrix_list) * bond_sign

                    matrix_list[site_index] = S[1]
                    matrix_list[site_index_neighbor] = S[1]
                    hamiltonian += J * kronecker_product(matrix_list) * bond_sign

                    matrix_list[site_index] = S[2]
                    matrix_list[site_index_neighbor] = S[2]
                    hamiltonian += J * kronecker_product(matrix_list) * bond_sign
                # Kitaev
                if K != 0:
                    matrix_list[site_index] = S[bond_type]
                    matrix_list[site_index_neighbor] = S[bond_type]
                    hamiltonian += K * kronecker_product(matrix_list) * bond_sign
                # Gamma
                if G != 0:
                    matrix_list[site_index]          = S[bond_type_other_1]
                    matrix_list[site_index_neighbor] = S[bond_type_other_2]
                    hamiltonian += G * kronecker_product(matrix_list) * bond_sign

                    matrix_list[site_index]          = S[bond_type_other_2]
                    matrix_list[site_index_neighbor] = S[bond_type_other_1]
                    hamiltonian += G * kronecker_product(matrix_list) * bond_sign

                # Reset matrix_list
                matrix_list[site_index] = identity
                matrix_list[site_index_neighbor] = identity

            # ================= Add Zeeman terms to the Hamiltonian =================
            if h != 0:
                # Add the Zeeman term for the B site in this unit cell to the Hamiltonian
                matrix_list[site_index] = h_direction[0] * S[0] + h_direction[1] * S[1] + h_direction[2] * S[2]
                hamiltonian -= h * kronecker_product(matrix_list)

                # Reset matrix_list
                matrix_list[site_index] = identity

                # Site index of the other site in this unit cell (i.e., the sublattice A site)
                site_index_other_site = get_site_index(Nx, i, j, 0)

                # Add the Zeeman term for the A site in this unit cell to the Hamiltonian
                matrix_list[site_index_other_site] = h_direction[0] * S[0] + h_direction[1] * S[1] + h_direction[2] * S[2]
                hamiltonian -= h * kronecker_product(matrix_list)

                # Reset matrix_list
                matrix_list[site_index_other_site] = identity

                # Note: The line below is equivalent to the line below it
                # matrix_list[site_index] = h_direction[0] * S[0] + h_direction[1] * S[1] + h_direction[2] * S[2]
                # matrix_list[site_index] = h_direction @ S
                # However, this last line only works in some versions of Python.

    return hamiltonian

# def JKG_raman_operator(J, K, G, h, h_direction, spin, Nx, Ny, epsilon_in, epsilon_out, boundary_conditions = 'periodic', basis = 'z'):
#     """Function that constructs the Raman operator for a superexchange-mediated honeycomb system of
#     magnetic metal ions with edge-sharing ligand octahedra.
#     Inputs:
#         J (Heisenberg coupling)
#         K (Kitaev coupling)
#         G (Gamma coupling)
#         h (magnitude of external magnetic field;
#            if negative, it also reverses the direction of the magnetic field (h_direction))
#         h_direction (list specifying the direction of external magnetic field)
#         spin (spin quantum number at each site; can be either 1/2, 1, 3/2, or 2)
#         Nx, Ny (length of system along the two lattice directions)
#         epsilon_in, epsilon_out (lists specifying the directions of the incoming and outgoing photon polarizations in x,y,z coordinates)
#         boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
#         basis (string specifying which basis is being used to represent the spin matrices; can be either 'z' or 'e3'.
#                For example, for spin = 1/2:
#                if basis = 'z', then Sx,Sy,Sz will be the Pauli spin matrices sigma_1,sigma_2,sigma_3.
#                if basis = 'e3', then Se1,Se2,Se3 will be the Pauli spin matrices sigma_1,sigma_2,sigma_3.)
#     Output:
#         raman_operator (SciPy csr (sparse) matrix of size D = dim^num_sites, where dim = 2*spin + 1. and num_sites = 2 Nx Ny)"""
#
#     # Number of sites
#     num_sites = 2 * Nx * Ny
#
#     # Make h_direction, epsilon_in, and epsilon_out NumPy arrays and normalize them
#     h_direction = np.array(h_direction)
#     h_direction = h_direction / np.linalg.norm(h_direction)
#
#     epsilon_in = np.array(epsilon_in)
#     epsilon_in = epsilon_in / np.linalg.norm(epsilon_in)
#
#     epsilon_out = np.array(epsilon_out)
#     epsilon_out = epsilon_out / np.linalg.norm(epsilon_out)
#
#     # Construct the spin matrices and the identity matrix
#     Sx, Sy, Sz, _, _, _, _, _, _, _, identity = spin_matrices(spin, basis)
#
#     # Make a 1D NumPy array of the three spin matrices
#     S = np.array([Sx, Sy, Sz])
#
# #     # Define matrix that rotates the crystal coordinates x,y,z into the lab coordinates e1,e2,e3:
# #     # [e1,e2,e3] = rotation_matrix.[x,y,z]
# #     rotation_matrix = np.array([[-1/np.sqrt(6), -1/np.sqrt(6), 2/np.sqrt(6)],
# #                                 [ 1/np.sqrt(2), -1/np.sqrt(2), 0           ],
# #                                 [ 1/np.sqrt(3),  1/np.sqrt(3), 1/np.sqrt(3)]])
#
# #     # e1,e2,e3 lab coordinate unit vectors in terms of the crystal coordinates x,y,z
# #     [e1, e2, e3] = rotation_matrix @ np.identity(3)
#
# #     # Lattice vectors (going from sublattice A sites to neighboring sublattice B sites) in e1,e2,e3 coordinates
# #     dx_e1e2e3 = 1/2 * np.array([-np.sqrt(3), -1, 0])
# #     dy_e1e2e3 = 1/2 * np.array([ np.sqrt(3), -1, 0])
# #     dz_e1e2e3 = np.array([0, 1, 0])
# #     # Rewrite them in crystal coordinates x,y,z
# #     [dx, dy, dz] = rotation_matrix.T @ [dx_e1e2e3, dy_e1e2e3, dz_e1e2e3]
#
# #     # Ligand vectors (going from a ligand to another neighboring ligand) in e1,e2,e3 coordinates
# #     fx_e1e2e3 = 1/(2*np.sqrt(3)) * np.array([1, -np.sqrt(3), 2*np.sqrt(2)])
# #     fy_e1e2e3 = 1/(2*np.sqrt(3)) * np.array([1,  np.sqrt(3), 2*np.sqrt(2)])
# #     fz_e1e2e3 = 1/np.sqrt(3) * np.array([-1, 0, np.sqrt(2)])
# #     # Rewrite them in crystal coordinates x,y,z
# #     [fx, fy, fz] = rotation_matrix.T @ [fx_e1e2e3, fy_e1e2e3, fz_e1e2e3]
#
#     # Make a 1D NumPy array of num_sites dim-by-dim sparse identity matrices.
#     # (For example, for spin=1/2 and num_sites=4, matrix_list is a NumPy array of four 2x2 sparse identity matrices.)
#     matrix_list = np.full(num_sites, identity)
#
#     # Initialize Raman operator
#     raman_operator = 0 * kronecker_product(matrix_list)
#
#     # Loop over all the unit cells, which are indexed by i and j
#     # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
#     #        two lattice directions a1 and a2)
#     for i in range(Nx):
#         for j in range(Ny):
#
#             # Site index of the sublattice B site in this unit cell
#             site_index = get_site_index(Nx, i, j, 1)
#
#             # =========== Add bond interaction terms to the Raman operator ===========
#             # Loop over the 3 bonds for the B site of this unit cell
#             # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
#             for bond_type in range(3):
#
#                 # Get the other two bond types (e.g., if bond_type = 0, then bond_type_other_1 = 1 and bond_type_other_2 = 0)
#                 bond_type_other_1 = (bond_type + 1) % 3
#                 bond_type_other_2 = (bond_type + 2) % 3
#
#                 # Initialize Boolean that specifies whether or not this bond is a boundary bond
#                 # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
#                 boundary_bond = False
#
#                 # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
#                 if bond_type == 0:
#                     i_neighbor = i
#                     j_neighbor = (j + 1) % Ny
#                     if j_neighbor != (j + 1):
#                         boundary_bond = True
#                 elif bond_type == 1:
#                     i_neighbor = (i - 1) % Nx
#                     j_neighbor = (j + 1) % Ny
#                     if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
#                         boundary_bond = True
#                 elif bond_type == 2:
#                     i_neighbor = i
#                     j_neighbor = j
#
#                 # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
#                 # and this is a boundary bond
#                 if (boundary_conditions == 'open') and (boundary_bond == True):
#                     continue
#
#                 # Factor by which to multiply each spin-spin interaction in this bond in order to implement antiperiodic boundary conditions,
#                 # if specified; this will be -1 for antiperiodic boundary bonds, and +1 for all other bonds
#                 if (boundary_conditions == 'antiperiodic') and (boundary_bond == True):
#                     bond_sign = -1
#                 else:
#                     bond_sign = 1
#
#                 # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
#                 site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)
#
#                 # Raman polarization factor (from both direct exchange and superexchange) for this type of bond
#                 P_dir = epsilon_in[bond_type_other_1] * epsilon_out[bond_type_other_1] + \
#                                             epsilon_in[bond_type_other_2] * epsilon_out[bond_type_other_2]
#
#                 # Add the interaction terms for this bond to the Raman operator
#                 # Heisenberg
#                 if J != 0:
#                     matrix_list[site_index] = S[0]
#                     matrix_list[site_index_neighbor] = S[0]
#                     raman_operator += J * kronecker_product(matrix_list) * bond_sign * P_dir
#
#                     matrix_list[site_index] = S[1]
#                     matrix_list[site_index_neighbor] = S[1]
#                     raman_operator += J * kronecker_product(matrix_list) * bond_sign * P_dir
#
#                     matrix_list[site_index] = S[2]
#                     matrix_list[site_index_neighbor] = S[2]
#                     raman_operator += J * kronecker_product(matrix_list) * bond_sign * P_dir
#                 # Kitaev
#                 if K != 0:
#                     matrix_list[site_index] = S[bond_type]
#                     matrix_list[site_index_neighbor] = S[bond_type]
#                     raman_operator += K * kronecker_product(matrix_list) * bond_sign * P_dir
#                 # Gamma
#                 if G != 0:
#                     matrix_list[site_index]          = S[bond_type_other_1]
#                     matrix_list[site_index_neighbor] = S[bond_type_other_2]
#                     raman_operator += G * kronecker_product(matrix_list) * bond_sign * P_dir
#
#                     matrix_list[site_index]          = S[bond_type_other_2]
#                     matrix_list[site_index_neighbor] = S[bond_type_other_1]
#                     raman_operator += G * kronecker_product(matrix_list) * bond_sign * P_dir
#
#                 # Reset matrix_list
#                 matrix_list[site_index] = identity
#                 matrix_list[site_index_neighbor] = identity
#
#     return raman_operator

# cont here
# def plaquette_operators(J, K, G, h, h_direction, spin, Nx, Ny, boundary_conditions = 'periodic', basis = 'z'):
#     """Function that constructs the JKG (Heisenberg + Kitaev + symmetric off-diagonal) Hamiltonian on a honeycomb lattice
#     in an external magnetic field, namely
#         H_JKG = Sum_{<ij> in alpha beta (gamma)} [ J S_i.S_j + K S_i^gamma S_j^gamma + G (S_i^alpha S_j^beta + S_i^beta S_j^alpha) ] - Sum_i h.S_i
#     where Sum_{<ij> in alpha beta (gamma)} denotes a sum over neighboring sites interacting through a gamma-bond
#     (gamma=x,y,z, and alpha,beta are the other two directions); and S_i.S_j denotes a dot product between the spins on sites i and j.
#     Inputs:
#         J (Heisenberg coupling)
#         K (Kitaev coupling)
#         G (Gamma coupling)
#         h (magnitude of external magnetic field;
#            if negative, it also reverses the direction of the magnetic field (h_direction))
#         h_direction (direction of external magnetic field as a list)
#         spin (spin quantum number at each site; can be either 1/2, 1, 3/2, or 2)
#         Nx, Ny (length of system along the two lattice directions)
#         boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
#         basis (string specifying which basis is being used to represent the spin matrices; can be either 'z' or 'e3'.
#                For example, for spin = 1/2:
#                if basis = 'z', then Sx,Sy,Sz will be the Pauli spin matrices sigma_1,sigma_2,sigma_3.
#                if basis = 'e3', then Se1,Se2,Se3 will be the Pauli spin matrices sigma_1,sigma_2,sigma_3.)
#     Output:
#         hamiltonian (SciPy csr (sparse) matrix of size D = dim^num_sites, where dim = 2*spin + 1 and num_sites = 2 Nx Ny)"""
#
#     # Number of sites
#     num_sites = 2 * Nx * Ny
#
#     # Make h_direction a NumPy array and normalize it
#     h_direction = np.array(h_direction)
#     h_direction = h_direction / np.linalg.norm(h_direction)
#
#     # Construct the spin matrices and the identity matrix
#     Sx, Sy, Sz, _, _, _, _, _, _, _, identity = spin_matrices(spin, basis)
#
#     # Make a 1D NumPy array of the three spin matrices
#     S = np.array([Sx, Sy, Sz])
#
#     # Make a 1D NumPy array of num_sites dim-by-dim sparse identity matrices, where dim = 2*spin + 1.
#     # ((For example, for spin=1/2 and num_sites=4, matrix_list is a NumPy array of four 2x2 sparse identity matrices.))
#     matrix_list = np.full(num_sites, identity)
#
#     # Initialize Hamiltonian matrix
#     hamiltonian = 0 * kronecker_product(matrix_list)
#
#     # Loop over all the unit cells, which are indexed by i and j
#     # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
#     #        two lattice directions a1 and a2)
#     for i in range(Nx):
#         for j in range(Ny):
#
#             # Site index of the sublattice B site in this unit cell
#             site_index = get_site_index(Nx, i, j, 1)
#
#             # ============ Add bond interaction terms to the Hamiltonian ============
#             # Loop over the 3 bonds for the B site of this unit cell
#             # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
#             for bond_type in range(3):
#
#                 # Get the other two bond types (e.g., if bond_type = 0, then bond_type_other_1 = 1 and bond_type_other_2 = 0)
#                 bond_type_other_1 = (bond_type + 1) % 3
#                 bond_type_other_2 = (bond_type + 2) % 3
#
#                 # Initialize Boolean that specifies whether or not this bond is a boundary bond
#                 # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
#                 boundary_bond = False
#
#                 # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
#                 if bond_type == 0:
#                     i_neighbor = i
#                     j_neighbor = (j + 1) % Ny
#                     if j_neighbor != (j + 1):
#                         boundary_bond = True
#                 elif bond_type == 1:
#                     i_neighbor = (i - 1) % Nx
#                     j_neighbor = (j + 1) % Ny
#                     if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
#                         boundary_bond = True
#                 elif bond_type == 2:
#                     i_neighbor = i
#                     j_neighbor = j
#
#                 # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
#                 # and this is a boundary bond
#                 if (boundary_conditions == 'open') and (boundary_bond == True):
#                     continue
#
#                 # Factor by which to multiply each spin-spin interaction in this bond in order to implement antiperiodic boundary conditions,
#                 # if specified; this will be -1 for antiperiodic boundary bonds, and +1 for all other bonds
#                 if (boundary_conditions == 'antiperiodic') and (boundary_bond == True):
#                     bond_sign = -1
#                 else:
#                     bond_sign = 1
#
#                 # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
#                 site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)
#
#                 # Add the interaction terms for this bond to the Hamiltonian
#                 # Heisenberg
#                 if J != 0:
#                     matrix_list[site_index] = S[0]
#                     matrix_list[site_index_neighbor] = S[0]
#                     hamiltonian += J * kronecker_product(matrix_list) * bond_sign
#
#                     matrix_list[site_index] = S[1]
#                     matrix_list[site_index_neighbor] = S[1]
#                     hamiltonian += J * kronecker_product(matrix_list) * bond_sign
#
#                     matrix_list[site_index] = S[2]
#                     matrix_list[site_index_neighbor] = S[2]
#                     hamiltonian += J * kronecker_product(matrix_list) * bond_sign
#                 # Kitaev
#                 if K != 0:
#                     matrix_list[site_index] = S[bond_type]
#                     matrix_list[site_index_neighbor] = S[bond_type]
#                     hamiltonian += K * kronecker_product(matrix_list) * bond_sign
#                 # Gamma
#                 if G != 0:
#                     matrix_list[site_index]          = S[bond_type_other_1]
#                     matrix_list[site_index_neighbor] = S[bond_type_other_2]
#                     hamiltonian += G * kronecker_product(matrix_list) * bond_sign
#
#                     matrix_list[site_index]          = S[bond_type_other_2]
#                     matrix_list[site_index_neighbor] = S[bond_type_other_1]
#                     hamiltonian += G * kronecker_product(matrix_list) * bond_sign
#
#                 # Reset matrix_list
#                 matrix_list[site_index] = identity
#                 matrix_list[site_index_neighbor] = identity
#
#             # ================= Add Zeeman terms to the Hamiltonian =================
#             if h != 0:
#                 # Add the Zeeman term for the B site in this unit cell to the Hamiltonian
#                 matrix_list[site_index] = h_direction[0] * S[0] + h_direction[1] * S[1] + h_direction[2] * S[2]
#                 hamiltonian -= h * kronecker_product(matrix_list)
#
#                 # Reset matrix_list
#                 matrix_list[site_index] = identity
#
#                 # Site index of the other site in this unit cell (i.e., the sublattice A site)
#                 site_index_other_site = get_site_index(Nx, i, j, 0)
#
#                 # Add the Zeeman term for the A site in this unit cell to the Hamiltonian
#                 matrix_list[site_index_other_site] = h_direction[0] * S[0] + h_direction[1] * S[1] + h_direction[2] * S[2]
#                 hamiltonian -= h * kronecker_product(matrix_list)
#
#                 # Reset matrix_list
#                 matrix_list[site_index_other_site] = identity
#
#                 # Note: The line below is equivalent to the line below it
#                 # matrix_list[site_index] = h_direction[0] * S[0] + h_direction[1] * S[1] + h_direction[2] * S[2]
#                 # matrix_list[site_index] = h_direction @ S
#                 # However, this last line only works in some versions of Python.
#
#     return hamiltonian cont here

def spin_matrix_elements_between_a_bra_and_many_kets(bra, list_of_kets, Nx, Ny, basis = 'z'):
    """Computes the matrix elements
        <bra|S_i^alpha|ket_m>
    for the spin operators S_i^alpha (alpha=x,y,z) between the eigenvector |bra> and the N eigenvectors {|ket_m>} (m=0,1,...,N-1).
    Inputs:
        bra (1D NumPy array of size D)
        list_of_kets (2D NumPy array of shape D x N, where N is the number of eigenvectors (i.e., kets) in list_of_kets)
        Nx, Ny (length of system along the two lattice directions)
        basis (string specifying which basis is being used to represent the spin matrices; can be either 'z' or 'e3'.
               For example, for spin = 1/2:
               if basis = 'z', then Sx,Sy,Sz will be the Pauli spin matrices sigma_1,sigma_2,sigma_3.
               if basis = 'e3', then Se1,Se2,Se3 will be the Pauli spin matrices sigma_1,sigma_2,sigma_3.)
    Output:
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <bra|S_i^x|ket_m>,
                                 and similarly for the other arrays)
    Note: The input bra is just the regular eigenvector |bra> without complex conjugating it.
          In other words, you should NOT input its complex conjugate <bra| = np.conj(|bra>)."""

    # Number of kets in list_of_kets
    if len(list_of_kets.shape) == 2:
        num_kets = list_of_kets.shape[1]
    elif len(list_of_kets.shape) == 1:
        num_kets = 1

    # Complex conjugate the input bra
    bra = np.conj(bra)

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Spin quantum number at each site
    spin = (len(bra)**(1/num_sites) - 1.)/2.

    # Construct the spin matrices and the identity matrix
    Sx, Sy, Sz, _, _, _, _, _, _, _, identity = spin_matrices(spin, basis)

    # Make a 1D NumPy array of num_sites dim-by-dim sparse identity matrices, where dim = 2*spin + 1.
    # (For example, for spin=1/2 and num_sites=4, matrix_list is a NumPy array of four 2x2 sparse identity matrices.)
    matrix_list = np.full(num_sites, identity)

    # Initialize NumPy arrays that will contain the spin matrix elements between the bra and the list of kets
    Six_0m = np.full((num_sites,num_kets), 0 + 0j)
    Siy_0m = np.full((num_sites,num_kets), 0 + 0j)
    Siz_0m = np.full((num_sites,num_kets), 0 + 0j)

    # ============ Compute the spin matrix elements between the bra and the list of kets ============
    # Loop over all sites
    for site_index in range(num_sites):

        # Six
        matrix_list[site_index] = Sx
        Six_0m[site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
        # Siy
        matrix_list[site_index] = Sy
        Siy_0m[site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
        # Siz
        matrix_list[site_index] = Sz
        Siz_0m[site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets

        # Reset matrix_list
        matrix_list[site_index] = identity
    # ===============================================================================================

    spin_matrix_elements_0m = {"Six_0m":Six_0m, "Siy_0m":Siy_0m, "Siz_0m":Siz_0m}

    return spin_matrix_elements_0m

def spin_matrix_elements_between_many_bras_and_a_ket(spin_matrix_elements_0m):
    """Computes the matrix elements
        <bra_m|S_i^alpha|ket>
    for the spin operators S_i^alpha (alpha=x,y,z) between the N eigenvectors {|bra_m>} (m=0,1,...,N-1) and the eigenvector |ket>.
    Inputs:
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <bra|S_i^x|ket_m>,
                                 and similarly for the other arrays)
    Output:
        spin_matrix_elements_m0 (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_m0, Siy_m0, Siz_m0;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_m0 corresponds to the matrix element <bra_m|S_i^x|ket>,
                                 and similarly for the other arrays)"""

    Six_m0 = spin_matrix_elements_0m['Six_0m'].conj()
    Siy_m0 = spin_matrix_elements_0m['Siy_0m'].conj()
    Siz_m0 = spin_matrix_elements_0m['Siz_0m'].conj()

    spin_matrix_elements_m0 = {"Six_m0":Six_m0, "Siy_m0":Siy_m0, "Siz_m0":Siz_m0}

    return spin_matrix_elements_m0

def spin_matrix_elements_between_a_bra_and_many_kets_e1e2e3(spin_matrix_elements_0m):
    """Computes the matrix elements
        <bra|S_i^alpha|ket_m>
    for the spin operators S_i^alpha (alpha=e1,e2,e3) between the eigenvector |bra> and
    the N eigenvectors {|ket_m>} (m=0,1,...,N-1), where
        e1 = 1/sqrt(6) * (- x - y + 2z)
        e2 = 1/sqrt(2) * (x - y)
        e3 = 1/sqrt(3) * (x + y + z)
    are the spin operators along the e1,e2,e3 lab coordinate directions (e1,e2 lie in the plane and e3 is perpendicular to the plane).
    Inputs:
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <bra|S_i^x|ket_m>,
                                 and similarly for the other arrays)
    Output:
        spin_matrix_elements_e1e2e3_0m (dictionary containing the arrays following 3 NumPy arrays:
                                            Sie1_0m, Sie2_0m, Sie3_0m;
                                        these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                        the (i,m)th element of Sie1_0m corresponds to the matrix element <bra|S_i^e1|ket_m>,
                                        and similarly for the other arrays)"""

    # Lab coordinate unit vectors e1,e2,e3 in terms of their crystal coordinates x,y,z
    e1, e2, e3 = e1e2e3_unit_vectors_in_terms_of_xyz_coordinates()

    Sie1_0m = spin_matrix_elements_0m['Six_0m']*e1[0] + spin_matrix_elements_0m['Siy_0m']*e1[1] + spin_matrix_elements_0m['Siz_0m']*e1[2]
    Sie2_0m = spin_matrix_elements_0m['Six_0m']*e2[0] + spin_matrix_elements_0m['Siy_0m']*e2[1] + spin_matrix_elements_0m['Siz_0m']*e2[2]
    Sie3_0m = spin_matrix_elements_0m['Six_0m']*e3[0] + spin_matrix_elements_0m['Siy_0m']*e3[1] + spin_matrix_elements_0m['Siz_0m']*e3[2]

    spin_matrix_elements_e1e2e3_0m = {"Sie1_0m":Sie1_0m, "Sie2_0m":Sie2_0m, "Sie3_0m":Sie3_0m}

    return spin_matrix_elements_e1e2e3_0m

def spin_matrix_elements_between_many_bras_and_a_ket_e1e2e3(spin_matrix_elements_e1e2e3_0m):
    """Computes the matrix elements
        <bra_m|S_i^alpha|ket>
    for the spin operators S_i^alpha (alpha=e1,e2,e3) between the N eigenvectors {|bra_m>} (m=0,1,...,N-1) and
    the eigenvector |ket>, where
        e1 = 1/sqrt(6) * (- x - y + 2z)
        e2 = 1/sqrt(2) * (x - y)
        e3 = 1/sqrt(3) * (x + y + z)
    are the e1,e2,e3 lab coordinate directions (e1,e2 lie in the plane and e3 is perpendicular to the plane).
    Inputs:
        spin_matrix_elements_e1e2e3_0m (dictionary containing the arrays following 3 NumPy arrays:
                                            Sie1_0m, Sie2_0m, Sie3_0m;
                                        these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                        the (i,m)th element of Sie1_0m corresponds to the matrix element <bra|S_i^e1|ket_m>,
                                        and similarly for the other arrays)
    Output:
        spin_matrix_elements_e1e2e3_m0 (dictionary containing the arrays following 3 NumPy arrays:
                                            Sie1_m0, Sie2_m0, Sie3_m0;
                                        these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                        the (i,m)th element of Sie1_m0 corresponds to the matrix element <bra_m|S_i^e1|ket>,
                                        and similarly for the other arrays)"""

    Sie1_m0 = spin_matrix_elements_e1e2e3_0m['Sie1_0m'].conj()
    Sie2_m0 = spin_matrix_elements_e1e2e3_0m['Sie2_0m'].conj()
    Sie3_m0 = spin_matrix_elements_e1e2e3_0m['Sie3_0m'].conj()

    spin_matrix_elements_e1e2e3_m0 = {"Sie1_m0":Sie1_m0, "Sie2_m0":Sie2_m0, "Sie3_m0":Sie3_m0}

    return spin_matrix_elements_e1e2e3_m0

def spin_flip_matrix_elements_between_a_bra_and_many_kets(spin_matrix_elements_0m):
    """Computes the matrix elements
        <bra|S_i^alpha|ket_m>
    for the spin raising and lowering operators S_i^alpha (alpha=+_z,-_z) between the eigenvector |bra> and
    the N eigenvectors {|ket_m>} (m=0,1,...,N-1), where
        S_i^{+_z} = S_i^x + i S_i^y
        S_i^{-_z} = S_i^x - i S_i^y
    flips the spin at site i toward and away from the z direction, respectively.
    Inputs:
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <bra|S_i^x|ket_m>,
                                 and similarly for the other arrays)
    Output:
        spin_flip_matrix_elements_0m (dictionary containing the arrays following 2 NumPy arrays:
                                          Sip_z_0m, Sim_z_0m;
                                      these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                      the (i,m)th element of Sip_z_0m corresponds to the matrix element <bra|S_i^{+_z}|ket_m>, and
                                      the (i,m)th element of Sim_z_0m corresponds to the matrix element <bra|S_i^{-_z}|ket_m>)"""

    Sip_z_0m = spin_matrix_elements_0m['Six_0m'] + 1j*spin_matrix_elements_0m['Siy_0m']
    Sim_z_0m = spin_matrix_elements_0m['Six_0m'] - 1j*spin_matrix_elements_0m['Siy_0m']

    spin_flip_matrix_elements_0m = {"Sip_z_0m":Sip_z_0m, "Sim_z_0m":Sim_z_0m}

    return spin_flip_matrix_elements_0m

def spin_flip_matrix_elements_between_many_bras_and_a_ket(spin_flip_matrix_elements_0m):
    """Computes the matrix elements
        <bra|S_i^alpha|ket_m>
    for the spin raising and lowering operators S_i^alpha (alpha=+_z,-_z) between the N eigenvectors {|bra_m>} (m=0,1,...,N-1) and
    the eigenvector |ket>, where
        S_i^{+_z} = S_i^x + i S_i^y
        S_i^{-_z} = S_i^x - i S_i^y
    flips the spin at site i toward and away from the z direction, respectively.
    Inputs:
        spin_flip_matrix_elements_0m (dictionary containing the arrays following 2 NumPy arrays:
                                          Sip_z_0m, Sim_z_0m;
                                      these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                      the (i,m)th element of Sip_z_0m corresponds to the matrix element <bra|S_i^{+_z}|ket_m>, and
                                      the (i,m)th element of Sim_z_0m corresponds to the matrix element <bra|S_i^{-_z}|ket_m>)
    Output:
        spin_flip_matrix_elements_m0 (dictionary containing the arrays following 2 NumPy arrays:
                                          Sip_e3_m0, Sim_e3_m0;
                                      these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                      the (i,m)th element of Sip_e3_m0 corresponds to the matrix element <bra_m|S_i^{+_z}|ket>, and
                                      the (i,m)th element of Sim_e3_m0 corresponds to the matrix element <bra_m|S_i^{-_z}|ket>)"""

    Sip_m0 = spin_flip_matrix_elements_0m['Sim_z_0m'].conj()
    Sim_m0 = spin_flip_matrix_elements_0m['Sip_z_0m'].conj()

    spin_flip_matrix_elements_m0 = {"Sip_m0":Sip_m0, "Sim_m0":Sim_m0}

    return spin_flip_matrix_elements_e1e2e3_0m

def spin_flip_matrix_elements_between_a_bra_and_many_kets_e1e2e3(spin_matrix_elements_e1e2e3_0m):
    """Computes the matrix elements
        <bra|S_i^alpha|ket_m>
    for the spin raising and lowering operators S_i^alpha (alpha=+_e3,-_e3) between the eigenvector |bra> and
    the N eigenvectors {|ket_m>} (m=0,1,...,N-1), where
        S_i^{+_e3} = S_i^e1 + i S_i^e2
        S_i^{-_e3} = S_i^e1 - i S_i^e2
    flips the spin at site i toward and away from the e3 direction (perpendicular to the plane), respectively,
        S_i^e1 = 1/sqrt(6) * (- S_i^x - S_i^y + 2 S_i^z)
        S_i^e2 = 1/sqrt(2) * (  S_i^x - S_i^y)
    are the spin operators along the e1 and e2 directions (lying in the plane), and
        e1 = 1/sqrt(6) * (- x - y + 2z)
        e2 = 1/sqrt(2) * (x - y)
        e3 = 1/sqrt(3) * (x + y + z)
    are the e1,e2,e3 lab coordinate directions (e1,e2 lie in the plane and e3 is perpendicular to the plane).
    Inputs:
        spin_matrix_elements_e1e2e3_0m (dictionary containing the arrays following 3 NumPy arrays:
                                            Sie1_0m, Sie2_0m, Sie3_0m;
                                        these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                        the (i,m)th element of Sie1_0m corresponds to the matrix element <bra|S_i^e1|ket_m>,
                                        and similarly for the other arrays)
    Output:
        spin_flip_matrix_elements_e1e2e3_0m (dictionary containing the arrays following 2 NumPy arrays:
                                                 Sip_e3_0m, Sim_e3_0m;
                                             these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                             the (i,m)th element of Sip_e3_0m corresponds to the matrix element <bra|S_i^{+_e3}|ket_m>, and
                                             the (i,m)th element of Sim_e3_0m corresponds to the matrix element <bra|S_i^{-_e3}|ket_m>)"""

    Sip_e3_0m = spin_matrix_elements_e1e2e3_0m['Sie1_0m'] + 1j*spin_matrix_elements_e1e2e3_0m['Sie2_0m']
    Sim_e3_0m = spin_matrix_elements_e1e2e3_0m['Sie1_0m'] - 1j*spin_matrix_elements_e1e2e3_0m['Sie2_0m']

    spin_flip_matrix_elements_e1e2e3_0m = {"Sip_e3_0m":Sip_e3_0m, "Sim_e3_0m":Sim_e3_0m}

    return spin_flip_matrix_elements_e1e2e3_0m

def spin_flip_matrix_elements_between_many_bras_and_a_ket_e1e2e3(spin_flip_matrix_elements_e1e2e3_0m):
    """Computes the matrix elements
        <bra_m|S_j^alpha|ket>
    for the spin raising and lowering operators S_i^alpha (alpha=+_e3,-_e3) between the N eigenvectors {|bra_m>} (m=0,1,...,N-1) and
    the eigenvector |ket>, where
        S_i^{+_e3} = S_i^e1 + i S_i^e2
        S_i^{-_e3} = S_i^e1 - i S_i^e2
    flips the spin at site i toward and away from the e3 direction (perpendicular to the plane), respectively,
        S_i^e1 = 1/sqrt(6) * (- S_i^x - S_i^y + 2 S_i^z)
        S_i^e2 = 1/sqrt(2) * (  S_i^x - S_i^y)
    are the spin operators along the e1 and e2 directions (lying in the plane), and
        e1 = 1/sqrt(6) * (- x - y + 2z)
        e2 = 1/sqrt(2) * (x - y)
        e3 = 1/sqrt(3) * (x + y + z)
    are the e1,e2,e3 lab coordinate directions (e1,e2 lie in the plane and e3 is perpendicular to the plane).
    Inputs:
        spin_flip_matrix_elements_e1e2e3_0m (dictionary containing the arrays following 2 NumPy arrays:
                                                 Sip_e3_0m, Sim_e3_0m;
                                             these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                             the (i,m)th element of Sip_e3_0m corresponds to the matrix element <bra|S_i^{+_e3}|ket_m>, and
                                             the (i,m)th element of Sim_e3_0m corresponds to the matrix element <bra|S_i^{-_e3}|ket_m>)
    Output:
        spin_flip_matrix_elements_e1e2e3_m0 (dictionary containing the arrays following 2 NumPy arrays:
                                                 Sip_e3_m0, Sim_e3_m0;
                                             these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                             the (i,m)th element of Sip_e3_m0 corresponds to the matrix element <bra_m|S_i^{+_e3}|ket>, and
                                             the (i,m)th element of Sim_e3_m0 corresponds to the matrix element <bra_m|S_i^{-_e3}|ket>)"""

    Sip_e3_m0 = spin_flip_matrix_elements_e1e2e3_0m['Sim_e3_0m'].conj()
    Sim_e3_m0 = spin_flip_matrix_elements_e1e2e3_0m['Sip_e3_0m'].conj()

    spin_flip_matrix_elements_e1e2e3_m0 = {"Sip_e3_m0":Sip_e3_m0, "Sim_e3_m0":Sim_e3_m0}

    return spin_flip_matrix_elements_e1e2e3_m0

def spin_spin_correlations_between_a_bra_and_many_kets(bra, list_of_kets, Nx, Ny, basis = 'z'):
    """Computes the spin-spin correlations
        <bra|S_i^alpha S_j^beta|ket_m>
    for the spin operators S_i^alpha and S_j^beta (alpha,beta=x,y,z) between the eigenvector |bra> and
    the N eigenvectors {|ket_m>} (m=0,1,...,N-1).
    Inputs:
        bra (1D NumPy array of size D)
        list_of_kets (2D NumPy array of shape D x N, where N is the number of eigenvectors (i.e., kets) in list_of_kets)
        Nx, Ny (length of system along the two lattice directions)
        basis (string specifying which basis is being used to represent the spin matrices; can be either 'z' or 'e3'.
               For example, for spin = 1/2:
               if basis = 'z', then Sx,Sy,Sz will be the Pauli spin matrices sigma_1,sigma_2,sigma_3.
               if basis = 'e3', then Se1,Se2,Se3 will be the Pauli spin matrices sigma_1,sigma_2,sigma_3.)
    Output:
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
    Note: The input bra is just the regular eigenvector |bra> without complex conjugating it.
          In other words, you should NOT input its complex conjugate <bra| = np.conj(|bra>)."""

    # Number of kets in list_of_kets
    if len(list_of_kets.shape) == 2:
        num_kets = list_of_kets.shape[1]
    elif len(list_of_kets.shape) == 1:
        num_kets = 1

    # Complex conjugate the input bra
    bra = np.conj(bra)

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Spin quantum number at each site
    spin = (len(bra)**(1/num_sites) - 1.)/2.

    # Construct the spin matrices and the identity matrix
    Sx, Sy, Sz, _, _, _, _, _, _, _, identity = spin_matrices(spin, basis)

    # Make a 1D NumPy array of num_sites dim-by-dim sparse identity matrices, where dim = 2*spin + 1.
    # (For example, for spin=1/2 and num_sites=4, matrix_list is a NumPy array of four 2x2 sparse identity matrices.)
    matrix_list = np.full(num_sites, identity)

    # Initialize NumPy arrays that will contain the spin-spin correlations between the bra and the list of kets
    SixSjx_0m = np.full((num_sites,num_sites,num_kets), 0 + 0j)
    SixSjy_0m = np.full((num_sites,num_sites,num_kets), 0 + 0j)
    SixSjz_0m = np.full((num_sites,num_sites,num_kets), 0 + 0j)
    SiySjx_0m = np.full((num_sites,num_sites,num_kets), 0 + 0j)
    SiySjy_0m = np.full((num_sites,num_sites,num_kets), 0 + 0j)
    SiySjz_0m = np.full((num_sites,num_sites,num_kets), 0 + 0j)
    SizSjx_0m = np.full((num_sites,num_sites,num_kets), 0 + 0j)
    SizSjy_0m = np.full((num_sites,num_sites,num_kets), 0 + 0j)
    SizSjz_0m = np.full((num_sites,num_sites,num_kets), 0 + 0j)

    # ========================= Fill in the lower-left triangle of the spin-spin correlation matrices =========================
    # Loop over values of i between 0 and num_sites-1
    for i_site_index in range(num_sites):
        # Loop over values of j between 0 and i
        for j_site_index in range(i_site_index+1):

            # Compute the spin-spin correlations between the sites i and j evaluated between the bra and the list of kets
            # ============================================= i != j =============================================
            if i_site_index != j_site_index:
                # =============== Six ===============
                matrix_list[i_site_index] = Sx
                # ======== Sjx ========
                matrix_list[j_site_index] = Sx
                SixSjx_0m[i_site_index,j_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
                # ======== Sjy ========
                matrix_list[j_site_index] = Sy
                SixSjy_0m[i_site_index,j_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
                # ======== Sjz ========
                matrix_list[j_site_index] = Sz
                SixSjz_0m[i_site_index,j_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets

                # =============== Siy ===============
                matrix_list[i_site_index] = Sy
                # ======== Sjx ========
                matrix_list[j_site_index] = Sx
                SiySjx_0m[i_site_index,j_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
                # ======== Sjy ========
                matrix_list[j_site_index] = Sy
                SiySjy_0m[i_site_index,j_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
                # ======== Sjz ========
                matrix_list[j_site_index] = Sz
                SiySjz_0m[i_site_index,j_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets

                # =============== Siz ===============
                matrix_list[i_site_index] = Sz
                # ======== Sjx ========
                matrix_list[j_site_index] = Sx
                SizSjx_0m[i_site_index,j_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
                # ======== Sjy ========
                matrix_list[j_site_index] = Sy
                SizSjy_0m[i_site_index,j_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
                # ======== Sjz ========
                matrix_list[j_site_index] = Sz
                SizSjz_0m[i_site_index,j_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
            # ============================================= i == j =============================================
            elif i_site_index == j_site_index:
                # =============== Six ===============
                # ======== Sjx ========
                matrix_list[i_site_index] = Sx @ Sx
                SixSjx_0m[i_site_index,i_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
                # ======== Sjy ========
                matrix_list[i_site_index] = Sx @ Sy
                SixSjy_0m[i_site_index,i_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
                # ======== Sjz ========
                matrix_list[i_site_index] = Sx @ Sz
                SixSjz_0m[i_site_index,i_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets

                # =============== Siy ===============
                # ======== Sjx ========
                matrix_list[i_site_index] = Sy @ Sx
                SiySjx_0m[i_site_index,i_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
                # ======== Sjy ========
                matrix_list[i_site_index] = Sy @ Sy
                SiySjy_0m[i_site_index,i_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
                # ======== Sjz ========
                matrix_list[i_site_index] = Sy @ Sz
                SiySjz_0m[i_site_index,i_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets

                # =============== Siz ===============
                # ======== Sjx ========
                matrix_list[i_site_index] = Sz @ Sx
                SizSjx_0m[i_site_index,i_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
                # ======== Sjy ========
                matrix_list[i_site_index] = Sz @ Sy
                SizSjy_0m[i_site_index,i_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
                # ======== Sjz ========
                matrix_list[i_site_index] = Sz @ Sz
                SizSjz_0m[i_site_index,i_site_index] = bra @ kronecker_product(matrix_list) @ list_of_kets
            # ==================================================================================================

            # Reset matrix_list
            matrix_list[i_site_index] = identity
            matrix_list[j_site_index] = identity
    # =========================================================================================================================

    # ========================= Fill in the upper-right triangle of the spin-spin correlation matrices ========================
    # Loop over values of i between 0 and num_sites-1
    for i_site_index in range(num_sites):
        # Loop over values of j between i+1 and num_sites-1
        for j_site_index in range(i_site_index+1,num_sites):

            # Obtain the spin-spin correlations between the sites i and j evaluated between the bra and the list of kets
            # =============== Six ===============
            # ======== Sjx ========
            SixSjx_0m[i_site_index,j_site_index] = SixSjx_0m[j_site_index,i_site_index]
            # ======== Sjy ========
            SixSjy_0m[i_site_index,j_site_index] = SiySjx_0m[j_site_index,i_site_index]
            # ======== Sjz ========
            SixSjz_0m[i_site_index,j_site_index] = SizSjx_0m[j_site_index,i_site_index]

            # =============== Siy ===============
            # ======== Sjx ========
            SiySjx_0m[i_site_index,j_site_index] = SixSjy_0m[j_site_index,i_site_index]
            # ======== Sjy ========
            SiySjy_0m[i_site_index,j_site_index] = SiySjy_0m[j_site_index,i_site_index]
            # ======== Sjz ========
            SiySjz_0m[i_site_index,j_site_index] = SizSjy_0m[j_site_index,i_site_index]

            # =============== Siz ===============
            # ======== Sjx ========
            SizSjx_0m[i_site_index,j_site_index] = SixSjz_0m[j_site_index,i_site_index]
            # ======== Sjy ========
            SizSjy_0m[i_site_index,j_site_index] = SiySjz_0m[j_site_index,i_site_index]
            # ======== Sjz ========
            SizSjz_0m[i_site_index,j_site_index] = SizSjz_0m[j_site_index,i_site_index]
    # =========================================================================================================================

    spin_spin_correlations_0m = {"SixSjx_0m":SixSjx_0m, "SixSjy_0m":SixSjy_0m, "SixSjz_0m":SixSjz_0m,
                                 "SiySjx_0m":SiySjx_0m, "SiySjy_0m":SiySjy_0m, "SiySjz_0m":SiySjz_0m,
                                 "SizSjx_0m":SizSjx_0m, "SizSjy_0m":SizSjy_0m, "SizSjz_0m":SizSjz_0m}

    return spin_spin_correlations_0m

def spin_spin_correlations_between_many_bras_and_a_ket(spin_spin_correlations_0m):
    """Computes the spin-spin correlations
        <bra_m|S_i^alpha S_j^beta|ket>
    for the spin operators S_i^alpha and S_j^beta (alpha,beta=x,y,z) between the N eigenvectors {|bra_m>} (m=0,1,...,N-1) and
    the eigenvector |ket>.
    Input:
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
    Output:
        spin_spin_correlations_m0 (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_m0, SixSjy_m0, SixSjz_m0,
                                       SiySjx_m0, SiySjy_m0, SiySjz_m0,
                                       SizSjx_m0, SizSjy_m0, SizSjz_m0;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_m0 corresponds to the matrix element <bra_m|S_i^x S_j^y|ket>,
                                   and similarly for the other arrays)"""

    SixSjx_m0 = spin_spin_correlations_0m['SixSjx_0m'].conj().transpose(1,0,2)
    SixSjy_m0 = spin_spin_correlations_0m['SiySjx_0m'].conj().transpose(1,0,2)
    SixSjz_m0 = spin_spin_correlations_0m['SizSjx_0m'].conj().transpose(1,0,2)
    SiySjx_m0 = spin_spin_correlations_0m['SixSjy_0m'].conj().transpose(1,0,2)
    SiySjy_m0 = spin_spin_correlations_0m['SiySjy_0m'].conj().transpose(1,0,2)
    SiySjz_m0 = spin_spin_correlations_0m['SizSjy_0m'].conj().transpose(1,0,2)
    SizSjx_m0 = spin_spin_correlations_0m['SixSjz_0m'].conj().transpose(1,0,2)
    SizSjy_m0 = spin_spin_correlations_0m['SiySjz_0m'].conj().transpose(1,0,2)
    SizSjz_m0 = spin_spin_correlations_0m['SizSjz_0m'].conj().transpose(1,0,2)

    spin_spin_correlations_m0 = {"SixSjx_m0":SixSjx_m0, "SixSjy_m0":SixSjy_m0, "SixSjz_m0":SixSjz_m0,
                                 "SiySjx_m0":SiySjx_m0, "SiySjy_m0":SiySjy_m0, "SiySjz_m0":SiySjz_m0,
                                 "SizSjx_m0":SizSjx_m0, "SizSjy_m0":SizSjy_m0, "SizSjz_m0":SizSjz_m0}

    return spin_spin_correlations_m0

def spin_spin_correlations_between_many_bras_and_a_ket_independent_test(list_of_bras, ket, Nx, Ny, basis = 'z'):
    """Computes the spin-spin correlations
        <bra_m|S_i^alpha S_j^beta|ket>
    for the spin operators S_i^alpha and S_j^beta (alpha,beta=x,y,z) between the N eigenvectors {|bra_m>} (m=0,1,...,N-1) and
    the eigenvector |ket>.
    Inputs:
        list_of_bras (2D NumPy array of shape D x N, where N is the number of eigenvectors (i.e., bras) in list_of_bras)
        ket (1D NumPy array of size D)
        Nx, Ny (length of system along the two lattice directions)
        basis (string specifying which basis is being used to represent the spin matrices; can be either 'z' or 'e3'.
               For example, for spin = 1/2:
               if basis = 'z', then Sx,Sy,Sz will be the Pauli spin matrices sigma_1,sigma_2,sigma_3.
               if basis = 'e3', then Se1,Se2,Se3 will be the Pauli spin matrices sigma_1,sigma_2,sigma_3.)
    Output:
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
    Note: The input bras are just the regular eigenvectors {|bra_m>} (m=0,1,...,N-1) without complex conjugating them.
          In other words, you should NOT input its complex conjugates {<bra_0|, ... ,<bra_(N-1)|} = np.conj({|bra_m>} (m=0,1,...,N-1))."""

    # Number of bras in list_of_bras
    if len(list_of_bras.shape) == 2:
        num_bras = list_of_bras.shape[1]
    elif len(list_of_bras.shape) == 1:
        num_bras = 1

    # Complex conjugate the input bras and transpose the array so that it has shape N x D
    list_of_bras = np.conj(list_of_bras).T

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Spin quantum number at each site
    spin = (len(ket)**(1/num_sites) - 1.)/2.

    # Construct the spin matrices and the identity matrix
    Sx, Sy, Sz, _, _, _, _, _, _, _, identity = spin_matrices(spin, basis)

    # Make a 1D NumPy array of num_sites dim-by-dim sparse identity matrices, where dim = 2*spin + 1.
    # (For example, for spin=1/2 and num_sites=4, matrix_list is a NumPy array of four 2x2 sparse identity matrices.)
    matrix_list = np.full(num_sites, identity)

    # Initialize NumPy arrays that will contain the spin-spin correlations between the list of bras and the ket
    SixSjx_m0 = np.full((num_sites,num_sites,num_bras), 0 + 0j)
    SixSjy_m0 = np.full((num_sites,num_sites,num_bras), 0 + 0j)
    SixSjz_m0 = np.full((num_sites,num_sites,num_bras), 0 + 0j)
    SiySjx_m0 = np.full((num_sites,num_sites,num_bras), 0 + 0j)
    SiySjy_m0 = np.full((num_sites,num_sites,num_bras), 0 + 0j)
    SiySjz_m0 = np.full((num_sites,num_sites,num_bras), 0 + 0j)
    SizSjx_m0 = np.full((num_sites,num_sites,num_bras), 0 + 0j)
    SizSjy_m0 = np.full((num_sites,num_sites,num_bras), 0 + 0j)
    SizSjz_m0 = np.full((num_sites,num_sites,num_bras), 0 + 0j)

    # ========================= Fill in the lower-left triangle of the spin-spin correlation matrices =========================
    # Loop over values of i between 0 and num_sites-1
    for i_site_index in range(num_sites):
        # Loop over values of j between 0 and i
        for j_site_index in range(i_site_index+1):

            # Compute the spin-spin correlations between the sites i and j evaluated between the list of bras and the ket
            # ============================================= i != j =============================================
            if i_site_index != j_site_index:
                # =============== Six ===============
                matrix_list[i_site_index] = Sx
                # ======== Sjx ========
                matrix_list[j_site_index] = Sx
                SixSjx_m0[i_site_index,j_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket
                # ======== Sjy ========
                matrix_list[j_site_index] = Sy
                SixSjy_m0[i_site_index,j_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket
                # ======== Sjz ========
                matrix_list[j_site_index] = Sz
                SixSjz_m0[i_site_index,j_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket

                # =============== Siy ===============
                matrix_list[i_site_index] = Sy
                # ======== Sjx ========
                matrix_list[j_site_index] = Sx
                SiySjx_m0[i_site_index,j_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket
                # ======== Sjy ========
                matrix_list[j_site_index] = Sy
                SiySjy_m0[i_site_index,j_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket
                # ======== Sjz ========
                matrix_list[j_site_index] = Sz
                SiySjz_m0[i_site_index,j_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket

                # =============== Siz ===============
                matrix_list[i_site_index] = Sz
                # ======== Sjx ========
                matrix_list[j_site_index] = Sx
                SizSjx_m0[i_site_index,j_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket
                # ======== Sjy ========
                matrix_list[j_site_index] = Sy
                SizSjy_m0[i_site_index,j_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket
                # ======== Sjz ========
                matrix_list[j_site_index] = Sz
                SizSjz_m0[i_site_index,j_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket
            # ============================================= i == j =============================================
            elif i_site_index == j_site_index:
                # =============== Six ===============
                # ======== Sjx ========
                matrix_list[i_site_index] = Sx @ Sx
                SixSjx_m0[i_site_index,i_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket
                # ======== Sjy ========
                matrix_list[i_site_index] = Sx @ Sy
                SixSjy_m0[i_site_index,i_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket
                # ======== Sjz ========
                matrix_list[i_site_index] = Sx @ Sz
                SixSjz_m0[i_site_index,i_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket

                # =============== Siy ===============
                # ======== Sjx ========
                matrix_list[i_site_index] = Sy @ Sx
                SiySjx_m0[i_site_index,i_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket
                # ======== Sjy ========
                matrix_list[i_site_index] = Sy @ Sy
                SiySjy_m0[i_site_index,i_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket
                # ======== Sjz ========
                matrix_list[i_site_index] = Sy @ Sz
                SiySjz_m0[i_site_index,i_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket

                # =============== Siz ===============
                # ======== Sjx ========
                matrix_list[i_site_index] = Sz @ Sx
                SizSjx_m0[i_site_index,i_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket
                # ======== Sjy ========
                matrix_list[i_site_index] = Sz @ Sy
                SizSjy_m0[i_site_index,i_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket
                # ======== Sjz ========
                matrix_list[i_site_index] = Sz @ Sz
                SizSjz_m0[i_site_index,i_site_index] = list_of_bras @ kronecker_product(matrix_list) @ ket
            # ==================================================================================================

            # Reset matrix_list
            matrix_list[i_site_index] = identity
            matrix_list[j_site_index] = identity
    # =========================================================================================================================

    # ========================= Fill in the upper-right triangle of the spin-spin correlation matrices ========================
    # Loop over values of i between 0 and num_sites-1
    for i_site_index in range(num_sites):
        # Loop over values of j between i+1 and num_sites-1
        for j_site_index in range(i_site_index+1,num_sites):

            # Obtain the spin-spin correlations between the sites i and j evaluated between the list of bras and the ket
            # =============== Six ===============
            # ======== Sjx ========
            SixSjx_m0[i_site_index,j_site_index] = SixSjx_m0[j_site_index,i_site_index]
            # ======== Sjy ========
            SixSjy_m0[i_site_index,j_site_index] = SiySjx_m0[j_site_index,i_site_index]
            # ======== Sjz ========
            SixSjz_m0[i_site_index,j_site_index] = SizSjx_m0[j_site_index,i_site_index]

            # =============== Siy ===============
            # ======== Sjx ========
            SiySjx_m0[i_site_index,j_site_index] = SixSjy_m0[j_site_index,i_site_index]
            # ======== Sjy ========
            SiySjy_m0[i_site_index,j_site_index] = SiySjy_m0[j_site_index,i_site_index]
            # ======== Sjz ========
            SiySjz_m0[i_site_index,j_site_index] = SizSjy_m0[j_site_index,i_site_index]

            # =============== Siz ===============
            # ======== Sjx ========
            SizSjx_m0[i_site_index,j_site_index] = SixSjz_m0[j_site_index,i_site_index]
            # ======== Sjy ========
            SizSjy_m0[i_site_index,j_site_index] = SiySjz_m0[j_site_index,i_site_index]
            # ======== Sjz ========
            SizSjz_m0[i_site_index,j_site_index] = SizSjz_m0[j_site_index,i_site_index]
    # =========================================================================================================================

    spin_spin_correlations_m0 = {"SixSjx_m0":SixSjx_m0, "SixSjy_m0":SixSjy_m0, "SixSjz_m0":SixSjz_m0,
                                 "SiySjx_m0":SiySjx_m0, "SiySjy_m0":SiySjy_m0, "SiySjz_m0":SiySjz_m0,
                                 "SizSjx_m0":SizSjx_m0, "SizSjy_m0":SizSjy_m0, "SizSjz_m0":SizSjz_m0}

    return spin_spin_correlations_m0

def spin_spin_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_0m):
    """Computes the spin-spin correlations
        <bra|S_i^alpha S_j^beta|ket_m>
    for the spin operators S_i^alpha and S_j^beta (alpha,beta=e1,e2,e3) between the eigenvector |bra> and
    the N eigenvectors {|ket_m>} (m=0,1,...,N-1).
    Input:
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
    Output:
        spin_spin_correlations_e1e2e3_0m (dictionary containing the arrays following 9 NumPy arrays:
                                              Sie1Sje1_0m, Sie1Sje2_0m, Sie1Sje3_0m,
                                              Sie2Sje1_0m, Sie2Sje2_0m, Sie2Sje3_0m,
                                              Sie3Sje1_0m, Sie3Sje2_0m, Sie3Sje3_0m;
                                          these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                          the (i,j,m)th element of Sie1Sje2_0m corresponds to the matrix element <bra|S_i^e1 S_j^e2|ket_m>,
                                          and similarly for the other arrays)"""

    # Lab coordinate unit vectors e1,e2,e3 in terms of their crystal coordinates x,y,z
    e1, e2, e3 = e1e2e3_unit_vectors_in_terms_of_xyz_coordinates()

    Sie1Sje1_0m = spin_spin_correlations_0m['SixSjx_0m']*e1[0]*e1[0] + spin_spin_correlations_0m['SixSjy_0m']*e1[0]*e1[1] + spin_spin_correlations_0m['SixSjz_0m']*e1[0]*e1[2] + \
                  spin_spin_correlations_0m['SiySjx_0m']*e1[1]*e1[0] + spin_spin_correlations_0m['SiySjy_0m']*e1[1]*e1[1] + spin_spin_correlations_0m['SiySjz_0m']*e1[1]*e1[2] + \
                  spin_spin_correlations_0m['SizSjx_0m']*e1[2]*e1[0] + spin_spin_correlations_0m['SizSjy_0m']*e1[2]*e1[1] + spin_spin_correlations_0m['SizSjz_0m']*e1[2]*e1[2]

    Sie1Sje2_0m = spin_spin_correlations_0m['SixSjx_0m']*e1[0]*e2[0] + spin_spin_correlations_0m['SixSjy_0m']*e1[0]*e2[1] + spin_spin_correlations_0m['SixSjz_0m']*e1[0]*e2[2] + \
                  spin_spin_correlations_0m['SiySjx_0m']*e1[1]*e2[0] + spin_spin_correlations_0m['SiySjy_0m']*e1[1]*e2[1] + spin_spin_correlations_0m['SiySjz_0m']*e1[1]*e2[2] + \
                  spin_spin_correlations_0m['SizSjx_0m']*e1[2]*e2[0] + spin_spin_correlations_0m['SizSjy_0m']*e1[2]*e2[1] + spin_spin_correlations_0m['SizSjz_0m']*e1[2]*e2[2]

    Sie1Sje3_0m = spin_spin_correlations_0m['SixSjx_0m']*e1[0]*e3[0] + spin_spin_correlations_0m['SixSjy_0m']*e1[0]*e3[1] + spin_spin_correlations_0m['SixSjz_0m']*e1[0]*e3[2] + \
                  spin_spin_correlations_0m['SiySjx_0m']*e1[1]*e3[0] + spin_spin_correlations_0m['SiySjy_0m']*e1[1]*e3[1] + spin_spin_correlations_0m['SiySjz_0m']*e1[1]*e3[2] + \
                  spin_spin_correlations_0m['SizSjx_0m']*e1[2]*e3[0] + spin_spin_correlations_0m['SizSjy_0m']*e1[2]*e3[1] + spin_spin_correlations_0m['SizSjz_0m']*e1[2]*e3[2]

    Sie2Sje1_0m = spin_spin_correlations_0m['SixSjx_0m']*e2[0]*e1[0] + spin_spin_correlations_0m['SixSjy_0m']*e2[0]*e1[1] + spin_spin_correlations_0m['SixSjz_0m']*e2[0]*e1[2] + \
                  spin_spin_correlations_0m['SiySjx_0m']*e2[1]*e1[0] + spin_spin_correlations_0m['SiySjy_0m']*e2[1]*e1[1] + spin_spin_correlations_0m['SiySjz_0m']*e2[1]*e1[2] + \
                  spin_spin_correlations_0m['SizSjx_0m']*e2[2]*e1[0] + spin_spin_correlations_0m['SizSjy_0m']*e2[2]*e1[1] + spin_spin_correlations_0m['SizSjz_0m']*e2[2]*e1[2]

    Sie2Sje2_0m = spin_spin_correlations_0m['SixSjx_0m']*e2[0]*e2[0] + spin_spin_correlations_0m['SixSjy_0m']*e2[0]*e2[1] + spin_spin_correlations_0m['SixSjz_0m']*e2[0]*e2[2] + \
                  spin_spin_correlations_0m['SiySjx_0m']*e2[1]*e2[0] + spin_spin_correlations_0m['SiySjy_0m']*e2[1]*e2[1] + spin_spin_correlations_0m['SiySjz_0m']*e2[1]*e2[2] + \
                  spin_spin_correlations_0m['SizSjx_0m']*e2[2]*e2[0] + spin_spin_correlations_0m['SizSjy_0m']*e2[2]*e2[1] + spin_spin_correlations_0m['SizSjz_0m']*e2[2]*e2[2]

    Sie2Sje3_0m = spin_spin_correlations_0m['SixSjx_0m']*e2[0]*e3[0] + spin_spin_correlations_0m['SixSjy_0m']*e2[0]*e3[1] + spin_spin_correlations_0m['SixSjz_0m']*e2[0]*e3[2] + \
                  spin_spin_correlations_0m['SiySjx_0m']*e2[1]*e3[0] + spin_spin_correlations_0m['SiySjy_0m']*e2[1]*e3[1] + spin_spin_correlations_0m['SiySjz_0m']*e2[1]*e3[2] + \
                  spin_spin_correlations_0m['SizSjx_0m']*e2[2]*e3[0] + spin_spin_correlations_0m['SizSjy_0m']*e2[2]*e3[1] + spin_spin_correlations_0m['SizSjz_0m']*e2[2]*e3[2]

    Sie3Sje1_0m = spin_spin_correlations_0m['SixSjx_0m']*e3[0]*e1[0] + spin_spin_correlations_0m['SixSjy_0m']*e3[0]*e1[1] + spin_spin_correlations_0m['SixSjz_0m']*e3[0]*e1[2] + \
                  spin_spin_correlations_0m['SiySjx_0m']*e3[1]*e1[0] + spin_spin_correlations_0m['SiySjy_0m']*e3[1]*e1[1] + spin_spin_correlations_0m['SiySjz_0m']*e3[1]*e1[2] + \
                  spin_spin_correlations_0m['SizSjx_0m']*e3[2]*e1[0] + spin_spin_correlations_0m['SizSjy_0m']*e3[2]*e1[1] + spin_spin_correlations_0m['SizSjz_0m']*e3[2]*e1[2]

    Sie3Sje2_0m = spin_spin_correlations_0m['SixSjx_0m']*e3[0]*e2[0] + spin_spin_correlations_0m['SixSjy_0m']*e3[0]*e2[1] + spin_spin_correlations_0m['SixSjz_0m']*e3[0]*e2[2] + \
                  spin_spin_correlations_0m['SiySjx_0m']*e3[1]*e2[0] + spin_spin_correlations_0m['SiySjy_0m']*e3[1]*e2[1] + spin_spin_correlations_0m['SiySjz_0m']*e3[1]*e2[2] + \
                  spin_spin_correlations_0m['SizSjx_0m']*e3[2]*e2[0] + spin_spin_correlations_0m['SizSjy_0m']*e3[2]*e2[1] + spin_spin_correlations_0m['SizSjz_0m']*e3[2]*e2[2]

    Sie3Sje3_0m = spin_spin_correlations_0m['SixSjx_0m']*e3[0]*e3[0] + spin_spin_correlations_0m['SixSjy_0m']*e3[0]*e3[1] + spin_spin_correlations_0m['SixSjz_0m']*e3[0]*e3[2] + \
                  spin_spin_correlations_0m['SiySjx_0m']*e3[1]*e3[0] + spin_spin_correlations_0m['SiySjy_0m']*e3[1]*e3[1] + spin_spin_correlations_0m['SiySjz_0m']*e3[1]*e3[2] + \
                  spin_spin_correlations_0m['SizSjx_0m']*e3[2]*e3[0] + spin_spin_correlations_0m['SizSjy_0m']*e3[2]*e3[1] + spin_spin_correlations_0m['SizSjz_0m']*e3[2]*e3[2]

    spin_spin_correlations_e1e2e3_0m = {"Sie1Sje1_0m":Sie1Sje1_0m, "Sie1Sje2_0m":Sie1Sje2_0m, "Sie1Sje3_0m":Sie1Sje3_0m,
                                        "Sie2Sje1_0m":Sie2Sje1_0m, "Sie2Sje2_0m":Sie2Sje2_0m, "Sie2Sje3_0m":Sie2Sje3_0m,
                                        "Sie3Sje1_0m":Sie3Sje1_0m, "Sie3Sje2_0m":Sie3Sje2_0m, "Sie3Sje3_0m":Sie3Sje3_0m}

    return spin_spin_correlations_e1e2e3_0m

def spin_spin_correlations_between_many_bras_and_a_ket_e1e2e3(spin_spin_correlations_e1e2e3_0m):
    """Computes the spin-spin correlations
        <bra_m|S_i^alpha S_j^beta|ket>
    for the spin operators S_i^alpha and S_j^beta (alpha,beta=e1,e2,e3) between the N eigenvectors {|bra_m>} (m=0,1,...,N-1) and
    the eigenvector |ket>.
    Input:
        spin_spin_correlations_e1e2e3_0m (dictionary containing the arrays following 9 NumPy arrays:
                                              Sie1Sje1_0m, Sie1Sje2_0m, Sie1Sje3_0m,
                                              Sie2Sje1_0m, Sie2Sje2_0m, Sie2Sje3_0m,
                                              Sie3Sje1_0m, Sie3Sje2_0m, Sie3Sje3_0m;
                                          these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                          the (i,j,m)th element of Sie1Sje2_0m corresponds to the matrix element <bra|S_i^e1 S_j^e2|ket_m>,
                                          and similarly for the other arrays)
    Output:
        spin_spin_correlations_e1e2e3_m0 (dictionary containing the arrays following 9 NumPy arrays:
                                              Sie1Sje1_m0, Sie1Sje2_m0, Sie1Sje3_m0,
                                              Sie2Sje1_m0, Sie2Sje2_m0, Sie2Sje3_m0,
                                              Sie3Sje1_m0, Sie3Sje2_m0, Sie3Sje3_m0;
                                          these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                          the (i,j,m)th element of Sie1Sje2_m0 corresponds to the matrix element <bra_m|S_i^e1 S_j^e2|ket>,
                                          and similarly for the other arrays)"""

    Sie1Sje1_m0 = spin_spin_correlations_e1e2e3_0m['Sie1Sje1_0m'].conj().transpose(1,0,2)
    Sie1Sje2_m0 = spin_spin_correlations_e1e2e3_0m['Sie2Sje1_0m'].conj().transpose(1,0,2)
    Sie1Sje3_m0 = spin_spin_correlations_e1e2e3_0m['Sie3Sje1_0m'].conj().transpose(1,0,2)
    Sie2Sje1_m0 = spin_spin_correlations_e1e2e3_0m['Sie1Sje2_0m'].conj().transpose(1,0,2)
    Sie2Sje2_m0 = spin_spin_correlations_e1e2e3_0m['Sie2Sje2_0m'].conj().transpose(1,0,2)
    Sie2Sje3_m0 = spin_spin_correlations_e1e2e3_0m['Sie3Sje2_0m'].conj().transpose(1,0,2)
    Sie3Sje1_m0 = spin_spin_correlations_e1e2e3_0m['Sie1Sje3_0m'].conj().transpose(1,0,2)
    Sie3Sje2_m0 = spin_spin_correlations_e1e2e3_0m['Sie2Sje3_0m'].conj().transpose(1,0,2)
    Sie3Sje3_m0 = spin_spin_correlations_e1e2e3_0m['Sie3Sje3_0m'].conj().transpose(1,0,2)

    spin_spin_correlations_e1e2e3_m0 = {"Sie1Sje1_m0":Sie1Sje1_m0, "Sie1Sje2_m0":Sie1Sje2_m0, "Sie1Sje3_m0":Sie1Sje3_m0,
                                        "Sie2Sje1_m0":Sie2Sje1_m0, "Sie2Sje2_m0":Sie2Sje2_m0, "Sie2Sje3_m0":Sie2Sje3_m0,
                                        "Sie3Sje1_m0":Sie3Sje1_m0, "Sie3Sje2_m0":Sie3Sje2_m0, "Sie3Sje3_m0":Sie3Sje3_m0}

    return spin_spin_correlations_e1e2e3_m0

def spin_flip_spin_flip_correlations_between_a_bra_and_many_kets(spin_spin_correlations_0m):
    """Computes the spin flip-spin flip correlations
        <bra|S_i^alpha S_j^beta|ket_m>
    for the spin raising and lowering operators S_i^alpha and S_j^beta (alpha,beta=+_z,-_z) between the eigenvector |bra> and
    the N eigenvectors {|ket_m>} (m=0,1,...,N-1), where
        S_i^{+_z} = S_i^x + i S_i^y
        S_i^{-_z} = S_i^x - i S_i^y
    flips the spin at site i toward and away from the z direction, respectively, and similarly for S_j^beta.
    Input:
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
    Output:
        spin_flip_spin_flip_correlations_0m (dictionary containing the arrays following 4 NumPy arrays:
                                                 Sip_z_Sjp_z_0m, Sip_z_Sjm_z_0m,
                                                 Sim_z_Sjp_z_0m, Sim_z_Sjm_z_0m;
                                             these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                             the (i,m)th element of Sip_z_Sjm_z_0m corresponds to the matrix element <bra|S_i^{+_z} S_j^{-_z}|ket_m>, and
                                             similarly for the other 3 arrays)"""

    Sip_z_Sjp_z_0m =     spin_spin_correlations_0m['SixSjx_0m'] - spin_spin_correlations_0m['SiySjy_0m'] + \
                     1j*(spin_spin_correlations_0m['SixSjy_0m'] + spin_spin_correlations_0m['SiySjx_0m'])

    Sip_z_Sjm_z_0m =     spin_spin_correlations_0m['SixSjx_0m'] + spin_spin_correlations_0m['SiySjy_0m'] + \
                    1j*(-spin_spin_correlations_0m['SixSjy_0m'] + spin_spin_correlations_0m['SiySjx_0m'])

    Sim_z_Sjp_z_0m =     spin_spin_correlations_0m['SixSjx_0m'] + spin_spin_correlations_0m['SiySjy_0m'] + \
                     1j*(spin_spin_correlations_0m['SixSjy_0m'] - spin_spin_correlations_0m['SiySjx_0m'])

    Sim_z_Sjm_z_0m =     spin_spin_correlations_0m['SixSjx_0m'] - spin_spin_correlations_0m['SiySjy_0m'] - \
                     1j*(spin_spin_correlations_0m['SixSjy_0m'] + spin_spin_correlations_0m['SiySjx_0m'])

    spin_flip_spin_flip_correlations_0m = {"Sip_z_Sjp_z_0m":Sip_z_Sjp_z_0m, "Sip_z_Sjm_z_0m":Sip_z_Sjm_z_0m,
                                           "Sim_z_Sjp_z_0m":Sim_z_Sjp_z_0m, "Sim_z_Sjm_z_0m":Sim_z_Sjm_z_0m}

    return spin_flip_spin_flip_correlations_0m

def spin_flip_spin_flip_correlations_between_many_bras_and_a_ket(spin_flip_spin_flip_correlations_0m):
    """Computes the spin flip-spin flip correlations
        <bra_m|S_i^alpha S_j^beta|ket>
    for the spin raising and lowering operators S_i^alpha and S_j^beta (alpha,beta=+_z,-_z) between the N eigenvectors {|bra_m>} (m=0,1,...,N-1) and
    the eigenvector |ket>, where
        S_i^{+_z} = S_i^x + i S_i^y
        S_i^{-_z} = S_i^x - i S_i^y
    flips the spin at site i toward and away from the z direction, respectively, and similarly for S_j^beta.
    Input:
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
    Output:
        spin_flip_spin_flip_correlations_m0 (dictionary containing the arrays following 4 NumPy arrays:
                                                 Sip_z_Sjp_z_m0, Sip_z_Sjm_z_m0,
                                                 Sim_z_Sjp_z_m0, Sim_z_Sjm_z_m0;
                                             these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                             the (i,m)th element of Sip_z_Sjm_z_m0 corresponds to the matrix element <bra_m|S_i^{+_z} S_j^{-_z}|ket>, and
                                             similarly for the other 3 arrays)"""

    Sip_z_Sjp_z_m0 = spin_flip_spin_flip_correlations_0m['Sim_z_Sjm_z_0m'].conj().transpose(1,0,2)
    Sip_z_Sjm_z_m0 = spin_flip_spin_flip_correlations_0m['Sip_z_Sjm_z_0m'].conj().transpose(1,0,2)
    Sim_z_Sjp_z_m0 = spin_flip_spin_flip_correlations_0m['Sim_z_Sjp_z_0m'].conj().transpose(1,0,2)
    Sim_z_Sjm_z_m0 = spin_flip_spin_flip_correlations_0m['Sip_z_Sjp_z_0m'].conj().transpose(1,0,2)

    spin_flip_spin_flip_correlations_m0 = {"Sip_z_Sjp_z_m0":Sip_z_Sjp_z_m0, "Sip_z_Sjm_z_m0":Sip_z_Sjm_z_m0,
                                           "Sim_z_Sjp_z_m0":Sim_z_Sjp_z_m0, "Sim_z_Sjm_z_m0":Sim_z_Sjm_z_m0}

    return spin_flip_spin_flip_correlations_m0

def spin_flip_spin_flip_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_e1e2e3_0m):
    """Computes the spin flip-spin flip correlations
        <bra|S_i^alpha S_j^beta|ket_m>
    for the spin raising and lowering operators S_i^alpha and S_j^beta (alpha,beta=+_e3,-_e3) between the eigenvector |bra> and
    the N eigenvectors {|ket_m>} (m=0,1,...,N-1), where
        S_i^{+_e3} = S_i^e1 + i S_i^e2
        S_i^{-_e3} = S_i^e1 - i S_i^e2
    flips the spin at site i toward and away from the e3 direction (perpendicular to the plane), respectively, and similarly for S_j^beta;
        S_i^e1 = 1/sqrt(6) * (- S_i^x - S_i^y + 2 S_i^z)
        S_i^e2 = 1/sqrt(2) * (  S_i^x - S_i^y)
    are the spin operators along the e1 and e2 directions (lying in the plane); and
        e1 = 1/sqrt(6) * (- x - y + 2z)
        e2 = 1/sqrt(2) * (x - y)
        e3 = 1/sqrt(3) * (x + y + z)
    are the e1,e2,e3 lab coordinate directions (e1,e2 lie in the plane and e3 is perpendicular to the plane).
    Input:
        spin_spin_correlations_e1e2e3_0m (dictionary containing the arrays following 9 NumPy arrays:
                                              Sie1Sje1_0m, Sie1Sje2_0m, Sie1Sje3_0m,
                                              Sie2Sje1_0m, Sie2Sje2_0m, Sie2Sje3_0m,
                                              Sie3Sje1_0m, Sie3Sje2_0m, Sie3Sje3_0m;
                                          these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                          the (i,j,m)th element of Sie1Sje2_0m corresponds to the matrix element <bra|S_i^e1 S_j^e2|ket_m>,
                                          and similarly for the other arrays)
    Output:
        spin_flip_spin_flip_correlations_e1e2e3_0m (dictionary containing the arrays following 4 NumPy arrays:
                                                        Sip_e3_Sjp_e3_0m, Sip_e3_Sjm_e3_0m,
                                                        Sim_e3_Sjp_e3_0m, Sim_e3_Sjm_e3_0m;
                                                    these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                                    the (i,m)th element of Sip_e3_Sjm_e3_0m corresponds to the matrix element <bra|S_i^{+_e3} S_j^{-_e3}|ket_m>, and
                                                    similarly for the other 3 arrays)"""

    Sip_e3_Sjp_e3_0m =     spin_spin_correlations_e1e2e3_0m['Sie1Sje1_0m'] - spin_spin_correlations_e1e2e3_0m['Sie2Sje2_0m'] + \
                       1j*(spin_spin_correlations_e1e2e3_0m['Sie1Sje2_0m'] + spin_spin_correlations_e1e2e3_0m['Sie2Sje1_0m'])

    Sip_e3_Sjm_e3_0m =     spin_spin_correlations_e1e2e3_0m['Sie1Sje1_0m'] + spin_spin_correlations_e1e2e3_0m['Sie2Sje2_0m'] + \
                      1j*(-spin_spin_correlations_e1e2e3_0m['Sie1Sje2_0m'] + spin_spin_correlations_e1e2e3_0m['Sie2Sje1_0m'])

    Sim_e3_Sjp_e3_0m =     spin_spin_correlations_e1e2e3_0m['Sie1Sje1_0m'] + spin_spin_correlations_e1e2e3_0m['Sie2Sje2_0m'] + \
                       1j*(spin_spin_correlations_e1e2e3_0m['Sie1Sje2_0m'] - spin_spin_correlations_e1e2e3_0m['Sie2Sje1_0m'])

    Sim_e3_Sjm_e3_0m =     spin_spin_correlations_e1e2e3_0m['Sie1Sje1_0m'] - spin_spin_correlations_e1e2e3_0m['Sie2Sje2_0m'] - \
                       1j*(spin_spin_correlations_e1e2e3_0m['Sie1Sje2_0m'] + spin_spin_correlations_e1e2e3_0m['Sie2Sje1_0m'])

    spin_flip_spin_flip_correlations_e1e2e3_0m = {"Sip_e3_Sjp_e3_0m":Sip_e3_Sjp_e3_0m, "Sip_e3_Sjm_e3_0m":Sip_e3_Sjm_e3_0m,
                                                  "Sim_e3_Sjp_e3_0m":Sim_e3_Sjp_e3_0m, "Sim_e3_Sjm_e3_0m":Sim_e3_Sjm_e3_0m}

    return spin_flip_spin_flip_correlations_e1e2e3_0m

def spin_flip_spin_flip_correlations_between_many_bras_and_a_ket_e1e2e3(spin_flip_spin_flip_correlations_e1e2e3_0m):
    """Computes the spin flip-spin flip correlations
        <bra_m|S_i^alpha S_j^beta|ket>
    for the spin raising and lowering operators S_i^alpha and S_j^beta (alpha,beta=+_e3,-_e3) between the N eigenvectors {|bra_m>} (m=0,1,...,N-1) and
    the eigenvector |ket>, where
        S_i^{+_e3} = S_i^e1 + i S_i^e2
        S_i^{-_e3} = S_i^e1 - i S_i^e2
    flips the spin at site i toward and away from the e3 direction (perpendicular to the plane), respectively, and similarly for S_j^beta;
        S_i^e1 = 1/sqrt(6) * (- S_i^x - S_i^y + 2 S_i^z)
        S_i^e2 = 1/sqrt(2) * (  S_i^x - S_i^y)
    are the spin operators along the e1 and e2 directions (lying in the plane); and
        e1 = 1/sqrt(6) * (- x - y + 2z)
        e2 = 1/sqrt(2) * (x - y)
        e3 = 1/sqrt(3) * (x + y + z)
    are the e1,e2,e3 lab coordinate directions (e1,e2 lie in the plane and e3 is perpendicular to the plane).
    Input:
        spin_flip_spin_flip_correlations_e1e2e3_0m (dictionary containing the arrays following 4 NumPy arrays:
                                                        Sip_e3_Sjp_e3_0m, Sip_e3_Sjm_e3_0m,
                                                        Sim_e3_Sjp_e3_0m, Sim_e3_Sjm_e3_0m;
                                                    these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                                    the (i,m)th element of Sip_e3_Sjm_e3_0m corresponds to the matrix element <bra|S_i^{+_e3} S_j^{-_e3}|ket_m>, and
                                                    similarly for the other 3 arrays)
    Output:
        spin_flip_spin_flip_correlations_e1e2e3_m0 (dictionary containing the arrays following 4 NumPy arrays:
                                                        Sip_e3_Sjp_e3_m0, Sip_e3_Sjm_e3_m0,
                                                        Sim_e3_Sjp_e3_m0, Sim_e3_Sjm_e3_m0;
                                                    these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                                    the (i,m)th element of Sip_e3_Sjm_e3_m0 corresponds to the matrix element <bra_m|S_i^{+_e3} S_j^{-_e3}|ket>, and
                                                    similarly for the other 3 arrays)"""

    Sip_e3_Sjp_e3_m0 = spin_flip_spin_flip_correlations_e1e2e3_0m['Sim_e3_Sjm_e3_0m'].conj().transpose(1,0,2)
    Sip_e3_Sjm_e3_m0 = spin_flip_spin_flip_correlations_e1e2e3_0m['Sip_e3_Sjm_e3_0m'].conj().transpose(1,0,2)
    Sim_e3_Sjp_e3_m0 = spin_flip_spin_flip_correlations_e1e2e3_0m['Sim_e3_Sjp_e3_0m'].conj().transpose(1,0,2)
    Sim_e3_Sjm_e3_m0 = spin_flip_spin_flip_correlations_e1e2e3_0m['Sip_e3_Sjp_e3_0m'].conj().transpose(1,0,2)

    spin_flip_spin_flip_correlations_m0 = {"Sip_e3_Sjp_e3_m0":Sip_e3_Sjp_e3_m0, "Sip_e3_Sjm_e3_m0":Sip_e3_Sjm_e3_m0,
                                           "Sim_e3_Sjp_e3_m0":Sim_e3_Sjp_e3_m0, "Sim_e3_Sjm_e3_m0":Sim_e3_Sjm_e3_m0}

    return spin_flip_spin_flip_correlations_m0

def probability_of_flipping_one_spin(spin_matrix_elements_0m, eigenvector_index):
    """Function that computes the probability of flipping 1 spin along the z direction when going
    from the eigenvector |0> to the mth eigenvector |m>, namely
        Prob(1 flip) = Sum_i Sum_{alpha=+_z,-_z} |<0|S_i^alpha|m>|^2
    where i indexes the lattice sites.
    Inputs:
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <0|S_i^x|m>,
                                 and similarly for the other arrays)
        eigenvector_index (integer specifying the eigenvector |m> for which to compute the probability of flipping 1 spin)
    Output:
        prob_one_flip (float; probability of flipping 1 spin along the z direction)"""

    # Compute the spin flip matrix elements for spin flips along the z axis: <0|S_i^alpha|m> (alpha=+_z,-_z)
    spin_flip_matrix_elements_0m = spin_flip_matrix_elements_between_a_bra_and_many_kets(spin_matrix_elements_0m)

    # Compute the sum of single spin flip (along the z axis) matrix elements squared
    prob_one_flip = np.sum( np.abs( spin_flip_matrix_elements_0m['Sip_z_0m'][:,eigenvector_index] ) ** 2 + \
                            np.abs( spin_flip_matrix_elements_0m['Sim_z_0m'][:,eigenvector_index] ) ** 2
                          )

    return prob_one_flip

def probability_of_flipping_one_spin_e1e2e3(spin_matrix_elements_0m, eigenvector_index):
    """Function that computes the probability of flipping 1 spin along the e3 direction when going
    from the eigenvector |bra> to the mth eigenvector |m>, namely
        Prob(1 flip) = Sum_i Sum_{alpha=+_e3,-_e3} |<0|S_i^alpha|m>|^2
    where i indexes the lattice sites.
    Inputs:
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <0|S_i^x|m>,
                                 and similarly for the other arrays)
        eigenvector_index (integer specifying the eigenvector |m> for which to compute the probability of flipping 1 spin)
    Output:
        prob_one_flip_e1e2e3 (float; probability of flipping 1 spin along the e3 direction)"""

    # Compute the spin matrix elements in e1,e2,e3 coordinates: <0|S_i^alpha|m> (alpha=e1,e2,e3)
    spin_matrix_elements_e1e2e3_0m = spin_matrix_elements_between_a_bra_and_many_kets_e1e2e3(spin_matrix_elements_0m)

    # Compute the spin flip matrix elements for spin flips along the e3 axis: <0|S_i^alpha|m> (alpha=+_e3,-_e3)
    spin_flip_matrix_elements_e1e2e3_0m = spin_flip_matrix_elements_between_a_bra_and_many_kets_e1e2e3(spin_matrix_elements_e1e2e3_0m)

    # Compute the sum of single spin flip (along the e3 axis, i.e., perpendicular to the crystal plane) matrix elements squared
    prob_one_flip_e1e2e3 = np.sum( np.abs( spin_flip_matrix_elements_e1e2e3_0m['Sip_e3_0m'][:,eigenvector_index] ) ** 2 + \
                                   np.abs( spin_flip_matrix_elements_e1e2e3_0m['Sim_e3_0m'][:,eigenvector_index] ) ** 2
                                 )

    return prob_one_flip_e1e2e3

def probability_of_flipping_two_spins(spin_spin_correlations_0m, eigenvector_index):
    """Function that computes the intra-sublattice and inter-sublattice contributions to the probability of flipping 2 spins
    along the z direction when going from the eigenvector |bra> to the mth eigenvector |m>, namely
        Prob_intra(2 flips) = Prob_{AA}(2 flips) + Prob_{BB}(2 flips)
        Prob_inter(2 flips) = Prob_{AB}(2 flips) + Prob_{BA}(2 flips)
    where i and j index the lattice sites and
        Prob_{mu nu}(2 flips) = Sum_{i in sublattice mu} Sum_{j in sublattice nu} Sum_{alpha=+_z,-_z} Sum_{beta=+_z,-_z} |<0|S_i^alpha S_j^beta|m>|^2
    Input:
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        eigenvector_index (integer specifying the eigenvector |m> for which to compute the probability of flipping 2 spins)
    Output:
        prob_two_flips_intra, prob_two_flips_inter (floats containing the intra- and inter-sublattice contributions
                                                    to the probability of flipping 2 spins along the z direction)"""

    # Compute the spin flip-spin flip correlations for spin flips along the z axis: <0|S_i^alpha S_j^beta|m> (alpha,beta=+_z,-_z)
    spin_flip_spin_flip_correlations_0m = spin_flip_spin_flip_correlations_between_a_bra_and_many_kets(spin_spin_correlations_0m)

    # Compute the sum of two spin flips (along the z axis) matrix elements squared (for i,j intra-sublattice and i,j inter-sublattice)
    prob_two_flips_intra = np.sum( np.abs( spin_flip_spin_flip_correlations_0m['Sip_z_Sjp_z_0m'][0::2,0::2,eigenvector_index] ) ** 2 + \
                                   np.abs( spin_flip_spin_flip_correlations_0m['Sim_z_Sjm_z_0m'][1::2,1::2,eigenvector_index] ) ** 2
                                 )

    prob_two_flips_inter = np.sum( np.abs( spin_flip_spin_flip_correlations_0m['Sip_z_Sjp_z_0m'][0::2,1::2,eigenvector_index] ) ** 2 + \
                                   np.abs( spin_flip_spin_flip_correlations_0m['Sim_z_Sjm_z_0m'][1::2,0::2,eigenvector_index] ) ** 2
                                 )

    return prob_two_flips_intra, prob_two_flips_inter

def probability_of_flipping_two_spins_e1e2e3(spin_spin_correlations_0m, eigenvector_index):
    """Function that computes the intra-sublattice and inter-sublattice contributions to the probability of flipping 2 spins
    along the e3 direction when going from the eigenvector |bra> to the mth eigenvector |m>, namely
        Prob_intra(2 flips) = Prob_{AA}(2 flips) + Prob_{BB}(2 flips)
        Prob_inter(2 flips) = Prob_{AB}(2 flips) + Prob_{BA}(2 flips)
    where i and j index the lattice sites and
        Prob_{mu nu}(2 flips) = Sum_{i in sublattice mu} Sum_{j in sublattice nu} Sum_{alpha=+_e3,-_e3} Sum_{beta=+_e3,-_e3} |<0|S_i^alpha S_j^beta|m>|^2
    Input:
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        eigenvector_index (integer specifying the eigenvector |m> for which to compute the probability of flipping 2 spins)
    Output:
        prob_two_flips_intra_e1e2e3, prob_two_flips_inter_e1e2e3 (floats containing the intra- and inter-sublattice contributions
                                                                  to the probability of flipping 2 spins along the e3 direction)"""

    # Compute the spin-spin correlations in e1,e2,e3 coordinates: <0|S_i^alpha S_j^beta|m> (alpha,beta=e1,e2,e3)
    spin_spin_correlations_e1e2e3_0m = spin_spin_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_0m)

    # Compute the spin flip-spin flip correlations for spin flips along the e3 axis: <0|S_i^alpha S_j^beta|m> (alpha,beta=+_e3,-_e3)
    spin_flip_spin_flip_correlations_e1e2e3_0m = spin_flip_spin_flip_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_e1e2e3_0m)

    # Compute the sum of two spin flips (along the e3 axis) matrix elements squared (for i,j intra-sublattice and i,j inter-sublattice)
    prob_two_flips_intra_e1e2e3 = np.sum( np.abs( spin_flip_spin_flip_correlations_e1e2e3_0m['Sip_e3_Sjp_e3_0m'][0::2,0::2,eigenvector_index] ) ** 2 + \
                                          np.abs( spin_flip_spin_flip_correlations_e1e2e3_0m['Sim_e3_Sjm_e3_0m'][1::2,1::2,eigenvector_index] ) ** 2
                                        )

    prob_two_flips_inter_e1e2e3 = np.sum( np.abs( spin_flip_spin_flip_correlations_e1e2e3_0m['Sip_e3_Sjp_e3_0m'][0::2,1::2,eigenvector_index] ) ** 2 + \
                                          np.abs( spin_flip_spin_flip_correlations_e1e2e3_0m['Sim_e3_Sjm_e3_0m'][1::2,0::2,eigenvector_index] ) ** 2
                                        )

    return prob_two_flips_intra_e1e2e3, prob_two_flips_inter_e1e2e3

def magnetization(spin_matrix_elements_0m):
    """Computes the system's ground state magnetization (per site)
        m = (1/num_sites) sqrt( Sum_alpha (Sum_i <0|S_i^alpha|0>)^2 )
          = sqrt( Sum_alpha m_alpha^2 )
    and magnetization direction (expressed in x,y,z coordinates)
        m_direction = [m_x, m_y, m_z] / m
    where num_sites is the number of sites in the system, |0> is the ground state, and
        m_alpha = (1/num_sites) Sum_i <0|S_i^alpha|0>
    is the magnetization along the alpha direction.
    Inputs:
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <bra|S_i^x|ket_m>,
                                 and similarly for the other arrays)
    Output:
        m (float containing the magnitude of the magnetization per site;
           this will be a number between 0 and spin (the spin moment at each site))
        m_direction (1D NumPy array of size 3 specifying the direction of the magnetization (as a unit vector) in x,y,z coordinates;
                     if m is zero, m_direction will be the zero vector)"""

    # Number of sites
    num_sites = len(spin_matrix_elements_0m['Six_0m'][:,0])

    # Components of the magnetization along the x,y,z directions
    m_x = (1/num_sites) * np.real(np.sum(spin_matrix_elements_0m['Six_0m'][:,0]))
    m_y = (1/num_sites) * np.real(np.sum(spin_matrix_elements_0m['Siy_0m'][:,0]))
    m_z = (1/num_sites) * np.real(np.sum(spin_matrix_elements_0m['Siz_0m'][:,0]))

    # Magnetization magnitude and direction
    m = chop(np.sqrt(m_x**2 + m_y**2 + m_z**2))
    if m != 0:
        m_direction = (1/m) * np.array([m_x, m_y, m_z])
    else:
        m_direction = np.array([0, 0, 0])

    return m, m_direction

def magnetization_e1e2e3(spin_matrix_elements_0m):
    """Computes the system's ground state magnetization (per site)
        m = (1/num_sites) sqrt( Sum_alpha (Sum_i <0|S_i^alpha|0>)^2 )
          = sqrt( Sum_alpha m_alpha^2 )
    and magnetization direction (expressed in e1,e2,e3 coordinates)
        m_direction_e1e2e3 = [m_e1, m_e2, m_e3] / m
    where num_sites is the number of sites in the system, |0> is the ground state, and
        m_alpha = (1/num_sites) Sum_i <0|S_i^alpha|0>
    is the magnetization along the alpha direction.
    Inputs:
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <bra|S_i^x|ket_m>,
                                 and similarly for the other arrays)
    Output:
        m (float containing the magnitude of the magnetization per site;
           this will be a number between 0 and spin (the spin moment at each site))
        m_direction_e1e2e3 (1D NumPy array of size 3 specifying the direction of the magnetization (as a unit vector) in e1,e2,e3 coordinates;
                            if m is zero, m_direction_e1e2e3 will be the zero vector)"""

    # Number of sites
    num_sites = len(spin_matrix_elements_0m['Six_0m'][:,0])

    # Compute the spin matrix elements in e1,e2,e3 coordinates: <0|S_i^alpha|m> (alpha=e1,e2,e3)
    spin_matrix_elements_e1e2e3_0m = spin_matrix_elements_between_a_bra_and_many_kets_e1e2e3(spin_matrix_elements_0m)

    # Components of the magnetization along the e1,e2,e3 directions
    m_e1 = (1/num_sites) * np.real(np.sum(spin_matrix_elements_e1e2e3_0m['Sie1_0m'][:,0]))
    m_e2 = (1/num_sites) * np.real(np.sum(spin_matrix_elements_e1e2e3_0m['Sie2_0m'][:,0]))
    m_e3 = (1/num_sites) * np.real(np.sum(spin_matrix_elements_e1e2e3_0m['Sie3_0m'][:,0]))

    # Magnetization magnitude and direction
    m = chop(np.sqrt(m_e1**2 + m_e2**2 + m_e3**2))
    if m != 0:
        m_direction_e1e2e3 = (1/m) * np.array([m_e1, m_e2, m_e3])
    else:
        m_direction_e1e2e3 = np.array([0, 0, 0])

    return m, m_direction_e1e2e3

def magnetization_staggered(spin_matrix_elements_0m):
    """Computes the system's ground state staggered magnetization (per site)
        m_staggered = (1/num_sites) sqrt( Sum_alpha (Sum_{i in sublattice A} <0|S_i^alpha|0> - Sum_{j in sublattice B} <0|S_j^alpha|0>)^2 )
                    = sqrt( Sum_alpha m_staggered_alpha^2 )
    and staggered magnetization direction (expressed in x,y,z coordinates)
        m_staggered_direction = [m_staggered_x, m_staggered_y, m_staggered_z] / m_staggered
    where num_sites is the number of sites in the system, |0> is the ground state, and
        m_staggered_alpha = (1/num_sites) (Sum_{i in sublattice A} <0|S_i^alpha|0> - Sum_{j in sublattice B} <0|S_j^alpha|0>)^2 )
    is the staggered magnetization along the alpha direction.
    Inputs:
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <bra|S_i^x|ket_m>,
                                 and similarly for the other arrays)
    Output:
        m_staggered (float containing the magnitude of the staggered magnetization per site;
                     this will be a number between 0 and spin (the spin moment at each site))
        m_staggered_direction (1D NumPy array of size 3 specifying the direction of the
                               staggered magnetization (as a unit vector) in x,y,z coordinates;
                               if m_staggered is zero, m_staggered_direction will be the zero vector)"""

    # Number of sites
    num_sites = len(spin_matrix_elements_0m['Six_0m'][:,0])

    # Components of the staggered magnetization along the x,y,z directions
    m_staggered_x = (1/num_sites) * np.real( (np.sum(spin_matrix_elements_0m['Six_0m'][0::2,0])) - (np.sum(spin_matrix_elements_0m['Six_0m'][1::2,0])) )
    m_staggered_y = (1/num_sites) * np.real( (np.sum(spin_matrix_elements_0m['Siy_0m'][0::2,0])) - (np.sum(spin_matrix_elements_0m['Siy_0m'][1::2,0])) )
    m_staggered_z = (1/num_sites) * np.real( (np.sum(spin_matrix_elements_0m['Siz_0m'][0::2,0])) - (np.sum(spin_matrix_elements_0m['Siz_0m'][1::2,0])) )

    # Staggered magnetization magnitude and direction
    m_staggered = chop(np.sqrt(m_staggered_x**2 + m_staggered_y**2 + m_staggered_z**2))
    if m_staggered != 0:
        m_staggered_direction = (1/m_staggered) * np.array([m_staggered_x, m_staggered_y, m_staggered_z])
    else:
        m_staggered_direction = np.array([0, 0, 0])

    return m_staggered, m_staggered_direction

def magnetization_staggered_e1e2e3(spin_matrix_elements_0m):
    """Computes the system's ground state staggered magnetization (per site)
        m_staggered = (1/num_sites) sqrt( Sum_alpha (Sum_{i in sublattice A} <0|S_i^alpha|0> - Sum_{j in sublattice B} <0|S_j^alpha|0>)^2 )
                    = sqrt( Sum_alpha m_staggered_alpha^2 )
    and staggered magnetization direction (expressed in e1,e2,e3 coordinates)
        m_staggered_direction_e1e2e3 = [m_staggered_e1, m_staggered_e2, m_staggered_e3] / m_staggered
    where num_sites is the number of sites in the system, |0> is the ground state, and
        m_staggered_alpha = (1/num_sites) (Sum_{i in sublattice A} <0|S_i^alpha|0> - Sum_{j in sublattice B} <0|S_j^alpha|0>)^2 )
    is the staggered magnetization along the alpha direction.
    Inputs:
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <bra|S_i^x|ket_m>,
                                 and similarly for the other arrays)
    Output:
        m_staggered (float containing the magnitude of the staggered magnetization per site;
                     this will be a number between 0 and spin (the spin moment at each site))
        m_staggered_direction_e1e2e3 (1D NumPy array of size 3 specifying the direction of the
                                      staggered magnetization (as a unit vector) in e1,e2,e3 coordinates;
                                      if m_staggered is zero, m_staggered_direction_e1e2e3 will be the zero vector)"""

    # Number of sites
    num_sites = len(spin_matrix_elements_0m['Six_0m'][:,0])

    # Compute the spin matrix elements in e1,e2,e3 coordinates: <0|S_i^alpha|m> (alpha=e1,e2,e3)
    spin_matrix_elements_e1e2e3_0m = spin_matrix_elements_between_a_bra_and_many_kets_e1e2e3(spin_matrix_elements_0m)

    # Components of the staggered magnetization along the e1,e2,e3 directions
    m_staggered_e1 = (1/num_sites) * np.real( (np.sum(spin_matrix_elements_e1e2e3_0m['Sie1_0m'][0::2,0])) - \
                                              (np.sum(spin_matrix_elements_e1e2e3_0m['Sie1_0m'][1::2,0])) )
    m_staggered_e2 = (1/num_sites) * np.real( (np.sum(spin_matrix_elements_e1e2e3_0m['Sie2_0m'][0::2,0])) - \
                                              (np.sum(spin_matrix_elements_e1e2e3_0m['Sie2_0m'][1::2,0])) )
    m_staggered_e3 = (1/num_sites) * np.real( (np.sum(spin_matrix_elements_e1e2e3_0m['Sie3_0m'][0::2,0])) - \
                                              (np.sum(spin_matrix_elements_e1e2e3_0m['Sie3_0m'][1::2,0])) )

    # Staggered magnetization magnitude and direction
    m_staggered = chop(np.sqrt(m_staggered_e1**2 + m_staggered_e2**2 + m_staggered_e3**2))
    if m_staggered != 0:
        m_staggered_direction_e1e2e3 = (1/m_staggered) * np.array([m_staggered_e1, m_staggered_e2, m_staggered_e3])
    else:
        m_staggered_direction_e1e2e3 = np.array([0, 0, 0])

    return m_staggered, m_staggered_direction_e1e2e3

# cont here
# This function is incomplete.
def susceptibility(m_values, h_values):
    """Computes the system's ground state magnetic susceptibility (per site)
        chi(H) = dm/dH
               = [ m(H + delta_H) - m(H - delta_H) ] / ( 2 * delta_H )
    where m is the magnetization (per site), H is the external magnetic field,
    and delta_H is the spacing between successive values of H.
    Inputs:
        ?
    Output:
        ?"""

    # Number of sites
    num_sites = len(spin_matrix_elements_0m['Six_0m'][:,0])

    # Components of the magnetization along the x,y,z directions
    m_x = (1/num_sites) * np.real(np.sum(spin_matrix_elements_0m['Six_0m'][:,0]))
    m_y = (1/num_sites) * np.real(np.sum(spin_matrix_elements_0m['Siy_0m'][:,0]))
    m_z = (1/num_sites) * np.real(np.sum(spin_matrix_elements_0m['Siz_0m'][:,0]))

    # Magnetization magnitude and direction
    m = chop(np.sqrt(m_x**2 + m_y**2 + m_z**2))
    if m != 0:
        m_direction = (1/m) * np.array([m_x, m_y, m_z])
    else:
        m_direction = np.array([0, 0, 0])

    return m, m_direction

# cont here
# This function is incomplete.
def susceptibility_e1e2e3(spin_matrix_elements_0m):
    """Computes the system's ground state magnetization (per site)
        m = (1/num_sites) sqrt( Sum_alpha (Sum_i <0|S_i^alpha|0>)^2 )
          = sqrt( Sum_alpha m_alpha^2 )
    and magnetization direction (expressed in x,y,z coordinates)
        m_direction = [m_x, m_y, m_z]/m
    where num_sites is the number of sites in the system, |0> is the ground state,
    and m_alpha = (1/num_sites) Sum_i <0|S_i^alpha|0> is the magnetization along the alpha direction.
    Inputs:
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <bra|S_i^x|ket_m>,
                                 and similarly for the other arrays)
    Output:
        m (float containing the magnitude of the magnetization per site;
           this will be a number between 0 and spin (the spin moment at each site))
        m_direction (1D NumPy array of size 3 specifying the direction of the magnetization (as a unit vector) in x,y,z coordinates;
                     if m is zero, m_direction will be the zero vector)"""

    # Number of sites
    num_sites = len(spin_matrix_elements_0m['Six_0m'][:,0])

    # Components of the magnetization along the x,y,z directions
    m_x = (1/num_sites) * np.real(np.sum(spin_matrix_elements_0m['Six_0m'][:,0]))
    m_y = (1/num_sites) * np.real(np.sum(spin_matrix_elements_0m['Siy_0m'][:,0]))
    m_z = (1/num_sites) * np.real(np.sum(spin_matrix_elements_0m['Siz_0m'][:,0]))

    # Magnetization magnitude and direction
    m = chop(np.sqrt(m_x**2 + m_y**2 + m_z**2))
    if m != 0:
        m_direction = (1/m) * np.array([m_x, m_y, m_z])
    else:
        m_direction = np.array([0, 0, 0])

    return m, m_direction

def one_magnon_DOS(omega_values, ground_state_eigenvalue, list_of_eigenvalues, spin_matrix_elements_0m, Nx, Ny, m_min = 1, m_max = None, eta = 0.01):
    """Function that computes the single-magnon density of states (DOS), namely
        S^alpha(omega) = 2 pi Sum_{m=m_min}^{m_max-1} Sum_i |<m|S_i^alpha|0>|^2 delta(omega - (E_m - E_0))
    for alpha=+_e3,-_e3,e3, where m indexes the energy eigenvalues E_m and eigenvectors |m>,
    i indexes the lattice sites in sublattice mu,
    and omega is the energy of a given excitation.
    Inputs:
        omega_values (1D NumPy array of size N_omega)
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <0|S_i^x|m>,
                                 and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_min, m_max (indices of the minimum and maximum (minus one) eigenvectors |m> to sum over;
                      (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_min=0 and m_max=400)
                      if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
        eta (float specifying the half-width at half-maximum of the Lorentzian distributions used for the delta functions)
    Outputs:
        one_magnon_DOS_plus_contribution,
        one_magnon_DOS_minus_contribution,
        one_magnon_DOS_e3_contribution (1D NumPy arrays of size N_omega containing the alpha = +,-,e3 contributions
                                        to the one-magnon DOS with momentum integrated out, described above;
                                        the epsilonth element of each array corresponds to the value of the
                                        one-magnon DOS alpha contribution at energy value omega = omega_values[epsilon])"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Compute the spin matrix elements in e1,e2,e3 coordinates: <0|S_i^alpha|m> (alpha=e1,e2,e3)
    spin_matrix_elements_e1e2e3_0m = spin_matrix_elements_between_a_bra_and_many_kets_e1e2e3(spin_matrix_elements_0m)

    # Compute the spin matrix elements in e1,e2,e3 coordinates: <m|S_i^alpha|0> (alpha=e1,e2,e3)
    spin_matrix_elements_e1e2e3_m0 = spin_matrix_elements_between_many_bras_and_a_ket_e1e2e3(spin_matrix_elements_e1e2e3_0m)

    # Compute the spin flip matrix elements for spin flips along the e3 axis: <0|S_i^alpha|m> (alpha=+_e3,-_e3)
    spin_flip_matrix_elements_e1e2e3_0m = spin_flip_matrix_elements_between_a_bra_and_many_kets_e1e2e3(spin_matrix_elements_e1e2e3_0m)

    # Compute the spin flip matrix elements for spin flips along the e3 axis: <m|S_i^alpha|0> (alpha=+_e3,-_e3)
    spin_flip_matrix_elements_e1e2e3_m0 = spin_flip_matrix_elements_between_many_bras_and_a_ket_e1e2e3(spin_flip_matrix_elements_e1e2e3_0m)

    # Compute the sum of single spin flip (along the e3 axis, i.e., perpendicular to the crystal plane) matrix elements squared
    # (Note: These are 1D NumPy arrays of size num_eigenstates_to_sum_over = m_max - m_min;
    #        the mth entry of each array is
    #            sum_of_single_site_spin_flip_matrix_elements_squared_alpha_contribution[m] = Sum_i Sum_{alpha=+_e3,-_e3} |<m|S_i^alpha|0>|^2
    #        where m indexes the energy eigenvalues E_m and eigenvectors |m>, and i indexes the lattice sites)
    sum_of_single_site_spin_flip_matrix_elements_squared_plus_contribution = \
        np.sum( np.abs( spin_flip_matrix_elements_e1e2e3_m0['Sip_e3_m0'][:,m_min:m_max] ) ** 2,
                axis=0
              )

    sum_of_single_site_spin_flip_matrix_elements_squared_minus_contribution = \
        np.sum( np.abs( spin_flip_matrix_elements_e1e2e3_m0['Sim_e3_m0'][:,m_min:m_max] ) ** 2,
                axis=0
              )

    sum_of_single_site_spin_flip_matrix_elements_squared_e3_contribution = \
        np.sum( np.abs( spin_matrix_elements_e1e2e3_m0['Sie3_m0'][:,m_min:m_max] ) ** 2,
                axis=0
              )

    # Make the 2D NumPy array energy_delta_function of shape ( (m_max - m_min + 1) x N_omega );
    # the (m,e)th element of this array is delta(omega_e - (E_m - E_0)), where
    # omega_e = omega_values[e], E_m = list_of_eigenvalues[m], and E_0 is the ground-state energy eigenvalue
    energy_delta_function = lorentzian(omega_values[np.newaxis,:], list_of_eigenvalues[m_min:m_max][:,np.newaxis] - ground_state_eigenvalue, eta)

    # Get the contributions to the single-magnon DOS with momentum integrated out from alpha = +_e3,-_e3,e3
    one_magnon_DOS_plus_contribution = \
        2*np.pi * sum_of_single_site_spin_flip_matrix_elements_squared_plus_contribution @ energy_delta_function

    one_magnon_DOS_minus_contribution = \
        2*np.pi * sum_of_single_site_spin_flip_matrix_elements_squared_minus_contribution @ energy_delta_function

    one_magnon_DOS_e3_contribution = \
        2*np.pi * sum_of_single_site_spin_flip_matrix_elements_squared_e3_contribution @ energy_delta_function

    return one_magnon_DOS_plus_contribution, \
           one_magnon_DOS_minus_contribution, \
           one_magnon_DOS_e3_contribution

def one_magnon_DOS_from_two_sites(site_1_index, site_2_index, omega_values, ground_state_eigenvalue, list_of_eigenvalues, spin_matrix_elements_0m, Nx, Ny, m_min = 1, m_max = None, eta = 0.01):
    """Function that approximately computes the single-magnon density of states (DOS) using information from two sites, namely
        S^alpha(omega) ~=~ pi N Sum_{m=m_min}^{m_max-1} ( |<m|S_1^alpha|0>|^2 + |<m|S_2^alpha|0>|^2 ) delta(omega - (E_m - E_0))
    for alpha=+_e3,-_e3,e3, where ~=~ means "is approximately equal to",
    N is the number of sites,
    m indexes the energy eigenvalues E_m and eigenvectors |m>,
    1 and 2 index the two sites to use (preferably from different sublattices),
    and omega is the energy of a given excitation.
    Inputs:
        site_1_index, site_2_index (integers specifying the site indices of the two sites to use;
                                    preferably, one will be even and the other one odd
                                    (so that they belong to different sublattices))
        omega_values (1D NumPy array of size N_omega)
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <0|S_i^x|m>,
                                 and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_min, m_max (indices of the minimum and maximum (minus one) eigenvectors |m> to sum over;
                      (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_min=0 and m_max=400)
                      if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
        eta (float specifying the half-width at half-maximum of the Lorentzian distributions used for the delta functions)
    Outputs:
        one_magnon_DOS_plus_contribution,
        one_magnon_DOS_minus_contribution,
        one_magnon_DOS_e3_contribution (1D NumPy arrays of size N_omega containing the alpha = +,-,e3 contributions
                                        to the one-magnon DOS with momentum integrated out, described above;
                                        the epsilonth element of each array corresponds to the value of the
                                        one-magnon DOS alpha contribution at energy value omega = omega_values[epsilon])"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Compute the spin matrix elements in e1,e2,e3 coordinates: <0|S_i^alpha|m> (alpha=e1,e2,e3)
    spin_matrix_elements_e1e2e3_0m = spin_matrix_elements_between_a_bra_and_many_kets_e1e2e3(spin_matrix_elements_0m)

    # Compute the spin matrix elements in e1,e2,e3 coordinates: <m|S_i^alpha|0> (alpha=e1,e2,e3)
    spin_matrix_elements_e1e2e3_m0 = spin_matrix_elements_between_many_bras_and_a_ket_e1e2e3(spin_matrix_elements_e1e2e3_0m)

    # Compute the spin flip matrix elements for spin flips along the e3 axis: <0|S_i^alpha|m> (alpha=+_e3,-_e3)
    spin_flip_matrix_elements_e1e2e3_0m = spin_flip_matrix_elements_between_a_bra_and_many_kets_e1e2e3(spin_matrix_elements_e1e2e3_0m)

    # Compute the spin flip matrix elements for spin flips along the e3 axis: <m|S_i^alpha|0> (alpha=+_e3,-_e3)
    spin_flip_matrix_elements_e1e2e3_m0 = spin_flip_matrix_elements_between_many_bras_and_a_ket_e1e2e3(spin_flip_matrix_elements_e1e2e3_0m)

    # Compute the sum of single spin flip (along the e3 axis, i.e., perpendicular to the crystal plane) matrix elements squared
    # (Note: These are 1D NumPy arrays of size num_eigenstates_to_sum_over = m_max - m_min;
    #        the mth entry of each array is
    #            sum_of_single_site_spin_flip_matrix_elements_squared_alpha_contribution[m] ~=~ N/2 Sum_{alpha=+_e3,-_e3} ( |<m|S_1^alpha|0>|^2 + |<m|S_2^alpha|0>|^2 )
    #        where m indexes the energy eigenvalues E_m and eigenvectors |m>)
    sum_of_single_site_spin_flip_matrix_elements_squared_plus_contribution = \
        (num_sites/2) * ( np.abs( spin_flip_matrix_elements_e1e2e3_m0['Sip_e3_m0'][site_1_index,m_min:m_max] ) ** 2 + \
                          np.abs( spin_flip_matrix_elements_e1e2e3_m0['Sip_e3_m0'][site_2_index,m_min:m_max] ) ** 2 )

    sum_of_single_site_spin_flip_matrix_elements_squared_minus_contribution = \
        (num_sites/2) * ( np.abs( spin_flip_matrix_elements_e1e2e3_m0['Sim_e3_m0'][site_1_index,m_min:m_max] ) ** 2 + \
                          np.abs( spin_flip_matrix_elements_e1e2e3_m0['Sim_e3_m0'][site_2_index,m_min:m_max] ) ** 2 )

    sum_of_single_site_spin_flip_matrix_elements_squared_e3_contribution = \
        (num_sites/2) * ( np.abs( spin_matrix_elements_e1e2e3_m0['Sie3_m0'][site_1_index,m_min:m_max] ) ** 2 + \
                          np.abs( spin_matrix_elements_e1e2e3_m0['Sie3_m0'][site_2_index,m_min:m_max] ) ** 2 )

    # Make the 2D NumPy array energy_delta_function of shape ( (m_max - m_min + 1) x N_omega );
    # the (m,e)th element of this array is delta(omega_e - (E_m - E_0)), where
    # omega_e = omega_values[e], E_m = list_of_eigenvalues[m], and E_0 is the ground-state energy eigenvalue
    energy_delta_function = lorentzian(omega_values[np.newaxis,:], list_of_eigenvalues[m_min:m_max][:,np.newaxis] - ground_state_eigenvalue, eta)

    # Get the contributions to the single-magnon DOS with momentum integrated out from alpha = +_e3,-_e3,e3
    one_magnon_DOS_plus_contribution = \
        2*np.pi * sum_of_single_site_spin_flip_matrix_elements_squared_plus_contribution @ energy_delta_function

    one_magnon_DOS_minus_contribution = \
        2*np.pi * sum_of_single_site_spin_flip_matrix_elements_squared_minus_contribution @ energy_delta_function

    one_magnon_DOS_e3_contribution = \
        2*np.pi * sum_of_single_site_spin_flip_matrix_elements_squared_e3_contribution @ energy_delta_function

    return one_magnon_DOS_plus_contribution, \
           one_magnon_DOS_minus_contribution, \
           one_magnon_DOS_e3_contribution

def one_magnon_DOS_from_one_site(site_1_index, omega_values, ground_state_eigenvalue, list_of_eigenvalues, spin_matrix_elements_0m, Nx, Ny, m_min = 1, m_max = None, eta = 0.01):
    """Function that approximately computes the single-magnon density of states (DOS) using information from one site, namely
        S^alpha(omega) ~=~ 2 pi N Sum_{m=m_min}^{m_max-1} |<m|S_1^alpha|0>|^2 delta(omega - (E_m - E_0))
    for alpha=+_e3,-_e3,e3, where ~=~ means "is approximately equal to",
    N is the number of sites,
    m indexes the energy eigenvalues E_m and eigenvectors |m>,
    1 indexes the site to use,
    and omega is the energy of a given excitation.
    Inputs:
        site_1_index, site_2_index (integers specifying the site indices of the two sites to use;
                                    preferably, one will be even and the other one odd
                                    (so that they belong to different sublattices))
        omega_values (1D NumPy array of size N_omega)
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <0|S_i^x|m>,
                                 and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_min, m_max (indices of the minimum and maximum (minus one) eigenvectors |m> to sum over;
                      (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_min=0 and m_max=400)
                      if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
        eta (float specifying the half-width at half-maximum of the Lorentzian distributions used for the delta functions)
    Outputs:
        one_magnon_DOS_plus_contribution,
        one_magnon_DOS_minus_contribution,
        one_magnon_DOS_e3_contribution (1D NumPy arrays of size N_omega containing the alpha = +,-,e3 contributions
                                        to the one-magnon DOS with momentum integrated out, described above;
                                        the epsilonth element of each array corresponds to the value of the
                                        one-magnon DOS alpha contribution at energy value omega = omega_values[epsilon])"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Compute the spin matrix elements in e1,e2,e3 coordinates: <0|S_i^alpha|m> (alpha=e1,e2,e3)
    spin_matrix_elements_e1e2e3_0m = spin_matrix_elements_between_a_bra_and_many_kets_e1e2e3(spin_matrix_elements_0m)

    # Compute the spin matrix elements in e1,e2,e3 coordinates: <m|S_i^alpha|0> (alpha=e1,e2,e3)
    spin_matrix_elements_e1e2e3_m0 = spin_matrix_elements_between_many_bras_and_a_ket_e1e2e3(spin_matrix_elements_e1e2e3_0m)

    # Compute the spin flip matrix elements for spin flips along the e3 axis: <0|S_i^alpha|m> (alpha=+_e3,-_e3)
    spin_flip_matrix_elements_e1e2e3_0m = spin_flip_matrix_elements_between_a_bra_and_many_kets_e1e2e3(spin_matrix_elements_e1e2e3_0m)

    # Compute the spin flip matrix elements for spin flips along the e3 axis: <m|S_i^alpha|0> (alpha=+_e3,-_e3)
    spin_flip_matrix_elements_e1e2e3_m0 = spin_flip_matrix_elements_between_many_bras_and_a_ket_e1e2e3(spin_flip_matrix_elements_e1e2e3_0m)

    # Compute the sum of single spin flip (along the e3 axis, i.e., perpendicular to the crystal plane) matrix elements squared
    # (Note: These are 1D NumPy arrays of size num_eigenstates_to_sum_over = m_max - m_min;
    #        the mth entry of each array is
    #            sum_of_single_site_spin_flip_matrix_elements_squared_alpha_contribution[m] ~=~ N/2 Sum_{alpha=+_e3,-_e3} ( |<m|S_1^alpha|0>|^2 + |<m|S_2^alpha|0>|^2 )
    #        where m indexes the energy eigenvalues E_m and eigenvectors |m>)
    sum_of_single_site_spin_flip_matrix_elements_squared_plus_contribution = \
        num_sites * np.abs( spin_flip_matrix_elements_e1e2e3_m0['Sip_e3_m0'][site_1_index,m_min:m_max] ) ** 2

    sum_of_single_site_spin_flip_matrix_elements_squared_minus_contribution = \
        num_sites * np.abs( spin_flip_matrix_elements_e1e2e3_m0['Sim_e3_m0'][site_1_index,m_min:m_max] ) ** 2

    sum_of_single_site_spin_flip_matrix_elements_squared_e3_contribution = \
        num_sites * np.abs( spin_matrix_elements_e1e2e3_m0['Sie3_m0'][site_1_index,m_min:m_max] ) ** 2

    # Make the 2D NumPy array energy_delta_function of shape ( (m_max - m_min + 1) x N_omega );
    # the (m,e)th element of this array is delta(omega_e - (E_m - E_0)), where
    # omega_e = omega_values[e], E_m = list_of_eigenvalues[m], and E_0 is the ground-state energy eigenvalue
    energy_delta_function = lorentzian(omega_values[np.newaxis,:], list_of_eigenvalues[m_min:m_max][:,np.newaxis] - ground_state_eigenvalue, eta)

    # Get the contributions to the single-magnon DOS with momentum integrated out from alpha = +_e3,-_e3,e3
    one_magnon_DOS_plus_contribution = \
        2*np.pi * sum_of_single_site_spin_flip_matrix_elements_squared_plus_contribution @ energy_delta_function

    one_magnon_DOS_minus_contribution = \
        2*np.pi * sum_of_single_site_spin_flip_matrix_elements_squared_minus_contribution @ energy_delta_function

    one_magnon_DOS_e3_contribution = \
        2*np.pi * sum_of_single_site_spin_flip_matrix_elements_squared_e3_contribution @ energy_delta_function

    return one_magnon_DOS_plus_contribution, \
           one_magnon_DOS_minus_contribution, \
           one_magnon_DOS_e3_contribution

def two_magnon_DOS(omega_values, ground_state_eigenvalue, list_of_eigenvalues, spin_spin_correlations_0m, Nx, Ny, m_min = 1, m_max = None, eta = 0.01, boundary_conditions = 'periodic'):
    """Function that computes the two-magnon density of states (DOS), namely
        P^alpha(omega) = 2 pi Sum_{m=m_min}^{m_max-1} Sum_<ij> |<m|S_i^alpha S_j^alpha|0>|^2 delta(omega - (E_m - E_0))
    for alpha=+_e3,-_e3,e3, where m indexes the energy eigenvalues E_m and eigenvectors |m>,
    <ij> indexes a sum over neighboring sites,
    and omega is the energy of a given excitation.
    Inputs:
        omega_values (1D NumPy array of size N_omega)
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_min, m_max (indices of the minimum and maximum (minus one) eigenvectors |m> to sum over;
                      (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_min=0 and m_max=400)
                      if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
        eta (float specifying the half-width at half-maximum of the Lorentzian distributions used for the delta functions)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Outputs:
        two_magnon_DOS_plus_contribution,
        two_magnon_DOS_minus_contribution,
        two_magnon_DOS_e3_contribution (1D NumPy arrays of size N_omega containing the alpha = +,-,e3 contributions
                                        to the two-magnon DOS, described above;
                                        the epsilonth element of each array corresponds to the value of the
                                        two-magnon DOS alpha contribution at energy value omega = omega_values[epsilon])"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Number of energy eigenvalues/eigenvectors in list_of_eigenvalues/list_of_eigenvectors
    num_eigenstates_to_sum_over = len(list_of_eigenvalues[m_min:m_max])

    # Compute the spin-spin correlations in e1,e2,e3 coordinates: <0|S_i^alpha S_j^beta|m> (alpha,beta=e1,e2,e3)
    spin_spin_correlations_e1e2e3_0m = spin_spin_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_0m)

    # Compute the spin-spin correlations in e1,e2,e3 coordinates: <m|S_i^alpha S_j^beta|0> (alpha,beta=e1,e2,e3)
    spin_spin_correlations_e1e2e3_m0 = spin_spin_correlations_between_many_bras_and_a_ket_e1e2e3(spin_spin_correlations_e1e2e3_0m)

    # Compute the spin flip-spin flip correlations for spin flips along the e3 axis: <0|S_i^alpha S_j^beta|m> (alpha,beta=+_e3,-_e3)
    spin_flip_spin_flip_correlations_e1e2e3_0m = spin_flip_spin_flip_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_e1e2e3_0m)

    # Compute the spin flip-spin flip correlations for spin flips along the e3 axis: <m|S_i^alpha S_j^beta|0> (alpha,beta=+_e3,-_e3)
    spin_flip_spin_flip_correlations_e1e2e3_m0 = spin_flip_spin_flip_correlations_between_many_bras_and_a_ket_e1e2e3(spin_flip_spin_flip_correlations_e1e2e3_0m)

    # Initialize the +,-,e3 contributions to the sum of neighboring sites' spin flip-spin flip correlations squared
    # (along the e3 axis, i.e., perpendicular to the crystal plane)
    # (Note: These are 1D NumPy arrays of size num_eigenstates_to_sum_over = m_max - m_min;
    #        the mth entry of each array is
    #            sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_alpha_contribution[m] = Sum_<ij> |<m|S_i^alpha S_j^alpha|0>|^2
    #        where m indexes the energy eigenvalues E_m and eigenvectors |m>, and <ij> index neighboring lattice sites)
    sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_plus_contribution = np.full(num_eigenstates_to_sum_over, 0.)
    sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_minus_contribution = np.full(num_eigenstates_to_sum_over, 0.)
    sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_e3_contribution = np.full(num_eigenstates_to_sum_over, 0.)

    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site index of the sublattice B site in this unit cell
            site_index = get_site_index(Nx, i, j, 1)

            # Loop over the 3 bonds for the B site of this unit cell
            # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
            for bond_type in range(3):

                # Initialize Boolean that specifies whether or not this bond is a boundary bond
                # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
                boundary_bond = False

                # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
                if bond_type == 0:
                    i_neighbor = i
                    j_neighbor = (j + 1) % Ny
                    if j_neighbor != (j + 1):
                        boundary_bond = True
                elif bond_type == 1:
                    i_neighbor = (i - 1) % Nx
                    j_neighbor = (j + 1) % Ny
                    if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                        boundary_bond = True
                elif bond_type == 2:
                    i_neighbor = i
                    j_neighbor = j

                # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
                # and this is a boundary bond
                if (boundary_conditions == 'open') and (boundary_bond == True):
                    continue

                # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
                site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

                # Add the contributions from these neighboring sites
                sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_plus_contribution += \
                    np.abs( spin_flip_spin_flip_correlations_e1e2e3_m0['Sip_e3_Sjp_e3_m0'][site_index,site_index_neighbor,m_min:m_max] ) ** 2

                sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_minus_contribution += \
                    np.abs( spin_flip_spin_flip_correlations_e1e2e3_m0['Sim_e3_Sjm_e3_m0'][site_index,site_index_neighbor,m_min:m_max] ) ** 2

                sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_e3_contribution += \
                    np.abs( spin_spin_correlations_e1e2e3_m0['Sie3Sje3_m0'][site_index,site_index_neighbor,m_min:m_max] ) ** 2

    # Make the 2D NumPy array energy_delta_function of shape ( num_eigenstates_to_sum_over x N_omega );
    # the (m,e)th element of this array is delta(omega_e - (E_m - E_0)), where
    # omega_e = omega_values[e], E_m = list_of_eigenvalues[m], and E_0 is the ground-state energy eigenvalue
    energy_delta_function = lorentzian(omega_values[np.newaxis,:], list_of_eigenvalues[m_min:m_max][:,np.newaxis] - ground_state_eigenvalue, eta)

    # Get the contributions to the two-magnon DOS from alpha = +_e3,-_e3,e3
    two_magnon_DOS_plus_contribution = 2*np.pi * sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_plus_contribution @ energy_delta_function
    two_magnon_DOS_minus_contribution = 2*np.pi * sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_minus_contribution @ energy_delta_function
    two_magnon_DOS_e3_contribution = 2*np.pi * sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_e3_contribution @ energy_delta_function

    return two_magnon_DOS_plus_contribution, \
           two_magnon_DOS_minus_contribution, \
           two_magnon_DOS_e3_contribution

def two_magnon_DOS_from_three_bonds(unit_cell_i_index, unit_cell_j_index, omega_values, ground_state_eigenvalue, list_of_eigenvalues, spin_spin_correlations_0m, Nx, Ny, m_min = 1, m_max = None, eta = 0.01, boundary_conditions = 'periodic'):
    """Function that approximately computes the two-magnon density of states (DOS) using information from the three bonds at a site, namely
        P^alpha(omega) ~=~ pi N Sum_{m=m_min}^{m_max-1} Sum_{gamma=x,y,z} |<m|S_1^alpha S_{1-delta_gamma}^alpha|0>|^2 delta(omega - (E_m - E_0))
    for alpha=+_e3,-_e3,e3, where where ~=~ means "is approximately equal to",
    N is the number of sites,
    m indexes the energy eigenvalues E_m and eigenvectors |m>,
    1 indexes the sublattice B site whose three bonds we will use,
    S_{1-delta_gamma}^alpha denotes the alpha component of the spin operator acting on
                            the sublattice A site interacting with site 1 through a gamma-bond,
    and omega is the energy of a given excitation.
    Inputs:
        unit_cell_i_index, unit_cell_j_index (integers specifying the unit cell indices of the sublattice B site
                                              whose three bonds we will use; the unit cell indices are the coordinates
                                              along the two lattice directions a1 and a2)
        omega_values (1D NumPy array of size N_omega)
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_min, m_max (indices of the minimum and maximum (minus one) eigenvectors |m> to sum over;
                      (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_min=0 and m_max=400)
                      if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
        eta (float specifying the half-width at half-maximum of the Lorentzian distributions used for the delta functions)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Outputs:
        two_magnon_DOS_plus_contribution,
        two_magnon_DOS_minus_contribution,
        two_magnon_DOS_e3_contribution (1D NumPy arrays of size N_omega containing the alpha = +,-,e3 contributions
                                        to the two-magnon DOS, described above;
                                        the epsilonth element of each array corresponds to the value of the
                                        two-magnon DOS alpha contribution at energy value omega = omega_values[epsilon])
    Note: This function assumes periodic boundary conditions; if this system's boundary conditions are open,
          then the sublattice B site in (unit_cell_i_index, unit_cell_j_index) must not be at the boundary,
          otherwise this function will return nonsense."""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Number of energy eigenvalues/eigenvectors in list_of_eigenvalues/list_of_eigenvectors
    num_eigenstates_to_sum_over = len(list_of_eigenvalues[m_min:m_max])

    # Compute the spin-spin correlations in e1,e2,e3 coordinates: <0|S_i^alpha S_j^beta|m> (alpha,beta=e1,e2,e3)
    spin_spin_correlations_e1e2e3_0m = spin_spin_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_0m)

    # Compute the spin-spin correlations in e1,e2,e3 coordinates: <m|S_i^alpha S_j^beta|0> (alpha,beta=e1,e2,e3)
    spin_spin_correlations_e1e2e3_m0 = spin_spin_correlations_between_many_bras_and_a_ket_e1e2e3(spin_spin_correlations_e1e2e3_0m)

    # Compute the spin flip-spin flip correlations for spin flips along the e3 axis: <0|S_i^alpha S_j^beta|m> (alpha,beta=+_e3,-_e3)
    spin_flip_spin_flip_correlations_e1e2e3_0m = spin_flip_spin_flip_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_e1e2e3_0m)

    # Compute the spin flip-spin flip correlations for spin flips along the e3 axis: <m|S_i^alpha S_j^beta|0> (alpha,beta=+_e3,-_e3)
    spin_flip_spin_flip_correlations_e1e2e3_m0 = spin_flip_spin_flip_correlations_between_many_bras_and_a_ket_e1e2e3(spin_flip_spin_flip_correlations_e1e2e3_0m)

    # Initialize the +,-,e3 contributions to the sum of neighboring sites' spin flip-spin flip correlations squared
    # (along the e3 axis, i.e., perpendicular to the crystal plane)
    # (Note: These are 1D NumPy arrays of size num_eigenstates_to_sum_over = m_max - m_min;
    #        the mth entry of each array is
    #            sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_alpha_contribution[m] ~=~ N/2 Sum_{gamma=x,y,z} |<m|S_1^alpha S_{1-delta_gamma}^alpha|0>|^2
    #        where m indexes the energy eigenvalues E_m and eigenvectors |m>)
    sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_plus_contribution = np.full(num_eigenstates_to_sum_over, 0.)
    sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_minus_contribution = np.full(num_eigenstates_to_sum_over, 0.)
    sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_e3_contribution = np.full(num_eigenstates_to_sum_over, 0.)

    # Give the unit cell indices shorter names for brevity
    i = unit_cell_i_index
    j = unit_cell_j_index

    # Site index of the sublattice B site whose three bonds we will use
    site_index = get_site_index(Nx, i, j, 1)

    # Loop over the 3 bonds for this B site
    # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
    for bond_type in range(3):

        # Initialize Boolean that specifies whether or not this bond is a boundary bond
        # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
        boundary_bond = False

        # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
        if bond_type == 0:
            i_neighbor = i
            j_neighbor = (j + 1) % Ny
            if j_neighbor != (j + 1):
                boundary_bond = True
        elif bond_type == 1:
            i_neighbor = (i - 1) % Nx
            j_neighbor = (j + 1) % Ny
            if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                boundary_bond = True
        elif bond_type == 2:
            i_neighbor = i
            j_neighbor = j

        # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
        # and this is a boundary bond
        if (boundary_conditions == 'open') and (boundary_bond == True):
            print("Error: The system has open boundary conditions, but this sublattice B site is at the system's boundary; this function will return nonsense.")

        # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
        site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

        # Add the contributions from these neighboring sites
        sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_plus_contribution += \
            (num_sites/2) * np.abs( spin_flip_spin_flip_correlations_e1e2e3_m0['Sip_e3_Sjp_e3_m0'][site_index,site_index_neighbor,m_min:m_max] ) ** 2

        sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_minus_contribution += \
            (num_sites/2) * np.abs( spin_flip_spin_flip_correlations_e1e2e3_m0['Sim_e3_Sjm_e3_m0'][site_index,site_index_neighbor,m_min:m_max] ) ** 2

        sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_e3_contribution += \
            (num_sites/2) * np.abs( spin_spin_correlations_e1e2e3_m0['Sie3Sje3_m0'][site_index,site_index_neighbor,m_min:m_max] ) ** 2

    # Make the 2D NumPy array energy_delta_function of shape ( num_eigenstates_to_sum_over x N_omega );
    # the (m,e)th element of this array is delta(omega_e - (E_m - E_0)), where
    # omega_e = omega_values[e], E_m = list_of_eigenvalues[m], and E_0 is the ground-state energy eigenvalue
    energy_delta_function = lorentzian(omega_values[np.newaxis,:], list_of_eigenvalues[m_min:m_max][:,np.newaxis] - ground_state_eigenvalue, eta)

    # Get the contributions to the two-magnon DOS from alpha = +_e3,-_e3,e3
    two_magnon_DOS_plus_contribution = 2*np.pi * sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_plus_contribution @ energy_delta_function
    two_magnon_DOS_minus_contribution = 2*np.pi * sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_minus_contribution @ energy_delta_function
    two_magnon_DOS_e3_contribution = 2*np.pi * sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_e3_contribution @ energy_delta_function

    return two_magnon_DOS_plus_contribution, \
           two_magnon_DOS_minus_contribution, \
           two_magnon_DOS_e3_contribution

def two_magnon_DOS_from_one_bond(unit_cell_i_index, unit_cell_j_index, bond_type, omega_values, ground_state_eigenvalue, list_of_eigenvalues, spin_spin_correlations_0m, Nx, Ny, m_min = 1, m_max = None, eta = 0.01, boundary_conditions = 'periodic'):
    """Function that (VERY crudely!) approximately computes the two-magnon density of states (DOS) using information from one bond, namely
        P^alpha(omega) ~=~ 3 pi N Sum_{m=m_min}^{m_max-1} |<m|S_1^alpha S_{1-delta_gamma}^alpha|0>|^2 delta(omega - (E_m - E_0))
    for alpha=+_e3,-_e3,e3, where where ~=~ means "is approximately equal to",
    N is the number of sites,
    m indexes the energy eigenvalues E_m and eigenvectors |m>,
    1 indexes the sublattice B site whose three bonds we will use,
    S_{1-delta_gamma}^alpha denotes the alpha component of the spin operator acting on
                            the sublattice A site interacting with site 1 through a gamma-bond
                            (gamma is specified in the input variable bond_type),
    and omega is the energy of a given excitation.
    Inputs:
        unit_cell_i_index, unit_cell_j_index (integers specifying the unit cell indices of the sublattice B site
                                              whose bond we will use; the unit cell indices are the coordinates
                                              along the two lattice directions a1 and a2)
        bond_type (integer specifying the type of bond to use;
                   0 corresponds to x-bonds, 1 to y-bonds, and 2 to z-bonds)
        omega_values (1D NumPy array of size N_omega)
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_min, m_max (indices of the minimum and maximum (minus one) eigenvectors |m> to sum over;
                      (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_min=0 and m_max=400)
                      if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
        eta (float specifying the half-width at half-maximum of the Lorentzian distributions used for the delta functions)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Outputs:
        two_magnon_DOS_plus_contribution,
        two_magnon_DOS_minus_contribution,
        two_magnon_DOS_e3_contribution (1D NumPy arrays of size N_omega containing the alpha = +,-,e3 contributions
                                        to the two-magnon DOS, described above;
                                        the epsilonth element of each array corresponds to the value of the
                                        two-magnon DOS alpha contribution at energy value omega = omega_values[epsilon])
    Note 1: This function assumes periodic boundary conditions; if this system's boundary conditions are open,
            then bond at the sublattice B site in (unit_cell_i_index, unit_cell_j_index) must not be at the boundary,
            otherwise this function will return nonsense. To be safe, make bond_type=2 (z-bond), since z-bonds are never boundary bonds
            because of the way I've defined the lattice.
    Note 2: This function will give a VERY crude approximation of the two-magnon DOS. It is recommended that you use the function
            two_magnon_DOS_from_three_bonds, which gives a much better approximation."""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Number of energy eigenvalues/eigenvectors in list_of_eigenvalues/list_of_eigenvectors
    num_eigenstates_to_sum_over = len(list_of_eigenvalues[m_min:m_max])

    # Compute the spin-spin correlations in e1,e2,e3 coordinates: <0|S_i^alpha S_j^beta|m> (alpha,beta=e1,e2,e3)
    spin_spin_correlations_e1e2e3_0m = spin_spin_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_0m)

    # Compute the spin-spin correlations in e1,e2,e3 coordinates: <m|S_i^alpha S_j^beta|0> (alpha,beta=e1,e2,e3)
    spin_spin_correlations_e1e2e3_m0 = spin_spin_correlations_between_many_bras_and_a_ket_e1e2e3(spin_spin_correlations_e1e2e3_0m)

    # Compute the spin flip-spin flip correlations for spin flips along the e3 axis: <0|S_i^alpha S_j^beta|m> (alpha,beta=+_e3,-_e3)
    spin_flip_spin_flip_correlations_e1e2e3_0m = spin_flip_spin_flip_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_e1e2e3_0m)

    # Compute the spin flip-spin flip correlations for spin flips along the e3 axis: <m|S_i^alpha S_j^beta|0> (alpha,beta=+_e3,-_e3)
    spin_flip_spin_flip_correlations_e1e2e3_m0 = spin_flip_spin_flip_correlations_between_many_bras_and_a_ket_e1e2e3(spin_flip_spin_flip_correlations_e1e2e3_0m)

    # Initialize the +,-,e3 contributions to the sum of neighboring sites' spin flip-spin flip correlations squared
    # (along the e3 axis, i.e., perpendicular to the crystal plane)
    # (Note: These are 1D NumPy arrays of size num_eigenstates_to_sum_over = m_max - m_min;
    #        the mth entry of each array is
    #            sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_alpha_contribution[m] ~=~ N/2 Sum_{gamma=x,y,z} |<m|S_1^alpha S_{1-delta_gamma}^alpha|0>|^2
    #        where m indexes the energy eigenvalues E_m and eigenvectors |m>)
    sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_plus_contribution = np.full(num_eigenstates_to_sum_over, 0.)
    sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_minus_contribution = np.full(num_eigenstates_to_sum_over, 0.)
    sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_e3_contribution = np.full(num_eigenstates_to_sum_over, 0.)

    # Give the unit cell indices shorter names for brevity
    i = unit_cell_i_index
    j = unit_cell_j_index

    # Site index of the sublattice B site whose three bonds we will use
    site_index = get_site_index(Nx, i, j, 1)

    # Initialize Boolean that specifies whether or not this bond is a boundary bond
    # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
    boundary_bond = False

    # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
    if bond_type == 0:
        i_neighbor = i
        j_neighbor = (j + 1) % Ny
        if j_neighbor != (j + 1):
            boundary_bond = True
    elif bond_type == 1:
        i_neighbor = (i - 1) % Nx
        j_neighbor = (j + 1) % Ny
        if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
            boundary_bond = True
    elif bond_type == 2:
        i_neighbor = i
        j_neighbor = j

    # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
    # and this is a boundary bond
    if (boundary_conditions == 'open') and (boundary_bond == True):
        print("Error: The system has open boundary conditions, but this sublattice B site is at the system's boundary; this function will return nonsense.")

    # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
    site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

    # Add the contributions from these neighboring sites
    sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_plus_contribution += \
        (3*num_sites/2) * np.abs( spin_flip_spin_flip_correlations_e1e2e3_m0['Sip_e3_Sjp_e3_m0'][site_index,site_index_neighbor,m_min:m_max] ) ** 2

    sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_minus_contribution += \
        (3*num_sites/2) * np.abs( spin_flip_spin_flip_correlations_e1e2e3_m0['Sim_e3_Sjm_e3_m0'][site_index,site_index_neighbor,m_min:m_max] ) ** 2

    sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_e3_contribution += \
        (3*num_sites/2) * np.abs( spin_spin_correlations_e1e2e3_m0['Sie3Sje3_m0'][site_index,site_index_neighbor,m_min:m_max] ) ** 2

    # Make the 2D NumPy array energy_delta_function of shape ( num_eigenstates_to_sum_over x N_omega );
    # the (m,e)th element of this array is delta(omega_e - (E_m - E_0)), where
    # omega_e = omega_values[e], E_m = list_of_eigenvalues[m], and E_0 is the ground-state energy eigenvalue
    energy_delta_function = lorentzian(omega_values[np.newaxis,:], list_of_eigenvalues[m_min:m_max][:,np.newaxis] - ground_state_eigenvalue, eta)

    # Get the contributions to the two-magnon DOS from alpha = +_e3,-_e3,e3
    two_magnon_DOS_plus_contribution = 2*np.pi * sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_plus_contribution @ energy_delta_function
    two_magnon_DOS_minus_contribution = 2*np.pi * sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_minus_contribution @ energy_delta_function
    two_magnon_DOS_e3_contribution = 2*np.pi * sum_of_neighboring_sites_spin_flip_spin_flip_correlations_squared_e3_contribution @ energy_delta_function

    return two_magnon_DOS_plus_contribution, \
           two_magnon_DOS_minus_contribution, \
           two_magnon_DOS_e3_contribution

def dynamical_spin_structure_factor_total(momentum_vectors, omega_values, ground_state_eigenvalue, list_of_eigenvalues, spin_matrix_elements_0m, Nx, Ny, m_min = 0, m_max = None, eta = 0.01):
    """Function that computes the intra-sublattice and inter-sublattice contributions to the total dynamical spin structure factor, namely
        S_intra^tot(k,omega) = S_{AA}^tot(k,omega) + S_{BB}^tot(k,omega)
        S_inter^tot(k,omega) = S_{AB}^tot(k,omega) + S_{BA}^tot(k,omega)
    where
        S_{mu nu}^tot(k,omega) = 2 pi/N_uc * Sum_{m=m_min}^{m_max-1} Sum_{alpha=x,y,z}
                                                      [Sum_{i in sublattice mu} e^(-i k.r_i) <0|S_i^alpha|m>]
                                                    * [Sum_{j in sublattice nu} e^(+i k.r_j) <m|S_j^alpha|0>]
                                                    * delta(omega - (E_m - E_0))
    where N_uc = num_sites/2 = Nx*Ny is the number of unit cells in the system;
    mu,nu=A,B label the two sublattices (A and B); m indexes the energy eigenvalues E_m and eigenvectors |m>;
    i and j index the sites in sublattices mu and nu, respectively;
    k is the momentum of a given excitation;
    omega is the energy of a given excitation;
    r_i and r_j denote the position vectors of sites i and j.
    Inputs:
        momentum_vectors (2D NumPy array of shape N_k x 2, where N_k is the number of momentum vectors to probe;
                          momentum_vectors[:,0] contains the e1 coordinates of the vectors, and
                          momentum_vectors[:,1] contains the e2 coordinates of the vectors)
        omega_values (1D NumPy array of size N_omega)
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <bra|S_i^x|ket_m>,
                                 and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_min, m_max (indices of the minimum and maximum (minus one) eigenvectors |m> to sum over;
                      (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_min=0 and m_max=400)
                      if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
        eta (float specifying the half-width at half-maximum of the Lorentzian distributions used for the delta functions)
    Outputs:
        dynamical_spin_structure_factor_total_intra,
        dynamical_spin_structure_factor_total_inter (2D NumPy arrays of shape N_k x N_omega
                                                     (or, if N_k = 1, then they will be 1D NumPy arrays of size N_omega instead)
                                                     containing the intra- and inter- sublattice contributions to the
                                                     total dynamical spin structure factor, described above;
                                                     the (kappa,epsilon)th element of these arrays corresponds
                                                     to the value of the corresponding structure factor at momentum
                                                     k = momentum_vectors[kappa] and energy value omega = omega_values[epsilon])"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Number of unit cells
    N_uc = Nx * Ny

    # Number of momentum vectors
    num_momentum_vectors = len(momentum_vectors)

    # Number of energy eigenvalues/eigenvectors in list_of_eigenvalues/list_of_eigenvectors
    num_eigenstates_to_sum_over = len(list_of_eigenvalues[m_min:m_max])

    # Nearest-neighbor vectors (going from sublattice A sites to neighboring sublattice B sites) in e1,e2 coordinates
    dx = 1/2 * np.array([-np.sqrt(3), -1])
    dy = 1/2 * np.array([ np.sqrt(3), -1])
    dz = np.array([0, 1])
    # Lattice vectors in e1,e2 coordinates
    a1 = dy - dx
    a2 = dz - dx

    # Get the matrix elements <m|S_i^alpha|0> (for all i,alpha)
    spin_matrix_elements_m0 = spin_matrix_elements_between_many_bras_and_a_ket(spin_matrix_elements_0m)

    # Initialize the four arrays that will contain the sum of spin matrix elements (with their corresponding momentum phase factors)
    # for a given sublattice
    # (Note: Each of these will be a 3D NumPy array of shape (num_momentum_vectors x num_eigenstates_to_sum_over x 3);
    #        the (kappa,m,alpha)th entry of these arrays are
    #            array_A_sublattice_A[kappa,m,alpha] = Sum_{j in sublattice A} e^(-i k_kappa.r_j) <0|S_j^alpha|m>
    #            array_B_sublattice_A[kappa,m,alpha] = Sum_{j in sublattice A} e^(+i k_kappa.r_j) <m|S_j^alpha|0>
    #            array_A_sublattice_B[kappa,m,alpha] = Sum_{j in sublattice B} e^(-i k_kappa.r_j) <0|S_j^alpha|m>
    #            array_B_sublattice_B[kappa,m,alpha] = Sum_{j in sublattice B} e^(+i k_kappa.r_j) <m|S_j^alpha|0>
    #        where kappa indexes the momentum vectors in momentum_vectors, m indexes the energy eigenvalues E_m and eigenvectors |m>,
    #        and alpha indexes the components of the spin operators)
    array_A_sublattice_A = np.full((num_momentum_vectors,num_eigenstates_to_sum_over,3), 0 + 0j)
    array_B_sublattice_A = np.full((num_momentum_vectors,num_eigenstates_to_sum_over,3), 0 + 0j)
    array_A_sublattice_B = np.full((num_momentum_vectors,num_eigenstates_to_sum_over,3), 0 + 0j)
    array_B_sublattice_B = np.full((num_momentum_vectors,num_eigenstates_to_sum_over,3), 0 + 0j)

    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site indices for the current sublattice A and B sites in this unit cell
            sublattice_A_site_index = get_site_index(Nx, i, j, 0)
            sublattice_B_site_index = get_site_index(Nx, i, j, 1)

            # Position vectors for the current sublattice A and B sites in this unit cell
            sublattice_A_site_position_vector = i*a1 + j*a2
            sublattice_B_site_position_vector = i*a1 + j*a2 + dz

            # Add the contributions from this unit cell
            # (Note: In np.einsum, we are using the indices 'k', 'm', 'a' to denote the following axes:
            #            k: momentum axis           (size of this axis: num_momentum_vectors)
            #            m: eigenvector axis        (size of this axis: num_eigenstates_to_sum_over)
            #            a: spin component axis     (size of this axis: 3)
            # )
            array_A_sublattice_A += np.einsum('k,am->kma',
                                              np.exp(-1j * momentum_vectors @ sublattice_A_site_position_vector),
                                              np.array([spin_matrix_elements_0m['Six_0m'][sublattice_A_site_index,m_min:m_max],
                                                        spin_matrix_elements_0m['Siy_0m'][sublattice_A_site_index,m_min:m_max],
                                                        spin_matrix_elements_0m['Siz_0m'][sublattice_A_site_index,m_min:m_max]])
                                             )

            array_B_sublattice_A += np.einsum('k,am->kma',
                                              np.exp( 1j * momentum_vectors @ sublattice_A_site_position_vector),
                                              np.array([spin_matrix_elements_m0['Six_m0'][sublattice_A_site_index,m_min:m_max],
                                                        spin_matrix_elements_m0['Siy_m0'][sublattice_A_site_index,m_min:m_max],
                                                        spin_matrix_elements_m0['Siz_m0'][sublattice_A_site_index,m_min:m_max]])
                                             )

            array_A_sublattice_B += np.einsum('k,am->kma',
                                              np.exp(-1j * momentum_vectors @ sublattice_B_site_position_vector),
                                              np.array([spin_matrix_elements_0m['Six_0m'][sublattice_B_site_index,m_min:m_max],
                                                        spin_matrix_elements_0m['Siy_0m'][sublattice_B_site_index,m_min:m_max],
                                                        spin_matrix_elements_0m['Siz_0m'][sublattice_B_site_index,m_min:m_max]])
                                             )

            array_B_sublattice_B += np.einsum('k,am->kma',
                                              np.exp( 1j * momentum_vectors @ sublattice_B_site_position_vector),
                                              np.array([spin_matrix_elements_m0['Six_m0'][sublattice_B_site_index,m_min:m_max],
                                                        spin_matrix_elements_m0['Siy_m0'][sublattice_B_site_index,m_min:m_max],
                                                        spin_matrix_elements_m0['Siz_m0'][sublattice_B_site_index,m_min:m_max]])
                                             )

            # Slightly faster way of computing the four array_X_sublattice_Y arrays (i.e., without using np.einsum)
            # array_A_sublattice_A += ( \
            #                          (np.exp(-1j * momentum_vectors @ sublattice_A_site_position_vector))[:,np.newaxis] @ \
            #                          (np.array([spin_matrix_elements_0m['Six_0m'][sublattice_A_site_index,m_min:m_max],
            #                                     spin_matrix_elements_0m['Siy_0m'][sublattice_A_site_index,m_min:m_max],
            #                                     spin_matrix_elements_0m['Siz_0m'][sublattice_A_site_index,m_min:m_max]]))[:,np.newaxis,:] \
            #                         ).transpose(1,2,0)
            #
            # array_B_sublattice_A += ( \
            #                          (np.exp( 1j * momentum_vectors @ sublattice_A_site_position_vector))[:,np.newaxis] @ \
            #                          (np.array([spin_matrix_elements_m0['Six_m0'][sublattice_A_site_index,m_min:m_max],
            #                                     spin_matrix_elements_m0['Siy_m0'][sublattice_A_site_index,m_min:m_max],
            #                                     spin_matrix_elements_m0['Siz_m0'][sublattice_A_site_index,m_min:m_max]]))[:,np.newaxis,:] \
            #                         ).transpose(1,2,0)
            #
            # array_A_sublattice_B += ( \
            #                          (np.exp(-1j * momentum_vectors @ sublattice_B_site_position_vector))[:,np.newaxis] @ \
            #                          (np.array([spin_matrix_elements_0m['Six_0m'][sublattice_B_site_index,m_min:m_max],
            #                                     spin_matrix_elements_0m['Siy_0m'][sublattice_B_site_index,m_min:m_max],
            #                                     spin_matrix_elements_0m['Siz_0m'][sublattice_B_site_index,m_min:m_max]]))[:,np.newaxis,:] \
            #                         ).transpose(1,2,0)
            #
            # array_B_sublattice_B += ( \
            #                          (np.exp( 1j * momentum_vectors @ sublattice_B_site_position_vector))[:,np.newaxis] @ \
            #                          (np.array([spin_matrix_elements_m0['Six_m0'][sublattice_B_site_index,m_min:m_max],
            #                                     spin_matrix_elements_m0['Siy_m0'][sublattice_B_site_index,m_min:m_max],
            #                                     spin_matrix_elements_m0['Siz_m0'][sublattice_B_site_index,m_min:m_max]]))[:,np.newaxis,:] \
            #                         ).transpose(1,2,0)

    # Make the 2D NumPy array energy_delta_function of shape ( num_eigenstates_to_sum_over x N_omega );
    # the (m,e)th element of this array is delta(omega_e - (E_m - E_0)), where
    # omega_e = omega_values[e], E_m = list_of_eigenvalues[m], and E_0 is the ground-state energy eigenvalue
    energy_delta_function = lorentzian(omega_values[np.newaxis,:], list_of_eigenvalues[m_min:m_max][:,np.newaxis] - ground_state_eigenvalue, eta)

    # Compute the contributions to the total dynamical spin structure factor from each of the four combinations
    # of the two sublattices, namely AA, AB, BA, BB
    # (Note 1: These four terms are the S_{AA}^tot(k,omega), S_{AB}^tot(k,omega), S_{BA}^tot(k,omega), S_{BB}^tot(k,omega) described
    #          in this function's documentation)
    # (Note 2: For illustrative purposes, in the code below, we are computing total_dynamical_spin_structure_factor_AA as follows:
    #              total_dynamical_spin_structure_factor_AA = np.einsum('kma,kma,me->ke',
    #                                                                   array_A_sublattice_A,
    #                                                                   array_B_sublattice_A,
    #                                                                   energy_delta_function
    #                                                                  )
    #          where in np.einsum we are using the indices 'k', 'm', 'a', 'e' to denote the following axes:
    #              k: momentum axis             (size of this axis: num_momentum_vectors)
    #              m: eigenvector axis          (size of this axis: num_eigenstates_to_sum_over)
    #              a: spin component axis       (size of this axis: 3)
    #              e: energy value (omega) axis (size of this axis: N_omega)
    # )
    total_dynamical_spin_structure_factor_AA = 2*np.pi/N_uc * (array_A_sublattice_A * array_B_sublattice_A).sum(axis=2) @ energy_delta_function
    total_dynamical_spin_structure_factor_AB = 2*np.pi/N_uc * (array_A_sublattice_A * array_B_sublattice_B).sum(axis=2) @ energy_delta_function
    total_dynamical_spin_structure_factor_BA = 2*np.pi/N_uc * (array_A_sublattice_B * array_B_sublattice_A).sum(axis=2) @ energy_delta_function
    total_dynamical_spin_structure_factor_BB = 2*np.pi/N_uc * (array_A_sublattice_B * array_B_sublattice_B).sum(axis=2) @ energy_delta_function

    # Get the intra- and inter-sublattice contributions to the total dynamical spin structure factor
    dynamical_spin_structure_factor_total_intra = total_dynamical_spin_structure_factor_AA + total_dynamical_spin_structure_factor_BB
    dynamical_spin_structure_factor_total_inter = total_dynamical_spin_structure_factor_AB + total_dynamical_spin_structure_factor_BA

    # If only one momentum vector was input, make dynamical_spin_structure_factor_total_intXX 1D NumPy arrays instead
    if num_momentum_vectors == 1:
        dynamical_spin_structure_factor_total_intra = dynamical_spin_structure_factor_total_intra[0]
        dynamical_spin_structure_factor_total_inter = dynamical_spin_structure_factor_total_inter[0]

    return dynamical_spin_structure_factor_total_intra, dynamical_spin_structure_factor_total_inter

def dynamical_spin_structure_factor_total_with_momentum_integrated_out(omega_values, ground_state_eigenvalue, list_of_eigenvalues, spin_matrix_elements_0m, Nx, Ny, m_min = 0, m_max = None, eta = 0.01):
    """Function that computes the total dynamical spin structure factor with momentum integrated out, namely
        S^tot(omega) = 2 pi Sum_{m=m_min}^{m_max-1} Sum_{alpha=x,y,z} Sum_i |<0|S_i^alpha|m>|^2 delta(omega - (E_m - E_0))
    where m indexes the energy eigenvalues E_m and eigenvectors |m>, i indexes the lattice sites,
    and omega is the energy of a given excitation.
    Inputs:
        omega_values (1D NumPy array of size N_omega)
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <bra|S_i^x|ket_m>,
                                 and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_min, m_max (indices of the minimum and maximum (minus one) eigenvectors |m> to sum over;
                      (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_min=0 and m_max=400)
                      if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
        eta (float specifying the half-width at half-maximum of the Lorentzian distributions used for the delta functions)
    Output:
        dynamical_spin_structure_factor_total_with_momentum_integrated_out (1D NumPy array of size N_omega containing the
                                                                            total dynamical spin structure factor with momentum
                                                                            integrated out, described above; the epsilonth element
                                                                            of this array corresponds to the value of the structure factor at
                                                                            energy value omega = omega_values[epsilon])
    Note: S^tot(omega) is equivalent to the intra-sublattice contribution S_intra^tot(omega), since integrating out the momentum yields
          a sum over single-site spin correlations, which makes S_{AB}^tot(omega) = S_{BA}^tot(omega) = 0, and thus S_inter^tot(omega) = 0."""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Compute the sum of single-site spin matrix elements squared over all sites and over all spin components
    # (Note: This is a 1D NumPy array of size (m_max - m_min + 1);
    #        the mth entry of this array is
    #            sum_of_single_site_spin_matrix_elements_squared[m] = Sum_i Sum_{alpha=x,y,z} |<0|S_i^alpha|m>|^2
    #        where m indexes the energy eigenvalues E_m and eigenvectors |m>, and i indexes the lattice sites)
    sum_of_single_site_spin_matrix_elements_squared = np.sum( np.abs( spin_matrix_elements_0m['Six_0m'][:,m_min:m_max] ) ** 2 + \
                                                              np.abs( spin_matrix_elements_0m['Siy_0m'][:,m_min:m_max] ) ** 2 + \
                                                              np.abs( spin_matrix_elements_0m['Siz_0m'][:,m_min:m_max] ) ** 2,
                                                              axis=0
                                                            )

    # Make the 2D NumPy array energy_delta_function of shape ( (m_max - m_min + 1) x N_omega );
    # the (m,e)th element of this array is delta(omega_e - (E_m - E_0)), where
    # omega_e = omega_values[e], E_m = list_of_eigenvalues[m], and E_0 is the ground-state energy eigenvalue
    energy_delta_function = lorentzian(omega_values[np.newaxis,:], list_of_eigenvalues[m_min:m_max][:,np.newaxis] - ground_state_eigenvalue, eta)

    # Compute the total dynamical spin structure factor with momentum integrated out
    dynamical_spin_structure_factor_total_with_momentum_integrated_out = 2*np.pi * sum_of_single_site_spin_matrix_elements_squared @ energy_delta_function

    return dynamical_spin_structure_factor_total_with_momentum_integrated_out

def dynamical_spin_structure_factor_total_with_momentum_and_energy_integrated_out(ground_state_eigenvalue, list_of_eigenvalues, spin_matrix_elements_0m, Nx, Ny, m_max = None):
    """Function that computes the total dynamical spin structure factor with momentum and energy integrated out, namely
        Integral_{omega=0}^{E_{m_max} - E_0} S^tot(omega) d omega = 2 pi Sum_{m=0}^{m_max-1} Sum_{alpha=x,y,z} Sum_i |<0|S_i^alpha|m>|^2
    where m indexes the energy eigenvalues E_m and eigenvectors |m>, i indexes the lattice sites,
    and omega is the energy of a given excitation.
    Inputs:
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_matrix_elements_0m (dictionary containing the arrays following 3 NumPy arrays:
                                     Six_0m, Siy_0m, Siz_0m;
                                 these arrays are 2D NumPy arrays of shape num_sites x N, where num_sites = 2 Nx Ny;
                                 the (i,m)th element of Six_0m corresponds to the matrix element <bra|S_i^x|ket_m>,
                                 and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_max (index of the maximum (minus one) eigenvectors |m> to sum over;
               (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_max=400)
               if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
    Output:
        dynamical_spin_structure_factor_total_with_momentum_and_energy_integrated_out (float containing the total dynamical
                                                                                       spin structure factor with momentum and energy
                                                                                       integrated out, described above)"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Index of the minimum eigenvector
    m_min = 0

    # Compute the total dynamical spin structure factor with momentum and energy integrated out
    dynamical_spin_structure_factor_total_with_momentum_and_energy_integrated_out = \
                                                      np.sum( np.abs( spin_matrix_elements_0m['Six_0m'][:,m_min:m_max] ) ** 2 + \
                                                              np.abs( spin_matrix_elements_0m['Siy_0m'][:,m_min:m_max] ) ** 2 + \
                                                              np.abs( spin_matrix_elements_0m['Siz_0m'][:,m_min:m_max] ) ** 2
                                                            )

    # Multiply dynamical_spin_structure_factor_total_with_momentum_and_energy_integrated_out by 2 pi
    dynamical_spin_structure_factor_total_with_momentum_and_energy_integrated_out *= 2*np.pi

    return dynamical_spin_structure_factor_total_with_momentum_and_energy_integrated_out

def dynamical_two_spin_structure_factor_total(momentum_vectors, omega_values, ground_state_eigenvalue, list_of_eigenvalues, spin_spin_correlations_0m, Nx, Ny, m_min = 0, m_max = None, eta = 0.01, boundary_conditions = 'periodic'):
    """Function that computes the total (summed over x,y,z --- note that this quantity is not rotation-symmetric)
    dynamical two-spin structure factor namely
        P^tot(k,omega) = 2 pi/N_uc * Sum_{m=m_min}^{m_max-1} Sum_{gamma=x,y,z} Sum_{alpha=x,y,z}
                                    [Sum_{i in sublattice B} e^(-i k.r_i) <0|S_i^alpha S_{i-delta_gamma}^alpha|m>]
                                  * [Sum_{j in sublattice B} e^(+i k.r_j) <m|S_j^alpha S_{j-delta_gamma}^alpha|0>]
                                  * delta(omega - (E_m - E_0))
    where N_uc = num_sites/2 = Nx*Ny is the number of unit cells in the system;
    m indexes the energy eigenvalues E_m and eigenvectors |m>;
    i and j each index the sites in sublattice B;
    k is the momentum of a given excitation;
    omega is the energy of a given excitation;
    r_i and r_j denote the position vectors of sites i and j; and
    S_{i-delta_gamma}^alpha denotes the alpha component of the spin operator acting on
                            the sublattice A site interacting with site i through a gamma-bond.
    Inputs:
        momentum_vectors (2D NumPy array of shape N_k x 2, where N_k is the number of momentum vectors to probe;
                          momentum_vectors[:,0] contains the e1 coordinates of the vectors, and
                          momentum_vectors[:,1] contains the e2 coordinates of the vectors)
        omega_values (1D NumPy array of size N_omega)
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_min, m_max (indices of the minimum and maximum (minus one) eigenvectors |m> to sum over;
                      (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_min=0 and m_max=400)
                      if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
        eta (float specifying the half-width at half-maximum of the Lorentzian distributions used for the delta functions)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Outputs:
        dynamical_two_spin_structure_factor_total (2D NumPy array of shape N_k x N_omega
                                                   (or, if N_k = 1, then it will be a 1D NumPy array of size N_omega instead)
                                                   containing the total (summed over x,y,z) dynamical two-spin structure factor,
                                                   described above; the (kappa,epsilon)th element of this array corresponds
                                                   to the value of the corresponding structure factor at momentum
                                                   k = momentum_vectors[kappa] and energy value omega = omega_values[epsilon])"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Number of unit cells
    N_uc = Nx * Ny

    # Number of momentum vectors
    num_momentum_vectors = len(momentum_vectors)

    # Number of energy eigenvalues/eigenvectors in list_of_eigenvalues/list_of_eigenvectors
    num_eigenstates_to_sum_over = len(list_of_eigenvalues[m_min:m_max])

    # Nearest-neighbor vectors (going from sublattice A sites to neighboring sublattice B sites) in e1,e2 coordinates
    dx = 1/2 * np.array([-np.sqrt(3), -1])
    dy = 1/2 * np.array([ np.sqrt(3), -1])
    dz = np.array([0, 1])
    # Lattice vectors in e1,e2 coordinates
    a1 = dy - dx
    a2 = dz - dx

    # Get the spin-spin correlations <m|S_i^alpha S_j^beta|0> (alpha,beta=x,y,z)
    spin_spin_correlations_m0 = spin_spin_correlations_between_many_bras_and_a_ket(spin_spin_correlations_0m)

    # Initialize the two arrays that will contain the sum of spin-spin correlations (with their corresponding momentum phase factors)
    # (Note: Each of these will be a 4D NumPy array of shape (num_momentum_vectors x num_eigenstates_to_sum_over x 3 x 3);
    #        the (kappa,m,alpha,gamma)th entry of these arrays are
    #            array_A[kappa,m,alpha,gamma] = Sum_{j in sublattice B} e^(-i k.r_j) <0|S_j^alpha S_{j-delta_gamma}^alpha|m>
    #            array_B[kappa,m,alpha,gamma] = Sum_{j in sublattice B} e^(+i k.r_j) <m|S_j^alpha S_{j-delta_gamma}^alpha|0>
    #        where kappa indexes the momentum vectors in momentum_vectors, m indexes the energy eigenvalues E_m and eigenvectors |m>,
    #        alpha indexes the components of the spin operators, and gamma indexes the bond types)
    array_A = np.full((num_momentum_vectors,num_eigenstates_to_sum_over,3,3), 0 + 0j)
    array_B = np.full((num_momentum_vectors,num_eigenstates_to_sum_over,3,3), 0 + 0j)

    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site index of the sublattice B site in this unit cell
            site_index = get_site_index(Nx, i, j, 1)

            # Position vector for the current sublattice B site in this unit cell
            position_vector = i*a1 + j*a2 + dz

            # Loop over the 3 bonds for the B site of this unit cell
            # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
            for bond_type in range(3):

                # Initialize Boolean that specifies whether or not this bond is a boundary bond
                # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
                boundary_bond = False

                # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
                if bond_type == 0:
                    i_neighbor = i
                    j_neighbor = (j + 1) % Ny
                    if j_neighbor != (j + 1):
                        boundary_bond = True
                elif bond_type == 1:
                    i_neighbor = (i - 1) % Nx
                    j_neighbor = (j + 1) % Ny
                    if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                        boundary_bond = True
                elif bond_type == 2:
                    i_neighbor = i
                    j_neighbor = j

                # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
                # and this is a boundary bond
                if (boundary_conditions == 'open') and (boundary_bond == True):
                    continue

                # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
                site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

                # cont here : test using this position_vector (using the sublattice A site) instead of the one above (for the sublattice B site);
                #             they should be equivalent
                # # Position vector for the current sublattice A site in this unit cell
                # position_vector = i*a1 + j*a2

                # Add the contributions from this bond
                # (Note: In np.einsum, we are using the indices 'k', 'm', 'a' to denote the following axes:
                #            k: momentum axis           (size of this axis: num_momentum_vectors)
                #            m: eigenvector axis        (size of this axis: num_eigenstates_to_sum_over)
                #            a: spin component axis     (size of this axis: 3)
                # )
                array_A[:,:,:,bond_type] += np.einsum('k,am->kma',
                                                      np.exp(-1j * momentum_vectors @ position_vector),
                                                      np.array([spin_spin_correlations_0m['SixSjx_0m'][site_index,site_index_neighbor,m_min:m_max],
                                                                spin_spin_correlations_0m['SiySjy_0m'][site_index,site_index_neighbor,m_min:m_max],
                                                                spin_spin_correlations_0m['SizSjz_0m'][site_index,site_index_neighbor,m_min:m_max]])
                                                     )

                array_B[:,:,:,bond_type] += np.einsum('k,am->kma',
                                                      np.exp( 1j * momentum_vectors @ position_vector),
                                                      np.array([spin_spin_correlations_m0['SixSjx_m0'][site_index,site_index_neighbor,m_min:m_max],
                                                                spin_spin_correlations_m0['SiySjy_m0'][site_index,site_index_neighbor,m_min:m_max],
                                                                spin_spin_correlations_m0['SizSjz_m0'][site_index,site_index_neighbor,m_min:m_max]])
                                                     )

    # Make the 2D NumPy array energy_delta_function of shape ( num_eigenstates_to_sum_over x N_omega );
    # the (m,e)th element of this array is delta(omega_e - (E_m - E_0)), where
    # omega_e = omega_values[e], E_m = list_of_eigenvalues[m], and E_0 is the ground-state energy eigenvalue
    energy_delta_function = lorentzian(omega_values[np.newaxis,:], list_of_eigenvalues[m_min:m_max][:,np.newaxis] - ground_state_eigenvalue, eta)

    # Compute the total dynamical two-spin structure factor
    # (Note: For illustrative purposes, in the code below, we are computing dynamical_two_spin_structure_factor_total as follows:
    #            dynamical_two_spin_structure_factor_total = np.einsum('kmag,kmag,me->ke',
    #                                                                  array_A,
    #                                                                  array_B,
    #                                                                  energy_delta_function
    #                                                                 )
    #        where in np.einsum we are using the indices 'k', 'm', 'a', 'g', 'e' to denote the following axes:
    #            k: momentum axis             (size of this axis: num_momentum_vectors)
    #            m: eigenvector axis          (size of this axis: num_eigenstates_to_sum_over)
    #            a: spin component axis       (size of this axis: 3)
    #            g: bond type axis            (size of this axis: 3)
    #            e: energy value (omega) axis (size of this axis: N_omega)
    # )
    dynamical_two_spin_structure_factor_total = 2*np.pi/N_uc * (array_A * array_B).sum(axis=(2,3)) @ energy_delta_function

    # If only one momentum vector was input, make dynamical_two_spin_structure_factor_total a 1D NumPy array instead
    if num_momentum_vectors == 1:
        dynamical_two_spin_structure_factor_total = dynamical_two_spin_structure_factor_total[0]

    return dynamical_two_spin_structure_factor_total

def dynamical_two_spin_structure_factor_total_e1e2e3(momentum_vectors, omega_values, ground_state_eigenvalue, list_of_eigenvalues, spin_spin_correlations_0m, Nx, Ny, m_min = 0, m_max = None, eta = 0.01, boundary_conditions = 'periodic'):
    """Function that computes the total (summed over e1,e2,e3 --- note that this quantity is not rotation-symmetric)
    dynamical two-spin structure factor namely
        P^tot(k,omega) = 2 pi/N_uc * Sum_{m=m_min}^{m_max-1} Sum_{gamma=x,y,z} Sum_{alpha=e1,e2,e3}
                                    [Sum_{i in sublattice B} e^(-i k.r_i) <0|S_i^alpha S_{i-delta_gamma}^alpha|m>]
                                  * [Sum_{j in sublattice B} e^(+i k.r_j) <m|S_j^alpha S_{j-delta_gamma}^alpha|0>]
                                  * delta(omega - (E_m - E_0))
    where N_uc = num_sites/2 = Nx*Ny is the number of unit cells in the system;
    m indexes the energy eigenvalues E_m and eigenvectors |m>;
    i and j each index the sites in sublattice B;
    k is the momentum of a given excitation;
    omega is the energy of a given excitation;
    r_i and r_j denote the position vectors of sites i and j; and
    S_{i-delta_gamma}^alpha denotes the alpha component of the spin operator acting on
                            the sublattice A site interacting with site i through a gamma-bond.
    Inputs:
        momentum_vectors (2D NumPy array of shape N_k x 2, where N_k is the number of momentum vectors to probe;
                          momentum_vectors[:,0] contains the e1 coordinates of the vectors, and
                          momentum_vectors[:,1] contains the e2 coordinates of the vectors)
        omega_values (1D NumPy array of size N_omega)
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_min, m_max (indices of the minimum and maximum (minus one) eigenvectors |m> to sum over;
                      (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_min=0 and m_max=400)
                      if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
        eta (float specifying the half-width at half-maximum of the Lorentzian distributions used for the delta functions)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Outputs:
        dynamical_two_spin_structure_factor_total_e1e2e3 (2D NumPy array of shape N_k x N_omega
                                                          (or, if N_k = 1, then it will be a 1D NumPy array of size N_omega instead)
                                                          containing the total (summed over e1,e2,e3) dynamical two-spin structure factor,
                                                          described above; the (kappa,epsilon)th element of this array corresponds
                                                          to the value of the corresponding structure factor at momentum
                                                          k = momentum_vectors[kappa] and energy value omega = omega_values[epsilon])"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Number of unit cells
    N_uc = Nx * Ny

    # Number of momentum vectors
    num_momentum_vectors = len(momentum_vectors)

    # Number of energy eigenvalues/eigenvectors in list_of_eigenvalues/list_of_eigenvectors
    num_eigenstates_to_sum_over = len(list_of_eigenvalues[m_min:m_max])

    # Nearest-neighbor vectors (going from sublattice A sites to neighboring sublattice B sites) in e1,e2 coordinates
    dx = 1/2 * np.array([-np.sqrt(3), -1])
    dy = 1/2 * np.array([ np.sqrt(3), -1])
    dz = np.array([0, 1])
    # Lattice vectors in e1,e2 coordinates
    a1 = dy - dx
    a2 = dz - dx

    # Compute the spin-spin correlations in e1,e2,e3 coordinates: <0|S_i^alpha S_j^beta|m> (alpha,beta=e1,e2,e3)
    spin_spin_correlations_e1e2e3_0m = spin_spin_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_0m)

    # Get the spin-spin correlations <m|S_i^alpha S_j^beta|0> (alpha,beta=e1,e2,e3)
    spin_spin_correlations_e1e2e3_m0 = spin_spin_correlations_between_many_bras_and_a_ket_e1e2e3(spin_spin_correlations_e1e2e3_0m)

    # Initialize the two arrays that will contain the sum of spin-spin correlations (with their corresponding momentum phase factors)
    # (Note: Each of these will be a 4D NumPy array of shape (num_momentum_vectors x num_eigenstates_to_sum_over x 3 x 3);
    #        the (kappa,m,alpha,gamma)th entry of these arrays are
    #            array_A[kappa,m,alpha,gamma] = Sum_{j in sublattice B} e^(-i k.r_j) <0|S_j^alpha S_{j-delta_gamma}^alpha|m>
    #            array_B[kappa,m,alpha,gamma] = Sum_{j in sublattice B} e^(+i k.r_j) <m|S_j^alpha S_{j-delta_gamma}^alpha|0>
    #        where kappa indexes the momentum vectors in momentum_vectors, m indexes the energy eigenvalues E_m and eigenvectors |m>,
    #        alpha indexes the components of the spin operators, and gamma indexes the bond types)
    array_A = np.full((num_momentum_vectors,num_eigenstates_to_sum_over,3,3), 0 + 0j)
    array_B = np.full((num_momentum_vectors,num_eigenstates_to_sum_over,3,3), 0 + 0j)

    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site index of the sublattice B site in this unit cell
            site_index = get_site_index(Nx, i, j, 1)

            # Position vector for the current sublattice B site in this unit cell
            position_vector = i*a1 + j*a2 + dz

            # Loop over the 3 bonds for the B site of this unit cell
            # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
            for bond_type in range(3):

                # Initialize Boolean that specifies whether or not this bond is a boundary bond
                # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
                boundary_bond = False

                # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
                if bond_type == 0:
                    i_neighbor = i
                    j_neighbor = (j + 1) % Ny
                    if j_neighbor != (j + 1):
                        boundary_bond = True
                elif bond_type == 1:
                    i_neighbor = (i - 1) % Nx
                    j_neighbor = (j + 1) % Ny
                    if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                        boundary_bond = True
                elif bond_type == 2:
                    i_neighbor = i
                    j_neighbor = j

                # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
                # and this is a boundary bond
                if (boundary_conditions == 'open') and (boundary_bond == True):
                    continue

                # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
                site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

                # cont here : test using this position_vector (using the sublattice A site) instead of the one above (for the sublattice B site);
                #             they should be equivalent
                # # Position vector for the current sublattice A site in this unit cell
                # position_vector = i*a1 + j*a2

                # Add the contributions from this bond
                # (Note: In np.einsum, we are using the indices 'k', 'm', 'a' to denote the following axes:
                #            k: momentum axis           (size of this axis: num_momentum_vectors)
                #            m: eigenvector axis        (size of this axis: num_eigenstates_to_sum_over)
                #            a: spin component axis     (size of this axis: 3)
                # )
                array_A[:,:,:,bond_type] += np.einsum('k,am->kma',
                                                      np.exp(-1j * momentum_vectors @ position_vector),
                                                      np.array([spin_spin_correlations_e1e2e3_0m['Sie1Sje1_0m'][site_index,site_index_neighbor,m_min:m_max],
                                                                spin_spin_correlations_e1e2e3_0m['Sie2Sje2_0m'][site_index,site_index_neighbor,m_min:m_max],
                                                                spin_spin_correlations_e1e2e3_0m['Sie3Sje3_0m'][site_index,site_index_neighbor,m_min:m_max]])
                                                     )

                array_B[:,:,:,bond_type] += np.einsum('k,am->kma',
                                                      np.exp( 1j * momentum_vectors @ position_vector),
                                                      np.array([spin_spin_correlations_e1e2e3_m0['Sie1Sje1_m0'][site_index,site_index_neighbor,m_min:m_max],
                                                                spin_spin_correlations_e1e2e3_m0['Sie2Sje2_m0'][site_index,site_index_neighbor,m_min:m_max],
                                                                spin_spin_correlations_e1e2e3_m0['Sie3Sje3_m0'][site_index,site_index_neighbor,m_min:m_max]])
                                                     )

    # Make the 2D NumPy array energy_delta_function of shape ( num_eigenstates_to_sum_over x N_omega );
    # the (m,e)th element of this array is delta(omega_e - (E_m - E_0)), where
    # omega_e = omega_values[e], E_m = list_of_eigenvalues[m], and E_0 is the ground-state energy eigenvalue
    energy_delta_function = lorentzian(omega_values[np.newaxis,:], list_of_eigenvalues[m_min:m_max][:,np.newaxis] - ground_state_eigenvalue, eta)

    # Compute the total dynamical two-spin structure factor
    # (Note: For illustrative purposes, in the code below, we are computing dynamical_two_spin_structure_factor_total_e1e2e3 as follows:
    #            dynamical_two_spin_structure_factor_total_e1e2e3 = np.einsum('kmag,kmag,me->ke',
    #                                                                         array_A,
    #                                                                         array_B,
    #                                                                         energy_delta_function
    #                                                                        )
    #        where in np.einsum we are using the indices 'k', 'm', 'a', 'g', 'e' to denote the following axes:
    #            k: momentum axis             (size of this axis: num_momentum_vectors)
    #            m: eigenvector axis          (size of this axis: num_eigenstates_to_sum_over)
    #            a: spin component axis       (size of this axis: 3)
    #            g: bond type axis            (size of this axis: 3)
    #            e: energy value (omega) axis (size of this axis: N_omega)
    # )
    dynamical_two_spin_structure_factor_total_e1e2e3 = 2*np.pi/N_uc * (array_A * array_B).sum(axis=(2,3)) @ energy_delta_function

    # If only one momentum vector was input, make dynamical_two_spin_structure_factor_total_e1e2e3 a 1D NumPy array instead
    if num_momentum_vectors == 1:
        dynamical_two_spin_structure_factor_total_e1e2e3 = dynamical_two_spin_structure_factor_total_e1e2e3[0]

    return dynamical_two_spin_structure_factor_total_e1e2e3

def dynamical_two_spin_structure_factor_total_with_momentum_integrated_out(omega_values, ground_state_eigenvalue, list_of_eigenvalues, spin_spin_correlations_0m, Nx, Ny, m_min = 1, m_max = None, eta = 0.01, boundary_conditions = 'periodic'):
    """Function that computes the total (summed over x,y,z --- note that this quantity is not rotation-symmetric)
    dynamical two-spin structure factor with momentum integrated out, namely
        P^tot(omega) = 2 pi Sum_{m=m_min}^{m_max-1} Sum_{alpha=x,y,z} Sum_<ij> |<0|S_i^alpha S_j^alpha|m>|^2 delta(omega - (E_m - E_0))
    where m indexes the energy eigenvalues E_m and eigenvectors |m>, <ij> indexes a sum over neighboring sites,
    and omega is the energy of a given excitation.
    Inputs:
        omega_values (1D NumPy array of size N_omega)
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_min, m_max (indices of the minimum and maximum (minus one) eigenvectors |m> to sum over;
                      (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_min=0 and m_max=400)
                      if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
        eta (float specifying the half-width at half-maximum of the Lorentzian distributions used for the delta functions)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Outputs:
        dynamical_two_spin_structure_factor_total_with_momentum_integrated_out (1D NumPy array of size N_omega containing the
                                                                                total (summed over x,y,z) dynamical two-spin structure
                                                                                factor with momentum integrated out, described above;
                                                                                the epsilonth element of this array corresponds to
                                                                                the value of the structure factor
                                                                                at energy value omega = omega_values[epsilon])"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Number of energy eigenvalues/eigenvectors in list_of_eigenvalues/list_of_eigenvectors
    num_eigenstates_to_sum_over = len(list_of_eigenvalues[m_min:m_max])

    # Initialize the sum of neighboring sites' spin-spin correlations squared
    # (Note: This is a 1D NumPy array of size num_eigenstates_to_sum_over = m_max - m_min;
    #        the mth entry of this array is
    #            sum_of_neighboring_sites_spin_spin_correlations_squared[m] = Sum_<ij> Sum_{alpha=x,y,z} |<0|S_i^alpha S_j^alpha|m>|^2
    #        where m indexes the energy eigenvalues E_m and eigenvectors |m>, and <ij> index neighboring lattice sites)
    sum_of_neighboring_sites_spin_spin_correlations_squared = np.full(num_eigenstates_to_sum_over, 0.)

    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site index of the sublattice B site in this unit cell
            site_index = get_site_index(Nx, i, j, 1)

            # Loop over the 3 bonds for the B site of this unit cell
            # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
            for bond_type in range(3):

                # Initialize Boolean that specifies whether or not this bond is a boundary bond
                # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
                boundary_bond = False

                # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
                if bond_type == 0:
                    i_neighbor = i
                    j_neighbor = (j + 1) % Ny
                    if j_neighbor != (j + 1):
                        boundary_bond = True
                elif bond_type == 1:
                    i_neighbor = (i - 1) % Nx
                    j_neighbor = (j + 1) % Ny
                    if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                        boundary_bond = True
                elif bond_type == 2:
                    i_neighbor = i
                    j_neighbor = j

                # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
                # and this is a boundary bond
                if (boundary_conditions == 'open') and (boundary_bond == True):
                    continue

                # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
                site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

                # Add the contributions from these neighboring sites
                sum_of_neighboring_sites_spin_spin_correlations_squared += \
                    np.abs( spin_spin_correlations_0m['SixSjx_0m'][site_index,site_index_neighbor,m_min:m_max] ) ** 2 + \
                    np.abs( spin_spin_correlations_0m['SiySjy_0m'][site_index,site_index_neighbor,m_min:m_max] ) ** 2 + \
                    np.abs( spin_spin_correlations_0m['SizSjz_0m'][site_index,site_index_neighbor,m_min:m_max] ) ** 2

    # Make the 2D NumPy array energy_delta_function of shape ( num_eigenstates_to_sum_over x N_omega );
    # the (m,e)th element of this array is delta(omega_e - (E_m - E_0)), where
    # omega_e = omega_values[e], E_m = list_of_eigenvalues[m], and E_0 is the ground-state energy eigenvalue
    energy_delta_function = lorentzian(omega_values[np.newaxis,:], list_of_eigenvalues[m_min:m_max][:,np.newaxis] - ground_state_eigenvalue, eta)

    # Compute the total (summed over x,y,z) dynamical two-spin structure factor with momentum integrated out
    dynamical_two_spin_structure_factor_total_with_momentum_integrated_out = 2*np.pi * sum_of_neighboring_sites_spin_spin_correlations_squared @ energy_delta_function

    return dynamical_two_spin_structure_factor_total_with_momentum_integrated_out

def dynamical_two_spin_structure_factor_total_with_momentum_and_energy_integrated_out(ground_state_eigenvalue, list_of_eigenvalues, spin_spin_correlations_0m, Nx, Ny, m_max = None, boundary_conditions = 'periodic'):
    """Function that computes the total (summed over x,y,z --- note that this quantity is not rotation-symmetric)
    dynamical two-spin structure factor with momentum integrated out, namely
        Integral_{omega=0}^{E_{m_max} - E_0} P^tot(omega) d omega = 2 pi Sum_{m=0}^{m_max-1} Sum_{alpha=x,y,z} Sum_<ij> |<0|S_i^alpha S_j^alpha|m>|^2
    where m indexes the energy eigenvalues E_m and eigenvectors |m>, <ij> indexes a sum over neighboring sites,
    and omega is the energy of a given excitation.
    Inputs:
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_max (index of the maximum (minus one) eigenvectors |m> to sum over;
               (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_max=400)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Outputs:
        dynamical_two_spin_structure_factor_total_with_momentum_and_energy_integrated_out (float containing the total (summed over x,y,z)
                                                                                           dynamical two-spin structure factor with
                                                                                           momentum and energy integrated out, described above)"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Index of the minimum eigenvector
    m_min = 0

    # Number of energy eigenvalues/eigenvectors in list_of_eigenvalues/list_of_eigenvectors
    num_eigenstates_to_sum_over = len(list_of_eigenvalues[m_min:m_max])

    # Initialize the total (summed over x,y,z) dynamical two-spin structure factor with momentum integrated out
    dynamical_two_spin_structure_factor_total_with_momentum_and_energy_integrated_out = 0.

    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site index of the sublattice B site in this unit cell
            site_index = get_site_index(Nx, i, j, 1)

            # Loop over the 3 bonds for the B site of this unit cell
            # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
            for bond_type in range(3):

                # Initialize Boolean that specifies whether or not this bond is a boundary bond
                # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
                boundary_bond = False

                # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
                if bond_type == 0:
                    i_neighbor = i
                    j_neighbor = (j + 1) % Ny
                    if j_neighbor != (j + 1):
                        boundary_bond = True
                elif bond_type == 1:
                    i_neighbor = (i - 1) % Nx
                    j_neighbor = (j + 1) % Ny
                    if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                        boundary_bond = True
                elif bond_type == 2:
                    i_neighbor = i
                    j_neighbor = j

                # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
                # and this is a boundary bond
                if (boundary_conditions == 'open') and (boundary_bond == True):
                    continue

                # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
                site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

                # Add the contributions from these neighboring sites
                dynamical_two_spin_structure_factor_total_with_momentum_and_energy_integrated_out += \
                    np.sum( np.abs( spin_spin_correlations_0m['SixSjx_0m'][site_index,site_index_neighbor,m_min:m_max] ) ** 2 + \
                            np.abs( spin_spin_correlations_0m['SiySjy_0m'][site_index,site_index_neighbor,m_min:m_max] ) ** 2 + \
                            np.abs( spin_spin_correlations_0m['SizSjz_0m'][site_index,site_index_neighbor,m_min:m_max] ) ** 2
                          )

    # Multiply dynamical_two_spin_structure_factor_total_with_momentum_and_energy_integrated_out by 2 pi
    dynamical_two_spin_structure_factor_total_with_momentum_and_energy_integrated_out *= 2*np.pi

    return dynamical_two_spin_structure_factor_total_with_momentum_and_energy_integrated_out

def dynamical_two_spin_structure_factor_total_with_momentum_integrated_out_e1e2e3(omega_values, ground_state_eigenvalue, list_of_eigenvalues, spin_spin_correlations_0m, Nx, Ny, m_min = 1, m_max = None, eta = 0.01, boundary_conditions = 'periodic'):
    """Function that computes the total (summed over e1,e2,e3 --- note that this quantity is not rotation-symmetric)
    dynamical two-spin structure factor with momentum integrated out, namely
        P^tot(omega) = 2 pi Sum_{m=m_min}^{m_max-1} Sum_{alpha=e1,e2,e3} Sum_<ij> |<0|S_i^alpha S_j^alpha|m>|^2 delta(omega - (E_m - E_0))
    where m indexes the energy eigenvalues E_m and eigenvectors |m>,
    <ij> indexes a sum over neighboring sites,
    and omega is the energy of a given excitation.
    Inputs:
        omega_values (1D NumPy array of size N_omega)
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_min, m_max (indices of the minimum and maximum (minus one) eigenvectors |m> to sum over;
                      (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_min=0 and m_max=400)
                      if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
        eta (float specifying the half-width at half-maximum of the Lorentzian distributions used for the delta functions)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Outputs:
        dynamical_two_spin_structure_factor_total_with_momentum_integrated_out_e1e2e3 (1D NumPy array of size N_omega containing the
                                                                                       total (summed over e1,e2,e3) dynamical two-spin
                                                                                       structure factor with momentum integrated out,
                                                                                       described above; the epsilonth element of this array
                                                                                       corresponds to the value of the structure factor
                                                                                       at energy value omega = omega_values[epsilon])"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Number of energy eigenvalues/eigenvectors in list_of_eigenvalues/list_of_eigenvectors
    num_eigenstates_to_sum_over = len(list_of_eigenvalues[m_min:m_max])

    # Compute the spin-spin correlations in e1,e2,e3 coordinates: <0|S_i^alpha S_j^beta|m> (alpha,beta=e1,e2,e3)
    spin_spin_correlations_e1e2e3_0m = spin_spin_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_0m)

    # Initialize the sum of neighboring sites' spin-spin correlations squared
    # (Note: This is a 1D NumPy array of size num_eigenstates_to_sum_over = m_max - m_min;
    #        the mth entry of this array is
    #            sum_of_neighboring_sites_spin_spin_correlations_squared[m] = Sum_<ij> Sum_{alpha=e1,e2,e3} |<0|S_i^alpha S_j^alpha|m>|^2
    #        where m indexes the energy eigenvalues E_m and eigenvectors |m>, and <ij> index neighboring lattice sites)
    sum_of_neighboring_sites_spin_spin_correlations_squared = np.full(num_eigenstates_to_sum_over, 0.)

    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site index of the sublattice B site in this unit cell
            site_index = get_site_index(Nx, i, j, 1)

            # Loop over the 3 bonds for the B site of this unit cell
            # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
            for bond_type in range(3):

                # Initialize Boolean that specifies whether or not this bond is a boundary bond
                # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
                boundary_bond = False

                # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
                if bond_type == 0:
                    i_neighbor = i
                    j_neighbor = (j + 1) % Ny
                    if j_neighbor != (j + 1):
                        boundary_bond = True
                elif bond_type == 1:
                    i_neighbor = (i - 1) % Nx
                    j_neighbor = (j + 1) % Ny
                    if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                        boundary_bond = True
                elif bond_type == 2:
                    i_neighbor = i
                    j_neighbor = j

                # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
                # and this is a boundary bond
                if (boundary_conditions == 'open') and (boundary_bond == True):
                    continue

                # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
                site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

                # Add the contributions from these neighboring sites
                sum_of_neighboring_sites_spin_spin_correlations_squared += \
                    np.abs( spin_spin_correlations_e1e2e3_0m['Sie1Sje1_0m'][site_index,site_index_neighbor,m_min:m_max] ) ** 2 + \
                    np.abs( spin_spin_correlations_e1e2e3_0m['Sie2Sje2_0m'][site_index,site_index_neighbor,m_min:m_max] ) ** 2 + \
                    np.abs( spin_spin_correlations_e1e2e3_0m['Sie3Sje3_0m'][site_index,site_index_neighbor,m_min:m_max] ) ** 2

    # Make the 2D NumPy array energy_delta_function of shape ( num_eigenstates_to_sum_over x N_omega );
    # the (m,e)th element of this array is delta(omega_e - (E_m - E_0)), where
    # omega_e = omega_values[e], E_m = list_of_eigenvalues[m], and E_0 is the ground-state energy eigenvalue
    energy_delta_function = lorentzian(omega_values[np.newaxis,:], list_of_eigenvalues[m_min:m_max][:,np.newaxis] - ground_state_eigenvalue, eta)

    # Compute the total (summed over e1,e2,e3) dynamical two-spin structure factor with momentum integrated out
    dynamical_two_spin_structure_factor_total_with_momentum_integrated_out_e1e2e3 = 2*np.pi * sum_of_neighboring_sites_spin_spin_correlations_squared @ energy_delta_function

    return dynamical_two_spin_structure_factor_total_with_momentum_integrated_out_e1e2e3

def dynamical_two_spin_structure_factor_total_with_momentum_and_energy_integrated_out_e1e2e3(ground_state_eigenvalue, list_of_eigenvalues, spin_spin_correlations_0m, Nx, Ny, m_max = None, boundary_conditions = 'periodic'):
    """Function that computes the total (summed over e1,e2,e3 --- note that this quantity is not rotation-symmetric)
    dynamical two-spin structure factor with momentum and energy integrated out, namely
        Integral_{omega=0}^{E_{m_max} - E_0} P^tot(omega) d omega = 2 pi Sum_{m=0}^{m_max-1} Sum_{alpha=e1,e2,e3} Sum_<ij> |<0|S_i^alpha S_j^alpha|m>|^2
    where m indexes the energy eigenvalues E_m and eigenvectors |m>, <ij> indexes a sum over neighboring sites,
    and omega is the energy of a given excitation.
    Inputs:
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_max (index of the maximum (minus one) eigenvectors |m> to sum over;
               (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_max=400)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Outputs:
        dynamical_two_spin_structure_factor_total_with_momentum_and_energy_integrated_out (float containing the total (summed over x,y,z)
                                                                                           dynamical two-spin structure factor with
                                                                                           momentum and energy integrated out, described above)"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Index of the minimum eigenvector
    m_min = 0

    # Number of energy eigenvalues/eigenvectors in list_of_eigenvalues/list_of_eigenvectors
    num_eigenstates_to_sum_over = len(list_of_eigenvalues[m_min:m_max])

    # Compute the spin-spin correlations in e1,e2,e3 coordinates: <0|S_i^alpha S_j^beta|m> (alpha,beta=e1,e2,e3)
    spin_spin_correlations_e1e2e3_0m = spin_spin_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_0m)

    # Initialize the total (summed over e1,e2,e3) dynamical two-spin structure factor with momentum integrated out
    dynamical_two_spin_structure_factor_total_with_momentum_and_energy_integrated_out_e1e2e3 = 0.

    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site index of the sublattice B site in this unit cell
            site_index = get_site_index(Nx, i, j, 1)

            # Loop over the 3 bonds for the B site of this unit cell
            # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
            for bond_type in range(3):

                # Initialize Boolean that specifies whether or not this bond is a boundary bond
                # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
                boundary_bond = False

                # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
                if bond_type == 0:
                    i_neighbor = i
                    j_neighbor = (j + 1) % Ny
                    if j_neighbor != (j + 1):
                        boundary_bond = True
                elif bond_type == 1:
                    i_neighbor = (i - 1) % Nx
                    j_neighbor = (j + 1) % Ny
                    if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                        boundary_bond = True
                elif bond_type == 2:
                    i_neighbor = i
                    j_neighbor = j

                # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
                # and this is a boundary bond
                if (boundary_conditions == 'open') and (boundary_bond == True):
                    continue

                # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
                site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

                # Add the contributions from these neighboring sites
                dynamical_two_spin_structure_factor_total_with_momentum_and_energy_integrated_out_e1e2e3 += \
                    np.sum( np.abs( spin_spin_correlations_e1e2e3_0m['Sie1Sje1_0m'][site_index,site_index_neighbor,m_min:m_max] ) ** 2 + \
                            np.abs( spin_spin_correlations_e1e2e3_0m['Sie2Sje2_0m'][site_index,site_index_neighbor,m_min:m_max] ) ** 2 + \
                            np.abs( spin_spin_correlations_e1e2e3_0m['Sie3Sje3_0m'][site_index,site_index_neighbor,m_min:m_max] ) ** 2
                          )

    # Multiply dynamical_two_spin_structure_factor_total_with_momentum_and_energy_integrated_out_e1e2e3 by 2 pi
    dynamical_two_spin_structure_factor_total_with_momentum_and_energy_integrated_out_e1e2e3 *= 2*np.pi

    return dynamical_two_spin_structure_factor_total_with_momentum_and_energy_integrated_out_e1e2e3

def raman_intensity(J_dir, K_dir, G_dir, J_indir, K_indir, G_indir, omega_values, ground_state_eigenvalue, list_of_eigenvalues, spin_spin_correlations_0m, Nx, Ny, epsilon_in, epsilon_out, m_min = 0, m_max = None, eta = 0.01, boundary_conditions = 'periodic'):
    """Function that computes the Raman intensity for a superexchange-mediated honeycomb system of magnetic metal ions
    with edge-sharing ligand octahedra described by the JKG (Heisenberg + Kitaev + symmetric off-diagonal) Hamiltonian
    (see the function JKG_hamiltonian(...)). The Raman intensity is given by
        I(omega) = 2 pi Sum_{m=m_min}^{m_max-1} |<0|R|m>|^2 delta(omega - (E_m - E_0))
    where
        R = Sum_{<ij> in alpha beta (gamma)} [ (P_dir^gamma J_dir + P_indir^gamma J_indir) S_i.S_j +
                                               (P_dir^gamma K_dir + P_indir^gamma K_indir) S_i^gamma S_j^gamma +
                                               (P_dir^gamma G_dir + P_indir^gamma G_indir) (S_i^alpha S_j^beta + S_i^beta S_j^alpha) ]
    is the Raman operator for the system,
    Sum_{<ij> in alpha beta (gamma)} denotes a sum over neighboring sites interacting through a gamma-bond
    (gamma=x,y,z, and alpha,beta are the other two directions),
    P_dir^gamma and P_indir^gamma are the polarization factors associated with direct and indirect exchange, given by
        P_dir^gamma = 1/2 * (epsilon_in^alpha - epsilon_in^beta) (epsilon_out^alpha - epsilon_out^beta)
        P_indir^gamma = epsilon_in^alpha epsilon_out^alpha + epsilon_in^beta epsilon_out^beta
    S_i.S_j denotes a dot product between the spins on sites i and j, and
    epsilon_in and epsilon_out are the polarization vectors of the incoming and outgoing photons.
    Inputs:
        J_dir (Heisenberg coupling - direct exchange contribution)
        K_dir (Kitaev coupling - direct exchange contribution)
        G_dir (Gamma coupling - direct exchange contribution)
        J_indir (Heisenberg coupling - indirect exchange contribution (i.e., superexchange contribution))
        K_indir (Kitaev coupling - indirect exchange contribution (i.e., superexchange contribution))
        G_indir (Gamma coupling - indirect exchange contribution (i.e., superexchange contribution))
        omega_values (1D NumPy array of size N_omega)
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        epsilon_in, epsilon_out (lists specifying the directions of the incoming and outgoing photon polarizations in x,y,z coordinates)
        m_min, m_max (indices of the minimum and maximum (minus one) eigenvectors |m> to sum over;
                      (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_min=0 and m_max=400)
                      if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
        eta (float specifying the half-width at half-maximum of the Lorentzian distributions used for the delta functions)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Output:
        raman_intensity (1D NumPy arrays of size N_omega containing the Raman intensity, described above;
                         the epsilonth element of this array corresponds to the value of the
                         Raman intensity at energy value omega = omega_values[epsilon])"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Convert epsilon_in, epsilon_out into NumPy arrays and make them unit vectors
    epsilon_in = np.array(epsilon_in)
    epsilon_in = epsilon_in / np.linalg.norm(epsilon_in)

    epsilon_out = np.array(epsilon_out)
    epsilon_out = epsilon_out / np.linalg.norm(epsilon_out)

    # Number of energy eigenvalues/eigenvectors in list_of_eigenvalues/list_of_eigenvectors
    num_eigenstates_to_sum_over = len(list_of_eigenvalues[m_min:m_max])

    # Initialize the matrix element of the Raman operator between the ground state and the mth excited state: <0|R|m>
    # (Note: This is a 1D NumPy array of size num_eigenstates_to_sum_over = m_max - m_min;
    #        the mth entry of this array is
    #            raman_matrix_element_0m[m] = <0|R|m>
    #        where m indexes the energy eigenvalues E_m and eigenvectors |m>)
    raman_matrix_element_0m = np.full(num_eigenstates_to_sum_over, 0.)

    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site index of the sublattice B site in this unit cell
            site_index = get_site_index(Nx, i, j, 1)

            # =========== Add bond interaction terms to the Raman operator ===========
            # Loop over the 3 bonds for the B site of this unit cell
            # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
            for bond_type in range(3):

                # Get the other two bond types (e.g., if bond_type = 0, then bond_type_other_1 = 1 and bond_type_other_2 = 0)
                bond_type_other_1 = (bond_type + 1) % 3
                bond_type_other_2 = (bond_type + 2) % 3

                # Initialize Boolean that specifies whether or not this bond is a boundary bond
                # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
                boundary_bond = False

                # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
                if bond_type == 0:
                    i_neighbor = i
                    j_neighbor = (j + 1) % Ny
                    if j_neighbor != (j + 1):
                        boundary_bond = True
                elif bond_type == 1:
                    i_neighbor = (i - 1) % Nx
                    j_neighbor = (j + 1) % Ny
                    if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                        boundary_bond = True
                elif bond_type == 2:
                    i_neighbor = i
                    j_neighbor = j

                # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
                # and this is a boundary bond
                if (boundary_conditions == 'open') and (boundary_bond == True):
                    continue

                # Factor by which to multiply each spin-spin interaction in this bond in order to implement antiperiodic boundary conditions,
                # if specified; this will be -1 for antiperiodic boundary bonds, and +1 for all other bonds
                if (boundary_conditions == 'antiperiodic') and (boundary_bond == True):
                    bond_sign = -1
                else:
                    bond_sign = 1

                # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
                site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

                # Raman polarization factors (from both direct exchange and superexchange) for this type of bond
                P_dir = 1/2 * (epsilon_in[bond_type_other_1] - epsilon_in[bond_type_other_2]) * \
                              (epsilon_out[bond_type_other_1] - epsilon_out[bond_type_other_2])
                P_indir = epsilon_in[bond_type_other_1] * epsilon_out[bond_type_other_1] + \
                          epsilon_in[bond_type_other_2] * epsilon_out[bond_type_other_2]

                # Add the exchange terms for this bond to the matrix element <0|R|m>
                # Heisenberg
                if (J_dir != 0) or (J_indir != 0):
                    raman_matrix_element_0m += (P_dir*J_dir + P_indir*J_indir) * \
                        (spin_spin_correlations_0m['SixSjx_0m'][site_index,site_index_neighbor,m_min:m_max] + \
                         spin_spin_correlations_0m['SiySjy_0m'][site_index,site_index_neighbor,m_min:m_max] + \
                         spin_spin_correlations_0m['SizSjz_0m'][site_index,site_index_neighbor,m_min:m_max]) * bond_sign
                # Kitaev
                if (K_dir != 0) or (K_indir != 0):
                    if bond_type == 0:
                        raman_matrix_element_0m += (P_dir*K_dir + P_indir*K_indir) * \
                            spin_spin_correlations_0m['SixSjx_0m'][site_index,site_index_neighbor,m_min:m_max] * bond_sign
                    elif bond_type == 1:
                        raman_matrix_element_0m += (P_dir*K_dir + P_indir*K_indir) * \
                            spin_spin_correlations_0m['SiySjy_0m'][site_index,site_index_neighbor,m_min:m_max] * bond_sign
                    elif bond_type == 2:
                        raman_matrix_element_0m += (P_dir*K_dir + P_indir*K_indir) * \
                            spin_spin_correlations_0m['SizSjz_0m'][site_index,site_index_neighbor,m_min:m_max] * bond_sign
                # Gamma
                if (G_dir != 0) or (G_indir != 0):
                    if bond_type == 0:
                        raman_matrix_element_0m += (P_dir*G_dir + P_indir*G_indir) * \
                            (spin_spin_correlations_0m['SiySjz_0m'][site_index,site_index_neighbor,m_min:m_max] + \
                             spin_spin_correlations_0m['SizSjy_0m'][site_index,site_index_neighbor,m_min:m_max]) * bond_sign
                    elif bond_type == 1:
                        raman_matrix_element_0m += (P_dir*G_dir + P_indir*G_indir) * \
                            (spin_spin_correlations_0m['SizSjx_0m'][site_index,site_index_neighbor,m_min:m_max] + \
                             spin_spin_correlations_0m['SixSjz_0m'][site_index,site_index_neighbor,m_min:m_max]) * bond_sign
                    elif bond_type == 2:
                        raman_matrix_element_0m += (P_dir*G_dir + P_indir*G_indir) * \
                            (spin_spin_correlations_0m['SixSjy_0m'][site_index,site_index_neighbor,m_min:m_max] + \
                             spin_spin_correlations_0m['SiySjx_0m'][site_index,site_index_neighbor,m_min:m_max]) * bond_sign

    # Make the 2D NumPy array energy_delta_function of shape ( (m_max - m_min + 1) x N_omega );
    # the (m,e)th element of this array is delta(omega_e - (E_m - E_0)), where
    # omega_e = omega_values[e], E_m = list_of_eigenvalues[m], and E_0 is the ground-state energy eigenvalue
    energy_delta_function = lorentzian(omega_values[np.newaxis,:], list_of_eigenvalues[m_min:m_max][:,np.newaxis] - ground_state_eigenvalue, eta)

    # Compute the Raman intensity
    raman_intensity = 2*np.pi * (np.abs(raman_matrix_element_0m) ** 2) @ energy_delta_function

    return raman_intensity

def raman_intensity_without_polarization_dependence(J, K, G, omega_values, ground_state_eigenvalue, list_of_eigenvalues, spin_spin_correlations_0m, Nx, Ny, m_min = 0, m_max = None, eta = 0.01, boundary_conditions = 'periodic'):
    """Function that computes the Raman intensity for a superexchange-mediated honeycomb system of magnetic metal ions
    with edge-sharing ligand octahedra described by the JKG (Heisenberg + Kitaev + symmetric off-diagonal) Hamiltonian
    (see the function JKG_hamiltonian(...)). The Raman intensity is given by
        I(omega) = 2 pi Sum_{m=m_min}^{m_max-1} |<0|H_JKG|m>|^2 delta(omega - (E_m - E_0))
    where
        H_JKG = Sum_{<ij> in alpha beta (gamma)} [ J S_i.S_j + K S_i^gamma S_j^gamma + G (S_i^alpha S_j^beta + S_i^beta S_j^alpha) ]
    is the Hamiltonian for the system,
    Sum_{<ij> in alpha beta (gamma)} denotes a sum over neighboring sites interacting through a gamma-bond
    (gamma=x,y,z, and alpha,beta are the other two directions), and
    S_i.S_j denotes a dot product between the spins on sites i and j.
    Inputs:
        J (Heisenberg coupling)
        K (Kitaev coupling)
        G (Gamma coupling)
        omega_values (1D NumPy array of size N_omega)
        ground_state_eigenvalue (float specifying the ground-state energy eigenvalue of the system)
        list_of_eigenvalues (1D NumPy array of size N containing the N lowest energy eigenvalues of the system)
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        m_min, m_max (indices of the minimum and maximum (minus one) eigenvectors |m> to sum over;
                      (for example, if we want to sum over the 400 eigenvectors |0>, |1>, ..., |399>, we set m_min=0 and m_max=400)
                      if you want m_max to go up to N (i.e., the highest eigenstate index), then set m_max = None)
        eta (float specifying the half-width at half-maximum of the Lorentzian distributions used for the delta functions)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Output:
        raman_intensity_without_polarization_dependence (1D NumPy arrays of size N_omega containing the Raman intensity
                                                         without polarization dependence, described above;
                                                         the epsilonth element of this array corresponds to the value of the
                                                         Raman intensity at energy value omega = omega_values[epsilon])"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Number of energy eigenvalues/eigenvectors in list_of_eigenvalues/list_of_eigenvectors
    num_eigenstates_to_sum_over = len(list_of_eigenvalues[m_min:m_max])

    # Initialize the matrix element of the Raman operator without polarization dependence
    # between the ground state and the mth excited state: <0|R|m>
    # (Note: This is a 1D NumPy array of size num_eigenstates_to_sum_over = m_max - m_min;
    #        the mth entry of this array is
    #            raman_matrix_element_0m[m] = <0|R|m>
    #        where m indexes the energy eigenvalues E_m and eigenvectors |m>)
    raman_matrix_element_0m = np.full(num_eigenstates_to_sum_over, 0 + 0j)

    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site index of the sublattice B site in this unit cell
            site_index = get_site_index(Nx, i, j, 1)

            # =========== Add bond interaction terms to the Raman operator ===========
            # Loop over the 3 bonds for the B site of this unit cell
            # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
            for bond_type in range(3):

                # Get the other two bond types (e.g., if bond_type = 0, then bond_type_other_1 = 1 and bond_type_other_2 = 0)
                bond_type_other_1 = (bond_type + 1) % 3
                bond_type_other_2 = (bond_type + 2) % 3

                # Initialize Boolean that specifies whether or not this bond is a boundary bond
                # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
                boundary_bond = False

                # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
                if bond_type == 0:
                    i_neighbor = i
                    j_neighbor = (j + 1) % Ny
                    if j_neighbor != (j + 1):
                        boundary_bond = True
                elif bond_type == 1:
                    i_neighbor = (i - 1) % Nx
                    j_neighbor = (j + 1) % Ny
                    if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                        boundary_bond = True
                elif bond_type == 2:
                    i_neighbor = i
                    j_neighbor = j

                # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
                # and this is a boundary bond
                if (boundary_conditions == 'open') and (boundary_bond == True):
                    continue

                # Factor by which to multiply each spin-spin interaction in this bond in order to implement antiperiodic boundary conditions,
                # if specified; this will be -1 for antiperiodic boundary bonds, and +1 for all other bonds
                if (boundary_conditions == 'antiperiodic') and (boundary_bond == True):
                    bond_sign = -1
                else:
                    bond_sign = 1

                # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
                site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

                # Add the exchange terms for this bond to the matrix element <0|R|m>
                # Heisenberg
                if (J != 0):
                    raman_matrix_element_0m += J * \
                        (spin_spin_correlations_0m['SixSjx_0m'][site_index,site_index_neighbor,m_min:m_max] + \
                         spin_spin_correlations_0m['SiySjy_0m'][site_index,site_index_neighbor,m_min:m_max] + \
                         spin_spin_correlations_0m['SizSjz_0m'][site_index,site_index_neighbor,m_min:m_max]) * bond_sign
                # Kitaev
                if (K != 0):
                    if bond_type == 0:
                        raman_matrix_element_0m += K * \
                            spin_spin_correlations_0m['SixSjx_0m'][site_index,site_index_neighbor,m_min:m_max] * bond_sign
                    elif bond_type == 1:
                        raman_matrix_element_0m += K * \
                            spin_spin_correlations_0m['SiySjy_0m'][site_index,site_index_neighbor,m_min:m_max] * bond_sign
                    elif bond_type == 2:
                        raman_matrix_element_0m += K * \
                            spin_spin_correlations_0m['SizSjz_0m'][site_index,site_index_neighbor,m_min:m_max] * bond_sign
                # Gamma
                if (G != 0):
                    if bond_type == 0:
                        raman_matrix_element_0m += G * \
                            (spin_spin_correlations_0m['SiySjz_0m'][site_index,site_index_neighbor,m_min:m_max] + \
                             spin_spin_correlations_0m['SizSjy_0m'][site_index,site_index_neighbor,m_min:m_max]) * bond_sign
                    elif bond_type == 1:
                        raman_matrix_element_0m += G * \
                            (spin_spin_correlations_0m['SizSjx_0m'][site_index,site_index_neighbor,m_min:m_max] + \
                             spin_spin_correlations_0m['SixSjz_0m'][site_index,site_index_neighbor,m_min:m_max]) * bond_sign
                    elif bond_type == 2:
                        raman_matrix_element_0m += G * \
                            (spin_spin_correlations_0m['SixSjy_0m'][site_index,site_index_neighbor,m_min:m_max] + \
                             spin_spin_correlations_0m['SiySjx_0m'][site_index,site_index_neighbor,m_min:m_max]) * bond_sign

    # Make the 2D NumPy array energy_delta_function of shape ( (m_max - m_min + 1) x N_omega );
    # the (m,e)th element of this array is delta(omega_e - (E_m - E_0)), where
    # omega_e = omega_values[e], E_m = list_of_eigenvalues[m], and E_0 is the ground-state energy eigenvalue
    energy_delta_function = lorentzian(omega_values[np.newaxis,:], list_of_eigenvalues[m_min:m_max][:,np.newaxis] - ground_state_eigenvalue, eta)

    # Compute the Raman intensity without polarization dependence
    raman_intensity_without_polarization_dependence = 2*np.pi * (np.abs(raman_matrix_element_0m) ** 2) @ energy_delta_function

    return raman_intensity_without_polarization_dependence

def magnon_pairing_order_parameter(spin_spin_correlations_0m, Nx, Ny, boundary_conditions = 'periodic'):
    """Function that computes the magnon pairing order parameter for magnons pairs with their spins in the z direction, namely
        Delta_alpha = 1/num_bonds Sum_<ij> |<0|S_i^{+_z} S_j^{+_z}|0>|
    where alpha=+_z,-_z labels the orientation of the bound magnons' spins,
    <ij> denotes a sum over neighboring sites, and num_bonds = 3*num_sites/2 the number of bonds in the system.
    Inputs:
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Output:
        Delta_alpha (1D NumPy array of size 2 containing numbers that correspond to the magnon pairing order parameters;
                     specifically, Delta_alpha[0] = Delta_{+_z} and Delta_alpha[1] = Delta_{-_z})"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Compute the spin flip-spin flip correlations for spin flips along the z axis: <0|S_i^alpha S_j^beta|m> (alpha,beta=+_z,-_z)
    spin_flip_spin_flip_correlations_0m = spin_flip_spin_flip_correlations_between_a_bra_and_many_kets(spin_spin_correlations_0m)

    # Initialize 1D NumPy array that will contain the magnon pairing order parameters Delta_{x/y/z}^{+_z/-_z}
    Delta_alpha = np.full(2, 0.)

    num_x_bonds = 0
    num_y_bonds = 0
    num_z_bonds = 0
    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site index of the sublattice B site in this unit cell
            site_index = get_site_index(Nx, i, j, 1)

            # Loop over the 3 bonds for the B site of this unit cell
            # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
            for bond_type in range(3):

                # Initialize Boolean that specifies whether or not this bond is a boundary bond
                # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
                boundary_bond = False

                # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
                if bond_type == 0:
                    i_neighbor = i
                    j_neighbor = (j + 1) % Ny
                    if j_neighbor != (j + 1):
                        boundary_bond = True
                elif bond_type == 1:
                    i_neighbor = (i - 1) % Nx
                    j_neighbor = (j + 1) % Ny
                    if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                        boundary_bond = True
                elif bond_type == 2:
                    i_neighbor = i
                    j_neighbor = j

                # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
                # and this is a boundary bond
                if (boundary_conditions == 'open') and (boundary_bond == True):
                    continue

                # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
                site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

                # Add the contribution from these neighboring sites
                Delta_alpha[0] += np.abs(spin_flip_spin_flip_correlations_0m['Sip_z_Sjp_z_0m'][site_index,site_index_neighbor,0])
                Delta_alpha[1] += np.abs(spin_flip_spin_flip_correlations_0m['Sim_z_Sjm_z_0m'][site_index,site_index_neighbor,0])

                # Update bond counter
                num_bonds += 1

    # Normalize each element of Delta_alpha by dividing by the number of bonds summed over
    Delta_alpha *= 1 / num_bonds

    return Delta_alpha

def magnon_pairing_order_parameter_e1e2e3(spin_spin_correlations_0m, Nx, Ny, boundary_conditions = 'periodic'):
    """Function that computes the magnon pairing order parameter for magnons pairs with their spins in the e3 direction, namely
        Delta_alpha = 1/num_bonds Sum_<ij> |<0|S_i^{+_z} S_j^{+_z}|0>|
    where alpha=+_e3,-_e3 labels the orientation of the bound magnons' spins,
    <ij> denotes a sum over neighboring sites, and num_bonds = 3*num_sites/2 the number of bonds in the system.
    Inputs:
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Output:
        Delta_alpha (1D NumPy array of size 2 containing numbers that correspond to the magnon pairing order parameters;
                     specifically, Delta_alpha[0] = Delta_{+_e3} and Delta_alpha[1] = Delta_{-_e3})"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Compute the spin-spin correlations in e1,e2,e3 coordinates: <0|S_i^alpha S_j^beta|m> (alpha,beta=e1,e2,e3)
    spin_spin_correlations_e1e2e3_0m = spin_spin_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_0m)

    # Compute the spin flip-spin flip correlations for spin flips along the e3 axis: <0|S_i^alpha S_j^beta|m> (alpha,beta=+_e3,-_e3)
    spin_flip_spin_flip_correlations_e1e2e3_0m = spin_flip_spin_flip_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_e1e2e3_0m)

    # Initialize 1D NumPy array that will contain the magnon pairing order parameters Delta_{x/y/z}^{+_e3/-_e3}
    Delta_alpha = np.full(2, 0.)

    num_bonds = 0
    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site index of the sublattice B site in this unit cell
            site_index = get_site_index(Nx, i, j, 1)

            # Loop over the 3 bonds for the B site of this unit cell
            # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
            for bond_type in range(3):

                # Initialize Boolean that specifies whether or not this bond is a boundary bond
                # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
                boundary_bond = False

                # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
                if bond_type == 0:
                    i_neighbor = i
                    j_neighbor = (j + 1) % Ny
                    if j_neighbor != (j + 1):
                        boundary_bond = True
                elif bond_type == 1:
                    i_neighbor = (i - 1) % Nx
                    j_neighbor = (j + 1) % Ny
                    if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                        boundary_bond = True
                elif bond_type == 2:
                    i_neighbor = i
                    j_neighbor = j

                # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
                # and this is a boundary bond
                if (boundary_conditions == 'open') and (boundary_bond == True):
                    continue

                # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
                site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

                # Add the contribution from these neighboring sites
                Delta_alpha[0] += np.abs(spin_flip_spin_flip_correlations_e1e2e3_0m['Sip_e3_Sjp_e3_0m'][site_index,site_index_neighbor,0])
                Delta_alpha[1] += np.abs(spin_flip_spin_flip_correlations_e1e2e3_0m['Sim_e3_Sjm_e3_0m'][site_index,site_index_neighbor,0])

                # Update bond counter
                num_bonds += 1

    # Normalize each element of Delta_alpha by dividing by the number of bonds summed over
    Delta_alpha *= 1 / num_bonds

    return Delta_alpha

def magnon_pairing_order_parameter_by_bond(spin_spin_correlations_0m, Nx, Ny, boundary_conditions = 'periodic'):
    """Function that computes the magnon pairing order parameter for magnons pairs with their spins in the z direction
    for each type of bond, namely
        Delta^alpha_beta = 1/num_z_bonds Sum_{<ij>_beta} |<0|S_i^alpha S_j^alpha|0>|
    where alpha=+_z,-_z labels the orientation of the bound magnons' spins,
    <ij> denotes a sum over neighboring sites interacting through a beta bond (beta=x,y,z),
    and num_z_bonds = num_x_bonds = num_y_bonds = num_sites/2 is the number of bonds of a given type in the system.
    Inputs:
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Output:
        Delta_alpha_beta (2D NumPy array of shape 2 x 3 containing numbers that correspond to the magnon pairing order parameters
                          Delta^alpha_beta (alpha=x,y,z; beta=+_z,-_z), described above;
                          the structure of Delta_alpha_beta is [[Delta^{+_z}_x, Delta^{+_z}_y, Delta^{+_z}_z],
                                                                [Delta^{-_z}_x, Delta^{-_z}_y, Delta^{-_z}_z]])"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Compute the spin flip-spin flip correlations for spin flips along the z axis: <0|S_i^alpha S_j^beta|m> (alpha,beta=+_z,-_z)
    spin_flip_spin_flip_correlations_0m = spin_flip_spin_flip_correlations_between_a_bra_and_many_kets(spin_spin_correlations_0m)

    # Initialize 1D NumPy array that will contain the magnon pairing order parameters Delta^{+_z/-_z}_{x/y/z}
    Delta_alpha_beta = np.full((2,3), 0.)

    num_x_bonds = 0
    num_y_bonds = 0
    num_z_bonds = 0
    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site index of the sublattice B site in this unit cell
            site_index = get_site_index(Nx, i, j, 1)

            # Loop over the 3 bonds for the B site of this unit cell
            # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
            for bond_type in range(3):

                # Initialize Boolean that specifies whether or not this bond is a boundary bond
                # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
                boundary_bond = False

                # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
                if bond_type == 0:
                    i_neighbor = i
                    j_neighbor = (j + 1) % Ny
                    if j_neighbor != (j + 1):
                        boundary_bond = True
                elif bond_type == 1:
                    i_neighbor = (i - 1) % Nx
                    j_neighbor = (j + 1) % Ny
                    if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                        boundary_bond = True
                elif bond_type == 2:
                    i_neighbor = i
                    j_neighbor = j

                # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
                # and this is a boundary bond
                if (boundary_conditions == 'open') and (boundary_bond == True):
                    continue

                # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
                site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

                # Add the contribution from these neighboring sites
                Delta_alpha_beta[0,bond_type] += np.abs(spin_flip_spin_flip_correlations_0m['Sip_z_Sjp_z_0m'][site_index,site_index_neighbor,0])
                Delta_alpha_beta[1,bond_type] += np.abs(spin_flip_spin_flip_correlations_0m['Sim_z_Sjm_z_0m'][site_index,site_index_neighbor,0])

                # Update bond counters
                if bond_type == 0:
                    num_x_bonds += 1
                elif bond_type == 1:
                    num_y_bonds += 1
                elif bond_type == 2:
                    num_z_bonds += 1

    # Normalize each element of Delta_alpha_beta by dividing by the number of bonds summed over
    Delta_alpha_beta[:,0] *= 1 / num_x_bonds
    Delta_alpha_beta[:,1] *= 1 / num_y_bonds
    Delta_alpha_beta[:,2] *= 1 / num_z_bonds

    return Delta_alpha_beta

def magnon_pairing_order_parameter_by_bond_e1e2e3(spin_spin_correlations_0m, Nx, Ny, boundary_conditions = 'periodic'):
    """Function that computes the magnon pairing order parameter for magnons pairs with their spins in the e3 direction
    for each type of bond, namely
        Delta^alpha_beta = 1/num_z_bonds Sum_{<ij>_beta} |<0|S_i^alpha S_j^alpha|0>|
    where alpha=+_e3,-_e3 labels the orientation of the bound magnons' spins,
    <ij> denotes a sum over neighboring sites interacting through a beta bond (beta=x,y,z),
    and num_z_bonds = num_x_bonds = num_y_bonds = num_sites/2 is the number of bonds of a given type in the system.
    Inputs:
        spin_spin_correlations_0m (dictionary containing the arrays following 9 NumPy arrays:
                                       SixSjx_0m, SixSjy_0m, SixSjz_0m,
                                       SiySjx_0m, SiySjy_0m, SiySjz_0m,
                                       SizSjx_0m, SizSjy_0m, SizSjz_0m;
                                   these arrays are 3D NumPy arrays of shape num_sites x num_sites x N, where num_sites = 2 Nx Ny;
                                   the (i,j,m)th element of SixSjy_0m corresponds to the matrix element <bra|S_i^x S_j^y|ket_m>,
                                   and similarly for the other arrays)
        Nx, Ny (length of system along the two lattice directions)
        boundary_conditions (string specifying the boundary conditions; can be either 'periodic', 'antiperiodic', or 'open')
    Output:
        Delta_alpha_beta (2D NumPy array of shape 2 x 3 containing numbers that correspond to the magnon pairing order parameters
                          Delta^alpha_beta (alpha=x,y,z; beta=+_e3,-_e3), described above;
                          the structure of Delta_alpha_beta is [[Delta^{+_e3}_x, Delta^{+_e3}_y, Delta^{+_e3}_z],
                                                                [Delta^{-_e3}_x, Delta^{-_e3}_y, Delta^{-_e3}_z]])"""

    # Number of sites
    num_sites = 2 * Nx * Ny

    # Compute the spin-spin correlations in e1,e2,e3 coordinates: <0|S_i^alpha S_j^beta|m> (alpha,beta=e1,e2,e3)
    spin_spin_correlations_e1e2e3_0m = spin_spin_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_0m)

    # Compute the spin flip-spin flip correlations for spin flips along the e3 axis: <0|S_i^alpha S_j^beta|m> (alpha,beta=+_e3,-_e3)
    spin_flip_spin_flip_correlations_e1e2e3_0m = spin_flip_spin_flip_correlations_between_a_bra_and_many_kets_e1e2e3(spin_spin_correlations_e1e2e3_0m)

    # Initialize 1D NumPy array that will contain the magnon pairing order parameters Delta^{+_e3/-_e3}_{x/y/z}
    Delta_alpha_beta = np.full((2,3), 0.)

    num_x_bonds = 0
    num_y_bonds = 0
    num_z_bonds = 0
    # Loop over all the unit cells, which are indexed by i and j
    # (Note: i and j are NOT site labels here; they are indices that specify a unit cell's coordinates along the
    #        two lattice directions a1 and a2)
    for i in range(Nx):
        for j in range(Ny):

            # Site index of the sublattice B site in this unit cell
            site_index = get_site_index(Nx, i, j, 1)

            # Loop over the 3 bonds for the B site of this unit cell
            # (Note: bond_type=0 corresponds to x-bonds, bond_type=1 to y-bonds, and bond_type=2 to z-bonds.)
            for bond_type in range(3):

                # Initialize Boolean that specifies whether or not this bond is a boundary bond
                # (i.e., a bond between sites on opposite ends of the lattice linked through periodic or antiperiodic boundary conditions)
                boundary_bond = False

                # Get the (i,j) index of the sublattice A site neighbor that the B site at (i,j) is bonding with
                if bond_type == 0:
                    i_neighbor = i
                    j_neighbor = (j + 1) % Ny
                    if j_neighbor != (j + 1):
                        boundary_bond = True
                elif bond_type == 1:
                    i_neighbor = (i - 1) % Nx
                    j_neighbor = (j + 1) % Ny
                    if (i_neighbor != (i - 1)) or (j_neighbor != (j + 1)):
                        boundary_bond = True
                elif bond_type == 2:
                    i_neighbor = i
                    j_neighbor = j

                # Skip to the next iteration of the current for loop (over bond_type in range(3)) if boundary_conditions = 'open'
                # and this is a boundary bond
                if (boundary_conditions == 'open') and (boundary_bond == True):
                    continue

                # Get the site index of the A site neighbor that the B site at (i,j) is bonding with
                site_index_neighbor = get_site_index(Nx, i_neighbor, j_neighbor, 0)

                # Add the contribution from these neighboring sites
                Delta_alpha_beta[0,bond_type] += np.abs(spin_flip_spin_flip_correlations_e1e2e3_0m['Sip_e3_Sjp_e3_0m'][site_index,site_index_neighbor,0])
                Delta_alpha_beta[1,bond_type] += np.abs(spin_flip_spin_flip_correlations_e1e2e3_0m['Sim_e3_Sjm_e3_0m'][site_index,site_index_neighbor,0])

                # Update bond counters
                if bond_type == 0:
                    num_x_bonds += 1
                elif bond_type == 1:
                    num_y_bonds += 1
                elif bond_type == 2:
                    num_z_bonds += 1

    # Normalize each element of Delta_alpha_beta by dividing by the number of bonds summed over
    Delta_alpha_beta[:,0] *= 1 / num_x_bonds
    Delta_alpha_beta[:,1] *= 1 / num_y_bonds
    Delta_alpha_beta[:,2] *= 1 / num_z_bonds

    return Delta_alpha_beta

def state_vector_in_terms_of_spin_configurations(state_vector, use_arrows = True, num_decimal_places = 2):
    """Function that represents a state vector corresponding to a system of spin-1/2 particles in terms of
    the spin configurations that make up this state.
    Inputs:
        state_vector (1D NumPy array)
        use_arrows (Boolean specifying whether to use arrows to represent the spin configurations; alternatively, use 0's and 1's.
                    For example, if use_arrows = True, a spin configuration will be represented in the form |↑↓↑↓>;
                                 if use_arrows = True, a spin configuration will be represented in the form |0101>)
        num_decimal_places (integer specifying the number of decimal places to round each amplitude to)
    Output:
        state_vector_configs_str (string representing the state vector in terms of the spin configurations that make it up)"""

    # Hilbert space size
    hilbert_space_size = len(state_vector)

    # Number of spin-1/2 particles in the system
    num_particles = int(np.log2(hilbert_space_size))

    # Clean up the state vector
    state_vector = chop(state_vector)

    # Indices corresponding to the configurations that contribute to the state vector
    # (i.e., the configurations that have a nonzero amplitude)
    config_indices = np.nonzero(state_vector)[0]

    # Initialize string that will contain the state vector in terms of the spin configurations
    state_vector_configs_str = ""

    # Loop over configurations that contribute to the state vector
    for config_index in config_indices:

        # Round this amplitude to 2 decimal places
        amplitude = np.round(state_vector[config_index], num_decimal_places)

        # If the amplitude is real, make 'amplitude' a float instead of a complex number
        if np.imag(amplitude) == 0:
            amplitude = np.real(amplitude)

        # String version of the amplitude for this configuration
        amplitude_str = str(amplitude).replace('j','i')

        # Ket (string) version of this configuration
        config_ket = '|' + np.binary_repr(config_index, num_particles) + '>'

        # If requested, convert binary representation to arrow representation
        if use_arrows == True:
            config_ket = config_ket.replace('0','↑').replace('1','↓')

        # Add this configuration's contribution to state_vector_configs_str
        state_vector_configs_str += amplitude_str + config_ket

        # If this isn't the last configuration, add +
        if config_index != config_indices[-1]:
            state_vector_configs_str += " + "

    # Replace all instances of '+ -' for '- '
    state_vector_configs_str.replace('+ -','- ')

    return state_vector_configs_str

def data_filename(num_sites, spin, J, K, G, h, h_direction, file_extension, num_decimal_places_h = 3, num_evals = 500, boundary_conditions = 'periodic', basis = 'z', diagonalization_method = 'lanczos'):
    """Function that generates a filename for the data file in which to store the variables for this calculation."""

    # Initialize filename
    filename = ''

    # # Make h_direction, epsilon_in, and epsilon_out NumPy arrays and normalize them
    # h_direction = np.array(h_direction)
    # h_direction = h_direction / np.linalg.norm(h_direction)
    #
    # epsilon_in = np.array(epsilon_in)
    # epsilon_in = epsilon_in / np.linalg.norm(epsilon_in)
    #
    # epsilon_out = np.array(epsilon_out)
    # epsilon_out = epsilon_out / np.linalg.norm(epsilon_out)

    # Get the e3 unit vector in terms of its crystal coordinates x,y,z
    _, _, e3 = e1e2e3_unit_vectors_in_terms_of_xyz_coordinates()

    # Make h_direction a NumPy array and normalize it
    h_direction = np.array(h_direction)
    h_direction_unit = h_direction / np.linalg.norm(h_direction)

    # Get projection of the magnetic field directional vector in the out-of-plane (OP) direction
    h_direction_OP_projection = np.round(h_direction_unit @ e3, 6)

    # Get x,y,z components of the projection of the magnetic field directional vector in the out-of-plane (OP) direction
    h_direction_x_component = np.round(h_direction_unit[0], 6)
    h_direction_y_component = np.round(h_direction_unit[1], 6)
    h_direction_z_component = np.round(h_direction_unit[2], 6)

    # # Get dot product of these two epsilon_XXX vectors
    # epsilon_dot_product = np.round(epsilon_in @ epsilon_out, 6)

    # Express the magnetic field strength up to the specified number of decimal places
    h_specified_decimal_places = ('%.'+str(num_decimal_places_h)+'f') % h

    # Add information to describe this calculation
    filename += 'N_' + str(num_sites)
    if spin == 1/2:
        filename += '_s_1,2_'
    elif spin == 1:
        filename += '_s_1_'
    elif spin == 3/2:
        filename += '_s_3,2_'
    elif spin == 2:
        filename += '_s_2_'
    if J != 0:
        filename += 'J'
        if J < 0:
            filename += '-'
        else:
            filename += '+'
    if K != 0:
        filename += 'K'
        if K < 0:
            filename += '-'
        else:
            filename += '+'
    if G != 0:
        filename += 'G'
        if G < 0:
            filename += '-'
        else:
            filename += '+'
    # if boundary_conditions == 'periodic':
    #     filename += '_PBC'
    if boundary_conditions == 'antiperiodic':
        filename += '_APBC'
    elif boundary_conditions == 'open':
        filename += '_open'
    filename += '_h_' + str(h_specified_decimal_places).replace('.', ',')
    if h != 0:
        if h_direction_OP_projection == 1:
            filename += '_OP' # OP = out-of-plane
        elif h_direction_OP_projection == 0:
            filename += '_IP' # IP = in-plane
        elif h_direction_x_component == 1:
            filename += '_x'
        elif h_direction_y_component == 1:
            filename += '_y'
        elif h_direction_z_component == 1:
            filename += '_z'
    if num_evals != 500:
        filename += '_states_' + str(num_evals)
    if basis != 'z':
        filename += '_basis_' + basis
    if diagonalization_method == 'exact':
        filename += '_exact'

    # Add file extension
    filename += '.' + file_extension

    return filename

# ================================================================================================================================



# ====================================== Old functions we are keeping for future reference =======================================

def matrix_elements_between_a_bra_and_many_kets(bra, operator, list_of_kets):
    """Computes the matrix elements
        <bra|operator|ket_m>
    for the given operator between the eigenvector |bra> and the N eigenvectors {|ket_m>} (m=0,1,...,N-1).
    Inputs:
        bra (1D NumPy array of size D)
        operator (matrix of shape D x D)
        list_of_kets (2D NumPy array of shape D x N, where N is the number of eigenvectors (i.e., kets) in list_of_kets)
    Output:
        list_of_matrix_elements (1D NumPy array of size N; each element corresponds to an eigenvector |m>)
    Note: The input bra is just the regular eigenvector |bra> without complex conjugating it.
          In other words, you should NOT input its complex conjugate <bra| = np.conj(|bra>)."""

    matrix_elements = np.conj(bra) @ operator @ list_of_kets

    return matrix_elements

def matrix_elements_between_many_bras_and_a_ket(list_of_bras, operator, ket):
    """Computes the matrix elements
        <bra_m|operator|ket>
    for the given operator between the N eigenvectors {|bra_m>} (m=0,1,...,N-1) and the eigenvector |ket>.
    Inputs:
        list_of_bras (2D NumPy array of shape D x N, where N is the number of eigenvectors (i.e., bras) in list_of_bras)
        operator (matrix of shape D x D)
        ket (1D NumPy array of size D)
    Output:
        list_of_matrix_elements (1D NumPy array of size N; each element corresponds to an eigenvector |m>)
    Note: The input bras are just the regular eigenvectors {|bra_m>} (m=0,1,...,N-1) without complex conjugating them.
          In other words, you should NOT input its complex conjugates {<bra_0|, ... ,<bra_(N-1)|} = np.conj({|bra_m>} (m=0,1,...,N-1))."""

    list_of_matrix_elements = np.conj(list_of_bras).T @ operator @ ket

    return list_of_matrix_elements

# ================================================================================================================================
