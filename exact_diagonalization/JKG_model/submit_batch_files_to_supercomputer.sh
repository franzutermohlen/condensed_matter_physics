#!/bin/bash

# This script is for submitting many batch jobs to the supercomputer to perform a field sweep
# (i.e., to probe many different evenly-spaced values of the magnetic field strength).

# =================================== Inputs ===================================
# Spin
spin="1/2"

# Number of sites along the x and y directions
Nx=3
Ny=3

# Model (J=Heisenberg, K=Kitaev, G=symmetric off-diagonal (Gamma))
model='J'

# Sign of the coupling (negative is ferromagnetic and positive is antiferromagnetic)
sign='+'

# Minimum and maximum values of the external magnetic field to probe, and
# the spacing between them
h_min=0.010
h_max=1.000
delta_h=0.010

# Direction of the external magnetic field in [x,y,z] coordinates
h_direction='[1,1,1]'

# Maximum number of lowest energy eigenvalues/eigenvectors to find
num_evals=500

# Boundary conditions; can be either 'periodic', 'antiperiodic', or 'open'
boundary_conditions='periodic'

# Diagonalization method; can be either 'lanczos' or 'exact'
diagonalization_method='lanczos'

# Basis used to represent the spin matrices; can be either 'z' or 'e3'
basis='z'

# ===================== Compute specifications =====================
# Hilbert space dimension: 2^12  ---  num_evals: 400
# days=0
# hours=00
# minutes=10
# nodes=1
# ppn=1
# mem_in_GB=2

# Hilbert space dimension: 2^18 (with h = 0)  ---  num_evals: 400
# days=0
# hours=01
# minutes=00
# nodes=1
# ppn=1
# mem_in_GB=10

# Hilbert space dimension: 2^18 (with h != 0) ---  num_evals: 400
# days=0
# hours=5
# minutes=00
# nodes=1
# ppn=1
# mem_in_GB=10

# Hilbert space dimension: 2^18 (with h != 0) ---  num_evals: 500
days=0
hours=16
minutes=00
nodes=1
ppn=1
mem_in_GB=10

# Hilbert space dimension: 2^19  ---  num_evals: 400
# days=0
# hours=03
# minutes=00
# nodes=1
# ppn=1
# mem_in_GB=15

# Hilbert space dimension: 2^24  ---  num_evals: 200
# days=3
# hours=00
# minutes=00
# nodes=2
# ppn=1
# mem_in_GB=200

# Hilbert space dimension: 2^24  ---  num_evals: 400
# days=4
# hours=00
# minutes=00
# nodes=3
# ppn=1
# mem_in_GB=300
# ==================================================================

# Template files to use
py_template="run_simulation-template.py"
pbs_template="batch_file-template.pbs"

# Local paths to the directories in which to save the pickle files and the text outputs
# (They will be automatically created if they don't exist)
output_directory="outputs"
saved_data_directory="saved_data"
# ==============================================================================

# Total number of sites
numSites=$(( 2 * $Nx * $Ny ))

# Dimension of the Hilbert space of this system
if [[ $spin == '1/2' ]]; then
	dim_Hilbert_space=$(( 2 ** $numSites ))
elif [[ $spin == '1' ]]; then
	dim_Hilbert_space=$(( 3 ** $numSites ))
elif [[ $spin == '3/2' ]]; then
	dim_Hilbert_space=$(( 4 ** $numSites ))
elif [[ $spin == '2' ]]; then
	dim_Hilbert_space=$(( 5 ** $numSites ))
fi

# If diagonalization_method = 'exact' or num_evals > dim_Hilbert_space, then set num_evals = dim_Hilbert_space
if [[ $diagonalization_method == 'exact' ]]; then
	num_evals=$dim_Hilbert_space
elif (( $num_evals > $dim_Hilbert_space )); then
	num_evals=$dim_Hilbert_space
fi

# String describing the spin, but slashes are replaced by commas (for naming files)
spin_commas=$(echo "$spin" | tr '/' ',')

# String describing the direction of the external magnetic field (for naming files)
if [[ $h_direction == '[1,1,1]' ]]; then
	field_direction_tag="_OP"
elif [[ $h_direction == '[1,0,0]' ]]; then
	field_direction_tag="_x"
elif [[ $h_direction == '[0,1,0]' ]]; then
	field_direction_tag="_y"
elif [[ $h_direction == '[0,0,1]' ]]; then
	field_direction_tag="_z"
else
	field_direction_tag=""
fi

# String describing the number of eigenstates to compute (for naming files)
if [[ $num_evals != 500 ]]; then
	evals_tag="_states_$num_evals"
else
	evals_tag=""
fi

# String describing the boundary conditions (for naming files)
if [[ $boundary_conditions == 'periodic' ]]; then
	boundary_conditions_tag=""
elif [[ $boundary_conditions == 'antiperiodic' ]]; then
	boundary_conditions_tag="_APBC"
elif [[ $boundary_conditions == 'open' ]]; then
	boundary_conditions_tag="_open"
fi

# String describing the basis used to represent the spin matrices (for naming files)
if [[ $basis != 'z' ]]; then
	basis_tag="_basis_$basis"
else
	basis_tag=""
fi

# String describing the diagonalization method (for naming files)
if [[ $diagonalization_method == 'exact' ]]; then
	diagonalization_method_tag="_exact"
else
	diagonalization_method_tag=""
fi

# Name of directories in which to save .txt output files and .pickle save files
directory_name_tag="N_${numSites}_s_$spin_commas"

# Name of directories in which to save text output files (.txt) and pickle save files (.pickle) (including their local paths)
text_output_directory="$output_directory/$directory_name_tag"
pickle_file_save_directory="$saved_data_directory/$directory_name_tag/field_sweep/h$field_direction_tag/${model}${sign}/$boundary_conditions/states_$num_evals/new"

# echo "Text output file directory: $text_output_directory"
# echo "Pickle file save directory: $pickle_file_save_directory"
# echo ""

# Create these two directories if they don't exist
[[ -d $text_output_directory ]] || mkdir -p $text_output_directory
[[ -d $pickle_file_save_directory ]] || mkdir -p $pickle_file_save_directory

# =========== Define the values of J, K, and G, in $py_file ===========
# Energy value of the coupling constant
E0=1
# Sign of the coupling constant
if [[ $sign == '-' ]]; then
	E0="-$E0"
fi
# Initialize coupling constants to 0
J0=0
K0=0
G0=0
# Define the values of the coupling constants
if [[ $model == 'J' ]]; then
	J0=$E0
elif [[ $model == 'K' ]]; then
	K0=$E0
elif [[ $model == 'G' ]]; then
	G0=$E0
fi
# =====================================================================

# Loop over the different magnetic field strengths
for h0 in $(seq $h_min $delta_h $h_max); do
	# Get the relevant string from which to obtain the number of decimal places to keep in h0
	# If h_min != h_max, then relevant_h_string=delta_h (excluding trailing zeros);
	# otherwise, relevant_h_string=h_min
	if [[ $h_min != $h_max ]]; then
		relevant_h_string=$delta_h
	else
		relevant_h_string=$h_min
	fi

	# Get the number of decimal places to keep in h0
	if [[ $relevant_h_string == *"."* ]]; then
		# Get only the decimals in relevant_h_string
		relevant_h_string_decimals=$(echo "$relevant_h_string" | sed "s/.*\.//g")
		# Get rid of trailing decimal zeros
		relevant_h_string_decimals=$(echo "$relevant_h_string_decimals" | sed "s/0*$//g")

		num_decimal_places_h0=${#relevant_h_string_decimals}
	else
		num_decimal_places_h0=0
	fi

	# Express the magnetic field strength up to the same number of decimal places as delta_h
	h0=$(printf "%0.${num_decimal_places_h0}f\n" $h0)

	# Magnetic field strength as a string in which periods are replaced for commas
	h0_commas=$(echo "$h0" | tr '.' ',')

	# Strings that will uniquely identify each parameter combination from this loop
	# The filename_tag will be used to name the files we will make, and the job_name (max. 16 characters)
	# will be used to name the jobs will submit to the supercomputer
	job_name="N=${numSites}_s=${spin_commas}_${model}${sign}${boundary_conditions_tag}"
	filename_tag="${directory_name_tag}_${model}${sign}${boundary_conditions_tag}"

	# Add the information about external the magnetic field
	# if [[ $h0 != 0 ]]; then
	if [[ $h0 != 'always do this' ]]; then
		job_name+="_h=$h0_commas"
		filename_tag+="_h_${h0_commas}${field_direction_tag}"
	fi

	# Add the information about the number of eigenvalues to compute, the basis used, and the diagonalization method
	job_name+="${evals_tag}${basis_tag}"
	filename_tag+="${evals_tag}${basis_tag}${diagonalization_method_tag}"

	# Names of the various files we will be creating
	pbs_file="$filename_tag.pbs"
	py_file="../$filename_tag.py"
	output_file="$text_output_directory/$filename_tag.txt"

	echo "Job name:     $job_name"
	# echo "Filename tag: $filename_tag"
	# echo "pbs file:     $pbs_file"
	# echo "py file:      $py_file"
	# echo ""

	# Copy and modify the .py file
	cp $py_template $py_file
	sed -i "s|spin0|$spin|g" $py_file
	sed -i "s/Nx0/$Nx/g" $py_file
	sed -i "s/Ny0/$Ny/g" $py_file
	sed -i "s/J0/$J0/g" $py_file
	sed -i "s/K0/$K0/g" $py_file
	sed -i "s/G0/$G0/g" $py_file
	sed -i "s/h_magnitude0/$h0/g" $py_file
	sed -i "s/h_direction0/$h_direction/g" $py_file
	sed -i "s/boundary_conditions0/$boundary_conditions/g" $py_file
	sed -i "s/method0/$diagonalization_method/g" $py_file
	sed -i "s/num_evals0/$num_evals/g" $py_file
	sed -i "s/basis0/$basis/g" $py_file
	sed -i "s/num_decimal_places_h0/$num_decimal_places_h0/g" $py_file
	sed -i "s|local_path0|$pickle_file_save_directory|g" $py_file

	# Copy and modify the batch file
	cp $pbs_template $pbs_file
	sed -i "s/days0/$days/g" $pbs_file
	sed -i "s/hours0/$hours/g" $pbs_file
	sed -i "s/minutes0/$minutes/g" $pbs_file
	sed -i "s/nodes0/$nodes/g" $pbs_file
	sed -i "s/ppn0/$ppn/g" $pbs_file
	sed -i "s/mem0/$mem_in_GB/g" $pbs_file
	sed -i "s|job_name|$job_name|g" $pbs_file
	sed -i "s|python_filename|$py_file|g" $pbs_file
	sed -i "s|output_filename|$output_file|g" $pbs_file

	# Submit job
	qsub $pbs_file
	rm $pbs_file
done
