import os
import pathlib
from mps_reader import read, eliminate_fixed_variables,\
    extract_matrix_data, expand_matrix_data, parse_mps_file

this_dir = pathlib.Path(__file__).parent.resolve()
for prob_name in os.listdir(this_dir/"mps_problems"):
    print("parsing", prob_name)
    prob_data = parse_mps_file(this_dir/"mps_problems"/prob_name)
    matrix_data = extract_matrix_data(prob_data)
    if matrix_data['fixed_inds'].shape[0] > 0:
        print("There are fixed variables. Eliminating them...")
    reduced_prob = eliminate_fixed_variables(matrix_data)
    assert reduced_prob['fixed_inds'].shape[0] == 0
    expanded_data = expand_matrix_data(matrix_data)

