import os
import pathlib
from mps_reader import read, eliminate_fixed_variables,\
    extract_matrix_data, expand_matrix_data, parse_mps_file
from scipy.optimize import linprog
import numpy as np
#solns is a local file
from solns import obj_vals #dictionary of optimal objective values

def solve_from_matrix_data(md, shift):
    ed = expand_matrix_data(md)
    c, A_ub, b_ub, A_eq, b_eq=\
    ed['c'], ed['A_ub'], ed['b_ub'], ed['A_eq'], ed['b_eq']
    bounds = list(zip(ed['l'], ed['u']))
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
    stat_dict = dict(zip(range(5),\
    ['Optimal', 'Iteration Limit', 'Infeasible', 'Unbounded', 'Numerical Issues']))
    print("Problem Status: ", stat_dict[res['status']])
    if res['status'] != 0:
        print("Problem did not solve. Status: ", stat_dict[res['status']])
    return res['status'], res['fun']

def validate_soln(name, val, shift):
    if val == None:
        print("ERROR. Problem unexpectedly did not solve")
    else:
        optval = obj_vals[name]
        rel_err = abs(optval - (val+shift))/abs(optval)
        if rel_err > 1e-5:
            print(f"ERROR. Problem {name} has high relative error of {rel_err}")
            print('Optimal Objective Val: ', optval)
            print('Our Objective Val: ', val + shift)
        else:
            print("SUCCESS. Our objective value agrees with literature.")
        assert rel_err <= 1e-5, f"ERROR. Problem {name} has high relative error of {rel_err}"

if __name__ == '__main__':
    this_dir = pathlib.Path(__file__).parent.resolve()
    for prob_name in os.listdir(this_dir/"mps_problems"):
        print('=============================================')
        print("parsing", prob_name)
        #if prob_name != "forplan_highs_export.mps":
        #    continue
        prob_data = parse_mps_file(this_dir/"mps_problems"/prob_name)
        shift = prob_data['obj_shift']
        matrix_data = extract_matrix_data(prob_data)
        status, val = solve_from_matrix_data(matrix_data, shift)
        validate_soln(prob_name, val, matrix_data['obj_shift'])
        if (matrix_data['l']==matrix_data['u']).sum() > 0:
            print("There are fixed variables. Eliminating them...")
            reduced_prob = eliminate_fixed_variables(matrix_data)
            assert (reduced_prob['l']==reduced_prob['u']).sum() == 0
            status, val = solve_from_matrix_data(reduced_prob, shift)
            validate_soln(prob_name, val, reduced_prob['obj_shift'])


