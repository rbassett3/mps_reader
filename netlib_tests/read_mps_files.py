import os; from mps_reader import read, eliminate_fixed_variables

for prob_name in os.listdir("mps_problems"):
    print("parsing", prob_name)
    prob_data = read("mps_problems/"+prob_name)
    if prob_data['fixed_inds'].shape[0] > 0:
        print("There are fixed variables. Eliminating them...")
        reduced_prob = eliminate_fixed_variables(prob_data)
        assert reduced_prob['fixed_inds'].shape[0] == 0
