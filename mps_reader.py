import numpy as np
import scipy.sparse


def read(path_to_mps_file):
    '''
    read takes path to an mps file as input and returns
    a dictionary containing the vectors c, b, b_ub, b_eq, l, and u
    and sparse matrices A_ub and C_eq such that the problem
    min c.T @ x
    s.t. A_eq @ x = b_eq
         A_ub @ x <= b_ub
         l <= x <= u
    defines the optimization problem in the mps file.
    '''
    #this first line contains all the problem data from the mps file
    #including things like variable names. The prob_data just gets the matrices
    #and vectors needs
    rows, columns, rhs, bounds, prob_name = parse_mps_file(path_to_mps_file)
    prob_data = construct_vecs_and_mats(rows, columns, rhs, bounds)
    return prob_data 

def construct_vecs_and_mats(rows, columns, rhs, bounds):
    #now we actually build the vectors and matrices
    col_to_ind = dict(zip(columns.keys(), range(len(columns.keys())))) 
    n = len(col_to_ind)
    rows_in_objs = [row for row in rows.keys() if rows[row]=='N']
    assert len(rows_in_objs) == 1, "More than 1 objective specified"
    rows_in_eq_ind = [row for row in rows.keys() if rows[row]=='E']
    rows_in_ub_ind = [row for row in rows.keys() if rows[row] in ['L', 'G']]
    m1 = len(rows_in_eq_ind)
    m2 = len(rows_in_ub_ind)
    row_to_eq_ind = dict(zip(rows_in_eq_ind, range(m1)))
    row_to_ub_ind = dict(zip(rows_in_ub_ind, range(m2)))

    c = np.zeros(n)
    l = np.zeros(n)
    u = np.inf*np.ones(n)
    b_eq = np.zeros(m1)
    b_ub = np.zeros(m2)
    A_eq = scipy.sparse.dok_matrix((m1, n), dtype=np.float64)
    A_ub = scipy.sparse.dok_matrix((m2, n), dtype=np.float64)

    #loop through column section to build c vector and A matrices
    for column in columns.keys():
        rows_for_this_column = columns[column]
        for (row, value) in rows_for_this_column:
            col_ind = col_to_ind[column]
            if rows[row]=='N': #objective
                c[col_ind] = value
            elif rows[row]=='L': #lower bound. negate b/c we only keep track of A_ub @ x <= b_ub
                A_ub[row_to_ub_ind[row], col_ind] = -1.0*float(value)
            elif rows[row]=='G': #upper bound. don't have to negate
                A_ub[row_to_ub_ind[row], col_ind] = float(value)
            elif rows[row]=='E': #equality constraint
                A_eq[row_to_eq_ind[row], col_ind] = float(value)
            else:
                raise ValueError("Row kind " + rows[row] + " not recognized")
    #loop through rhs to build b vectors
    for this_rhs_name in rhs.keys():
        for (row, value) in rhs[this_rhs_name]:
            if rows[row]=='L': #lower bound. negate b/c we only keep track of A_ub @ x <= b_ub
                b_ub[row_to_ub_ind[row]] = -1.0*float(value)
            elif rows[row]=='G': #upper bound. don't have to negate
                b_ub[row_to_ub_ind[row]] = float(value)
            elif rows[row]=='E': #equality constraint
                b_eq[row_to_eq_ind[row]] = float(value)
            else:
                raise ValueError("Row kind " + rows[row] + " not recognized")
    #loop through bounds to build l and u vectors
    for bnd in bounds.keys():
        for (kind, column, value) in bounds[bnd]:
            if kind == 'UP':
                u[col_to_ind[column]] = value
            elif kind == 'LO':
                l[col_to_ind[column]] = value
            elif kind == 'FX': #why do these variables even exist?
                #they should be added to the b terms instead
                l[col_to_ind[column]] = value
                u[col_to_ind[column]] = value
            else:
                raise ValueError("Bound kind " + kind + " not recognized")
    #convert the completed matrices from dok to csr
    A_eq = scipy.sparse.csr_matrix(A_eq)
    A_ub = scipy.sparse.csr_matrix(A_ub)
    return {'c':c, 'A_eq':A_eq, 'A_ub':A_ub, 'b_eq':b_eq, 'b_ub':b_ub, 'l':l, 'u':u}

def parse_mps_file(path_to_mps_file):
    '''takes mps file path as input and returns four dictionaries and the name of 
    the problem. The four dictionaries are:
    1. rows: has row names as keys and kind (N, L, G, E) as values
    2. columns: has columns as keys and lists of (row name, value) pairs as values
    3. rhs: has rhs name as key and list of of (row name, value) pairs as values
    4. bounds: has bound name as key and list of (kind, column, value) tuples as values
    '''
    flags = {'in_rows':False, 'in_columns':False, 'in_bounds':False, 'in_rhs':False}
    rows = {}
    columns = {}
    rhs = {}
    bounds = {}
    this_col = ''
    this_rhs = ''
    this_bnd = ''

    with open(path_to_mps_file, 'r') as f:
        for line in f:
            if line[0] != ' ':
                reset_flags_to_false(flags)
                sec_name = line.split()[0]
                #print("In section ", sec_name) 
                if sec_name == "NAME":
                    prob_name = line.split()[1]
                elif sec_name == "ROWS":
                   flags['in_rows'] = True 
                elif sec_name == "COLUMNS":
                   flags['in_columns'] = True 
                elif sec_name == "RHS":
                   flags['in_rhs'] = True 
                elif sec_name == "BOUNDS":
                   flags['in_bounds'] = True 
                elif sec_name == "ENDATA":
                    break #end of file
                else:
                    raise ValueError("MPS file has unrecognized section " + sec_name)
            elif flags['in_rows']:
                kind, name = line.split()
                rows[name] = kind
            elif flags['in_columns']:
                col_data = line.split()
                if col_data[0] != this_col: #column name hasn't been seen. We need to add it
                    #this works b/c all entries associated with a column must occur consecutively
                    #I use the same logic to detect new rhs and bounds.
                    this_col = col_data[0]
                    #columns[col] = [(row_name, value_in_row), ...] for all the rows
                    columns[this_col] = [(col_data[1], col_data[2])]
                    for i in range(3, len(col_data), 2):
                        columns[this_col] += [(col_data[i], col_data[i+1])]
                else:  #column has been seen. We need to add new values to existing list
                    for i in range(1, len(col_data), 2):
                        columns[this_col] += [(col_data[i], col_data[i+1])]
            elif flags['in_rhs']:
                rhs_data = line.split()
                if rhs_data[0] != this_rhs: #new rhs. need to add it to dictionary
                    this_rhs = rhs_data[0]
                    #rhs[rhs] = [(row_name, value_of_rhs), ...] for all the rows
                    rhs[this_rhs] = [(rhs_data[1], rhs_data[2])]
                    for i in range(3, len(rhs_data), 2):
                        rhs[this_rhs] += [(rhs_data[i], rhs_data[i+1])]
                else: #rhs we have seen. need to add its values to existing list
                    for i in range(1, len(rhs_data), 2):
                        rhs[this_rhs] += [(rhs_data[i], rhs_data[i+1])]
            elif flags['in_bounds']:
                bnd_data = line.split()
                if bnd_data[1] != this_bnd: #new bound. Need to add it to list
                    this_bnd = bnd_data[1]
                    #bounds[bnd] = [(kind, column, value), ...] for all the bounds
                    if bnd_data[0] == 'FR': #unconstrained variable
                        #since variables are assumed to be in (0, np.inf) if
                        #a bound is not provided we only need to change the lower bound
                        bounds[this_bnd] = [('LO', bnd_data[2], -np.inf)]
                    else: #LO, UP, or FX variable
                        bounds[this_bnd] = [(bnd_data[0], bnd_data[2], bnd_data[3])]
                else: #bound we have seen. need to add its values to existing list
                    if bnd_data[0] == 'FR': #unconstrained variable
                        #since variables are assumed to be in (0, np.inf) if
                        #a bound is not provided we only need to change the lower bound
                        bounds[this_bnd] += [('LO', bnd_data[2], -np.inf)]
                    else: #LO, UP, or FX variable
                        bounds[this_bnd] += [(bnd_data[0], bnd_data[2], bnd_data[3])]
    return rows, columns, rhs, bounds, prob_name

    
def reset_flags_to_false(flags):
    for key in flags.keys():
        flags[key] = False
     
    
