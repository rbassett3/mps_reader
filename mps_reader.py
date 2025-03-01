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
    rows, columns, rhs, ranges, bounds, obj_shift, prob_name = parse_mps_file(path_to_mps_file)
    prob_data = construct_vecs_and_mats(rows, columns, rhs, ranges, bounds)
    return prob_data 

def construct_vecs_and_mats(rows, columns, rhs, ranges, bounds):
    #now we actually build the vectors and matrices
    col_to_ind = dict(zip(columns.keys(), range(len(columns.keys())))) 
    n = len(col_to_ind)
    rows_in_objs = [row for row in rows.keys() if rows[row]=='N']
    assert len(rows_in_objs) == 1, "More than 1 objective specified"
    #number of inequality constraints is number of rows w/ type E
    rows_in_eq_ind = [row for row in rows.keys() if rows[row]=='E']
    #number of inequality constraints is number of rows w/ type L or G + 
    #number of additional L and G constraints implied by RANGE
    #the range constraints have an emoji added to their labels
    #which allows referencing them later as unique rows
    #by adding an emoji we guarantee no conflict with another
    #row name since mps only permits ascii characters
    #surprisingly, range on a E row yields two inequality constraints
    rows_in_ub_ind = [row for row in rows.keys() if rows[row] in ['L','G']] +\
        [row+"\U0001f600" for range_ in ranges.values()\
            for (row, value) in range_ if rows[row] in ['L','G']]+\
        [row+"\U0001f600" for range_ in ranges.values()\
            for (row, value) in range_ if rows[row]=='E']+\
        [row+"\U0001f606" for range_ in ranges.values()\
            for (row, value) in range_ if rows[row]=='E']
    m1 = len(rows_in_eq_ind)
    m2 = len(rows_in_ub_ind)
    row_to_eq_ind = dict(zip(rows_in_eq_ind, range(m1)))
    row_to_ub_ind = dict(zip(rows_in_ub_ind, range(m2)))
    #number of fixed variables. We'll treat these separately
    num_fixed = len([kind for bnds in bounds.values() for (kind, col, val) in bnds if kind=='FX'])
    fixed_inds = np.empty(num_fixed, dtype=np.int64)
    fixed_vals = np.empty(num_fixed, dtype=np.float64)
    fixed_itr = 0 #how many we've seen as we loop through them later

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
    for range_ in ranges.keys():
        for (row, value) in ranges[range_]:
            #The range section has a different meaning depending on whether
            #the row it references is of kind G, L, or E. What I do here
            #is in page 164 of Advanced Linear Programming by Murtagh
            #note that per the mps rules ROWS must come before RANGE (if RANGE exists)
            if rows[row]=='L': #Range adds an upper bound of b[i] + abs(r[i])
                A_ub[row_to_ub_ind[row+"\U0001f600"],:] = A_ub[row_to_ub_ind[row],:]
                b_ub[row_to_ub_ind[row+"\U0001f600"]] = b_ub[row_to_ub_ind[row]]+abs(float(value))
            elif rows[row]=='G': #Range adds a lower bound of b[i] - abs(r[i])
                A_ub[row_to_ub_ind[row+"\U0001f600"],:] = -A_ub[row_to_ub_ind[row],:]
                b_ub[row_to_ub_ind[row+"\U0001f600"]] = -1.*(b_ub[row_to_ub_ind[row]]-abs(float(value)))
            elif rows[row]=='E': #equality constraint. Adds an upper and a lower bound
                #where value depends on the sign
                sign_val = float(value) >= 0
                if sign_val: #(b, b+|r|) constraint
                    A_ub[row_to_ub_ind[row+"\U0001f600"],:] = -A_ub[row_to_ub_ind[row],:]
                    b_ub[row_to_ub_ind[row+"\U0001f600"]] = -1.*b_ub[row_to_ub_ind[row]]
                    A_ub[row_to_ub_ind[row+"\U0001f606"],:] = A_ub[row_to_ub_ind[row],:]
                    b_ub[row_to_ub_ind[row+"\U0001f606"]] = b_ub[row_to_ub_ind[row]]+abs(float(value))
                else: #(b-|r|, b) constraint
                    A_ub[row_to_ub_ind[row+"\U0001f600"],:] = -A_ub[row_to_ub_ind[row],:]
                    b_ub[row_to_ub_ind[row+"\U0001f600"]] = -1.*(b_ub[row_to_ub_ind[row]]-\
                                                                abs(float(value)))
                    A_ub[row_to_ub_ind[row+"\U0001f606"],:] = A_ub[row_to_ub_ind[row],:]
                    b_ub[row_to_ub_ind[row+"\U0001f606"]] = b_ub[row_to_ub_ind[row]]
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
                fixed_inds[fixed_itr] = col_to_ind[column]
                fixed_vals[fixed_itr] = float(value)
                fixed_itr += 1
            elif kind == 'MI': #x \in (-\infty, 0)
                u[col_to_ind[column]] = 0.0
                l[col_to_ind[column]] = -np.inf
            elif kind == "PL": #x \in (0, \infty)
                continue #this is the default bound. Nothing to do
            else:
                raise ValueError("Bound kind " + kind + " not recognized")
    #convert the completed matrices from dok to csr
    A_eq = scipy.sparse.csr_matrix(A_eq)
    A_ub = scipy.sparse.csr_matrix(A_ub)
    return {'c':c, 'A_eq':A_eq, 'A_ub':A_ub, 'b_eq':b_eq, 'b_ub':b_ub, 'l':l, 'u':u,\
            'fixed_inds':fixed_inds, 'fixed_vals':fixed_vals}

def parse_mps_file(path_to_mps_file):
    '''takes mps file path as input and returns five dictionaries, the constant
    shift of the objective, and the name of the problem. The five dictionaries are:
    1. rows: has row names as keys and kind (N, L, G, E) as values
    2. columns: has columns as keys and lists of (row name, value) pairs as values
    3. rhs: has rhs name as key and list of of (row name, value) pairs as values
    4. ranges: has range name as key and list of of (row name, value) pairs as values
    5. bounds: has bound name as key and list of (kind, column, value) tuples as values
    '''
    flags = {'in_rows':False, 'in_columns':False, 'in_bounds':False, 'in_rhs':False,\
        'in_ranges':True}
    rows = {}
    columns = {}
    rhs = {}
    bounds = {}
    ranges = {}
    this_col = ' '
    this_rhs = ' '
    this_bnd = ' '
    this_range = ' '
    obj_shift = 0.

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
                elif sec_name == "RANGES":
                   flags['in_ranges'] = True
                elif sec_name == "BOUNDS":
                   flags['in_bounds'] = True 
                elif sec_name == "ENDATA":
                    break #end of file
                else:
                    raise ValueError("MPS file has unrecognized section " + sec_name)
            elif flags['in_rows']:
                kind, name, _, _, _, _ = get_fields(line)
                rows[name] = kind
            elif flags['in_columns']:
                col_data = get_fields(line)[1:] #first entry of column line is empty
                if col_data[0] != this_col: #column name hasn't been seen. We need to add it
                    #this works b/c all entries associated with a column must occur consecutively
                    #I use the same logic to detect new rhs and bounds.
                    this_col = col_data[0]
                    #columns[col] = [(row_name, value_in_row), ...] for all the rows
                    columns[this_col] = [(col_data[1], col_data[2])]
                    for i in range(3, len(col_data), 2):
                        if col_data[i+1] != '':
                            columns[this_col] += [(col_data[i], col_data[i+1])]
                else:  #column has been seen. We need to add new values to existing list
                    for i in range(1, len(col_data), 2):
                        if col_data[i+1] != '':
                            columns[this_col] += [(col_data[i], col_data[i+1])]
            elif flags['in_rhs']:
                rhs_data = get_fields(line)[1:] #first entry of rhs line is empty
                if rhs_data[0] != this_rhs: #new rhs. need to add it to dictionary
                    this_rhs = rhs_data[0]
                    if rows[rhs_data[1]] == 'N': #this is a constant shift of the objective
                        obj_shift = float(rhs_data[2])
                        rhs[this_rhs] = []
                    else:
                        #rhs[rhs] = [(row_name, value_of_rhs), ...] for all the rows
                        rhs[this_rhs] = [(rhs_data[1], rhs_data[2])]
                    for i in range(3, len(rhs_data), 2):
                        if rhs_data[i] != '':
                            if rows[rhs_data[i]] == 'N': #this is a constant shift of the objective
                                obj_shift = float(rhs_data[i+1])
                            else:
                                rhs[this_rhs] += [(rhs_data[i], rhs_data[i+1])]
                else: #rhs we have seen. need to add its values to existing list
                    for i in range(1, len(rhs_data), 2):
                        if rhs_data[i] != '':
                            rhs[this_rhs] += [(rhs_data[i], rhs_data[i+1])]
            elif flags['in_ranges']:
                range_data = get_fields(line)[1:]
                if range_data[0] != this_range:
                    this_range = range_data[0]
                    ranges[this_range] = [(range_data[1], range_data[2])]
                    for i in range(3, len(range_data), 2):
                        if range_data[i] != '':
                            ranges[this_range] += [(range_data[i], range_data[i+1])]
                else: 
                    for i in range(1, len(range_data), 2):
                        if range_data[i] != '':
                            ranges[this_range] += [(range_data[i], range_data[i+1])]
            elif flags['in_bounds']:
                bnd_data = get_fields(line)
                if bnd_data[1] != this_bnd: #new bound. Need to add it to list
                    this_bnd = bnd_data[1]
                    #bounds[bnd] = [(kind, column, value), ...] for all the bounds
                    if bnd_data[0] == 'FR': #unconstrained variable
                        #since variables are assumed to be in (0, np.inf) if
                        #a bound is not provided we only need to change the lower bound
                        bounds[this_bnd] = [('LO', bnd_data[2], -np.inf)]
                    else: #LO, UP, FX, MI, or PL variable
                        bounds[this_bnd] = [(bnd_data[0], bnd_data[2], bnd_data[3])]
                else: #bound we have seen. need to add its values to existing list
                    if bnd_data[0] == 'FR': #unconstrained variable
                        #since in MPS format variables are assumed to be in (0, np.inf) if
                        #a bound is not provided we only need to change the lower bound
                        bounds[this_bnd] += [('LO', bnd_data[2], -np.inf)]
                    else: #LO, UP, FX, MI, or PL variable
                        bounds[this_bnd] += [(bnd_data[0], bnd_data[2], bnd_data[3])]
    return rows, columns, rhs, ranges, bounds, obj_shift, prob_name

def eliminate_fixed_from_prob_data(prob_data):
    prob_data['b_eq'] -= prob_data['A_eq'][:, prob_data['fixed_inds']] @ prob_data['fixed_vals']
    prob_data['b_ub'] -= prob_data['A_ub'][:, prob_data['fixed_inds']] @ prob_data['fixed_vals']
    inds_to_keep = np.ones(prob_data['c'].shape[0], dtype=np.bool_)
    inds_to_keep[prob_data['fixed_inds']] = False
    prob_data['A_eq'] = prob_data['A_eq'][:,inds_to_keep]
    prob_data['A_ub'] = prob_data['A_ub'][:,inds_to_keep]
    prob_data['c'] = prob_data['c'][inds_to_keep]
    prob_data['l'] = prob_data['l'][inds_to_keep]
    prob_data['u'] = prob_data['u'][inds_to_keep]
    #make these arrays empty b/c we've eliminated the fixed variables
    prob_data['fixed_inds'] = np.empty(0, dtype=np.int64)
    prob_data['fixed_vals'] = np.empty(0, dtype=np.float64)

def get_fields(line):
    field1 = line[1:3].strip()
    field2 = line[4:12].strip()
    field3 = line[14:22].strip()
    field4 = line[24:36].strip()
    field5 = line[39:47].strip()
    field6 = line[49:61].strip()
    return field1, field2, field3, field4, field5, field6
    
def reset_flags_to_false(flags):
    for key in flags.keys():
        flags[key] = False
     
    
