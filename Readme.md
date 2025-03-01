# MPS_ Reader

`mps_reader` is a Python package for parsing [MPS (Mathematical Programming System)](https://en.wikipedia.org/wiki/MPS_(format) files. The primary purpose of the package is to convert linear programs saved in MPS format to problem data $c, b_ub, b_eq, l, u, A_ub, A_eq$ such that the problem
$$\min_{x} \quad c^{T} x$$
$$\text{s.t.} \quad A_ub x \leq b_ub$$
$$\qquad $A_eq x = b_eq$$
$$l \leq x \leq u$$
matches the problem saved in the MPS file. In Python, $c, b_ub, b_eq, l, u$ are NumPy arrays and $A_ub$ and $A_eq$ are SciPy sparse arrays.

