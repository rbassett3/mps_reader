# MPS Parser for Python

`mps_reader` is a Python script for parsing [MPS (Mathematical Programming System)](https://en.wikipedia.org/wiki/MPS_(format)) files. Its primary purpose is to convert linear programs saved in MPS format to problem data $c, b_{ub}, b_{eq}, l, u, A_{ub}, A_{eq}$ such that the problem

$$\min_{x} \quad c^{T} x$$

$$\text{s.t.} \quad A_{ub} x \leq b_{ub}$$

$$\quad \quad A_{eq} x = b_{eq}$$

$$\quad \quad l \leq x \leq u$$

matches the problem saved in the MPS file. In Python, $c, b_{ub}, b_{eq}, l, u$ are NumPy arrays and $A_{ub}$ and $A_{eq}$ are SciPy sparse arrays.

## How to Use

The primary user-facing function is `read`, which takes a MPS file path as input and returns a dictionary containing $c, b_{ub}, b_{eq}, l, u, A_{ub}, A_{eq}$.

The problem data returned by the `read` function performs some very basic preprocessing by eliminating fixed variables and ignoring any constants added to the objective function. The correspondence between variable/constraint names in the MPS file and indexing in the rows/columns of the constraint matrices is not provided since the primary motivation for this script is benchmarking. For users who prefer to interface directly with the MPS file data directly, the `parse_mps_file` function can be used to return a dictionary containing the rows, columns, rhs, bounds, and ranges provided in the MPS file.

## Test Cases

This project has been validated on all the problems in the [NetLib LP Benchmark](https://netlib.org/lp/data/index.html).

## If You Find This Useful Or Have Questions

Don't hesitate to contact me: robert 'dot' bassett 'at' nps 'dot' edu

