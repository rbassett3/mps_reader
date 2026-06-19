# Loading the Netlib problems in Python

**Note**: The download and decompress script has only has been tested on Linux. It may run on MacOS but will definitely not run on Windows. 

1. Run `bash download_netlib_and_decompress.sh` to download and decompress the netlib lp files. The downloaded content will be in the directory `netlib.org` and the ready-to-use MPS files will be in `mps_problems`.

2. Run `python read_mps_files.py` to loop through the problems in the `mps_problems` directory and load them into Python.

## Netlib problems currently failing for unknown reasons.

- `80bau3b`. HiGHS says: `WARNING: Problem has some excessively small column bounds`

- `forplan`. HiGHS says: `WARNING: Problem has some excessively large column bounds. WARNING: Problem has some excessively large row bounds. WARNING:    Consider scaling the    bounds by 1e-1, or setting the user_bound_scale option to -4`

- `bnl2`. No warnings.

- `czprob`. No warnings.

To help diagnose these I loaded them and re-exported them to mps format using HiGHS. This can be done using the `--write-model-file` option on the command line. It gave me various warnings for each file, which I flagged above.
