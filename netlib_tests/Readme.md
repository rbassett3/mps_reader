# Loading the Netlib problems in Python

**Note**: The download and decompress script has only has been tested on Linux. It may run on MacOS but will definitely not run on Windows. 

1. Run `bash download_netlib_and_decompress.sh` to download and decompress the netlib lp files. The downloaded content will be in the directory `netlib.org` and the ready-to-use MPS files will be in `mps_problems`.

2. Run `python read_mps_files.py` to loop through the problems in the `mps_problems` directory and load them into Python.
