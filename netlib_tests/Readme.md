# Loading the Netlib problems in Python

**Note**: The download and decompress script has only has been tested on Linux. It may run on MacOS but will definitely not run on Windows. 

1. Run `bash download_netlib_and_decompress.sh` to download and decompress the netlib lp files. The downloaded content will be in the directory `netlib.org` and the ready-to-use MPS files will be in `mps_problems`.

2. Install `mps_reader`.

3. Run `python verify_netlib_solns.py` to loop through the problems in the `mps_problems` directory and verify the optimal objective values in the literature match our problem formulations as read from the MPS files.

