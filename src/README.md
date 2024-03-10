run run.sh to compile and execute the code
First a canny edge detection is performed on the original image.
Then a Distance tranform algorithm is performed where the seeds are the white pixels obtained at the previous step.
The original code is:

Sample: boxFilterNPP
Minimum spec: SM 2.0

A NPP CUDA Sample that demonstrates how to use NPP FilterBox function to perform a Box Filter.

Key concepts:
Performance Strategies
Image Processing
NPP Library