X-ray Single Photon Energy Dispersive Program

This program was built upon data with two clear peaks due to the L-alpha and L-beta emission lines of Germanium.

All saved variables are put within a folder "stored_variables" that is within the program 
repository as shown on GitHub. 
The general structure is as follows:
- Folder: "Index of Interest" - Following the indices of a python string s.t. 1 denotes the 2nd in the list
  - File: "quadratic_fits.npy" - Array with [ALeft, BLeft, cLeft, cLeft_Peak_unc,ARight, BRight, cRight, cRight_Peak_unc] 
  denoting the paramaters of the left and right curves associated with the L-beta and L-alpha emission respectively. The 
  curves are parameterised by the quadratic x = A(y-B)**2 + C
  - File: "quadratic_fits_log.txt" - Text file describing the results and inputs of the optimisation
  - File: "geometric_fits.npy"
  - File: "quadratic_fits_log.txt"


For photos and notes made in the process of making this see https://miro.com/welcomeonboard/QUM2cCtDaDhjRmtNQmw0dXVzSkFZNm5XYUZRcEh6WVkrbDRnMVN3VkN3eTVtZmUrSTZSQ2k3M1EzNFVYZ2djdDg0WEtjVlpPNnE5aWJ2OTczaUtWUFBDL3J6NTl6NFpXaXY2SG1nTndrK29BK0NLTWVmT1JOelJvOGdmcENYTVl0R2lncW1vRmFBVnlLcVJzTmdFdlNRPT0hdjE=?share_link_id=640928823258

Important Human Inputs:
- The bounds of the c values for each curve

Run Times:
- calibrate_and_save_quadratics ~ 
- Solid Angle per Bin using bin_width=1,numberOfPoints=3 is ~ 12 minutes