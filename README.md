X-ray Single Photon Energy Dispersive Spectroscopy Program

This program was built upon data with two clear peaks due to the L-$\alpha$ and L-$\beta$ emission lines of Germanium.

All saved variables are put within a folder "stored_variables" that is within the program 
repository as shown on GitHub. 
The general structure is as follows:
- Folder: "Index of Interest" - Following the indices of a python 
string s.t. 1 denotes the 2nd in the list
  - File: "ellipse_fits.npy" - Array with parameters definining the 
  ellipse created by the L-$\alpha$ and L-$\beta$ lines
  denoting the paramaters of the left and right curves associated with 
  the L-beta and L-alpha emission respectively. The 
  curves are parameterised by the quadratic $x = C+A-A \sqrt{1- \frac{(y-y_0)^2}{B^2}}$
  - File: "geometric_fits.npy" - Array with the geometry parameters (crystal pitch, crystal roll,
  camera pitch, cameral roll and $\left|\vec{r}_\textbf{cam}\right|$)
  - File: "solid_angle_of_pixel.npy" - Matrix with the values of solid angle associated
  with the pixel of the same index of the CCD
  - File: "energy_of_pixel.npy" - Matrix with the values of the energy of the photon
  that hits that pixel on the CCD
  - Folder: "Spectrum":
    - File: "energy_list.npy" - Array with all energies found for that image following 
    the use of the spectrum engine i.e. [1000,1000,1001] where each number is associated with 
    a photon and its energy. 
    - FIle: "solid_angle_of_bin_width_().xlsx" where () is replaced by the relevant bin width. Excel
    file containing the solid angle associated with that energy bin


For photos and notes made in the process of making this see https://miro.com/welcomeonboard/QUM2cCtDaDhjRmtNQmw0dXVzSkFZNm5XYUZRcEh6WVkrbDRnMVN3VkN3eTVtZmUrSTZSQ2k3M1EzNFVYZ2djdDg0WEtjVlpPNnE5aWJ2OTczaUtWUFBDL3J6NTl6NFpXaXY2SG1nTndrK29BK0NLTWVmT1JOelJvOGdmcENYTVl0R2lncW1vRmFBVnlLcVJzTmdFdlNRPT0hdjE=?share_link_id=640928823258

Important Human Inputs:
- The bounds of the parameters of the ellipse are quite important 
due to the noise of the image. Although the geometric fitting is relatively
robust to error here, I recommend using the quadratic fit to get a quick curve
and then using desmos to fine tune the bounds on the ellipse https://www.desmos.com/calculator/xa8ud5juop
- The relevant emission lines from the XSPEDS experiment
- The approximate distances and theta polar coordinate of the image camera relative to the source.
The bounds of the geometrical parameters.