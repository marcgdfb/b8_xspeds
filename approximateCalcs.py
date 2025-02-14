from constants import *
from tools import bragg_E_to_theta

# Given that the lines are of order 150 pixels apart

d_alphaBeta = 150*pixel_width
angleDif = bragg_E_to_theta(E_Lalpha_eV)-bragg_E_to_theta(E_Lbeta_eV)  # 0.008647037282303704 rad

# Assuming the crystal is normal we have:
rApprox = (d_alphaBeta/2) / np.tan(angleDif/2)    # 0.2341827976709863 m

thetaSpherical_Emax = np.pi/2 + bragg_E_to_theta(E_max_eV)  # 1.8160099328197494
thetaSpherical_Emin = np.pi/2 + bragg_E_to_theta(E_min_eV)  # 1.9316901438515075
thetaSpherical_Ealpha = np.pi/2 + bragg_E_to_theta(E_Lalpha_eV) # 1.9038752412123663
thetaSpherical_Ebeta = np.pi/2 + bragg_E_to_theta(E_Lbeta_eV) # 1.8952282039300625


# this implies
r_test = np.array([0.2341827976709863,1.85,np.pi])

# We are told that the angle with respect to the norm is approximately 30 degrees
angle_rad = (90+60)*np.pi/180  # 2.6179938779914944
print(angle_rad)
deltaTheta = angle_rad - thetaSpherical_Ealpha  # 0.7141186367791281

print(deltaTheta)
