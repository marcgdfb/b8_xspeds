from constants import *
from tools import bragg_E_to_theta, bragg_theta_to_E

# These were some rough of order calcul

# Given that the lines are of order 150 pixels apart

d_alphaBeta = 150*pixel_width
angleDif = bragg_E_to_theta(E_Lalpha_eV)-bragg_E_to_theta(E_Lbeta_eV)  # 0.021437217216182414 rad

# print(bragg_E_to_theta(1660)+np.pi/2)
# print(bragg_E_to_theta(1080)+np.pi/2)
approxdist = (d_alphaBeta/2) / np.tan(angleDif/2)  # 0.09445826989579484

angleLeft = bragg_E_to_theta(E_Lalpha_eV) - np.arctan2(1280*pixel_width, approxdist)
ELeft = bragg_theta_to_E(angleLeft)
print("ELeft", ELeft)



thetaSpherical_Ealpha = np.pi/2 + bragg_E_to_theta(E_Lalpha_eV)  # 2.2835354140812525 , 0.7127390872863559
thetaSpherical_Ebeta = np.pi/2 + bragg_E_to_theta(E_Lbeta_eV)  # 2.26209819686507


# We are told that the angle with respect to the norm is approximately 30 degrees
angle_rad = (90+60)*np.pi/180  # 2.6179938779914944
deltaTheta = angle_rad - thetaSpherical_Ealpha  # 0.3344584639102419

