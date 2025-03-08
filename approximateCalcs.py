from constants import *
from tools import bragg_E_to_theta

# Given that the lines are of order 150 pixels apart

d_alphaBeta = 150*pixel_width
angleDif = bragg_E_to_theta(E_Lalpha_eV)-bragg_E_to_theta(E_Lbeta_eV)  # 0.021437217216182414 rad
# print("HI",angleDif)
approxdist = (d_alphaBeta/2) / np.tan(angleDif/2)  # 0.09445826989579484
# print(approxdist)

thetaSpherical_Emax = np.pi/2 + bragg_E_to_theta(E_max_eV)  # 2.0777623217902943
print(thetaSpherical_Emax- np.pi/2)
thetaSpherical_Emin = np.pi/2 + bragg_E_to_theta(E_min_eV)  # 2.3549429089112266
print(thetaSpherical_Emin- np.pi/2)
thetaSpherical_Ealpha = np.pi/2 + bragg_E_to_theta(E_Lalpha_eV)  # 2.2835354140812525 , 0.7127390872863559
# print(thetaSpherical_Ealpha - np.pi/2)
thetaSpherical_Ebeta = np.pi/2 + bragg_E_to_theta(E_Lbeta_eV)  # 2.26209819686507
# print(thetaSpherical_Ebeta- np.pi/2)

print(bragg_E_to_theta(1050))
print(bragg_E_to_theta(1650))

# We are told that the angle with respect to the norm is approximately 30 degrees
angle_rad = (90+60)*np.pi/180  # 2.6179938779914944
# print(angle_rad)
deltaTheta = angle_rad - thetaSpherical_Ealpha  # 0.3344584639102419
# print(deltaTheta)
