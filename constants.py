import numpy as np

def d_beryl_(a_beryl=9.21*10**-10,c_beryl=9.19*10**-10,miller_indices=None,):
    if miller_indices is None:
        miller_indices = [1,0,0]
    h_idx = miller_indices[0]
    k_idx = miller_indices[1]
    l_idx = miller_indices[2]

    return ((4/3)*(h_idx**2+k_idx**2+l_idx**2)/(a_beryl**2) + l_idx**2/c_beryl**2)**(-1/2)

d_beryl = (15.96 / 2) * (10 ** -10)
# d_beryl = d_beryl_()
q_e = 1.602176634 * (10 ** -19)
E_min_eV = 1100   # 0.78414658211633
E_max_eV = 1600   # 0.5069659949953977
h_planck = 6.62607015 * (10 ** -34)
h_bar = h_planck / (2 * np.pi)
c = 299792458

# For this experiment Germanium (Z=32) plasmas were investigated
E_Lalpha_eV = 1188  # gives an angle of 0.7127390872863559  # Right Hand Side
E_Lbeta_eV = 1218.5  # gives an angle of 0.6913018700701734  # Left Hand Side

# Similarly a Princeton Instruments PI-MTE 2048B CCD camera
# was used. The Manual can be accessed at https://usermanual.wiki/Princeton/4411-0097.1370993303.pdf
# The relevant values used are:

pixel_width = 13.5 * (10 ** (-6))
length_detector_pixels = 2048
length_detector = 2048 * pixel_width  # 0.027648

