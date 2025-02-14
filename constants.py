import numpy as np

d_beryl = (15.96 / 2) * (10 ** -10)
q_e = 1.602176634 * (10 ** -19)
E_min_eV = 1100  # gives an angle of 0.360893817056611
E_max_eV = 1600  # gives an angle of 0.24521360602485284
h_planck = 6.62607015 * (10 ** -34)
h_bar = h_planck / (2 * np.pi)
c = 299792458

# For this experiment Germanium (Z=32) plasmas were investigated
E_Lalpha_eV = 1188  # gives an angle of 0.3330789144174696  # Right Hand Side
E_Lbeta_eV = 1218.5  # gives an angle of 0.3244318771351659  # Left Hand Side

# Similarly a Princeton Instruments PI-MTE 2048B CCD camera
# was used. The Manual can be accessed at https://usermanual.wiki/Princeton/4411-0097.1370993303.pdf
# The relevant values used are:

pixel_width = 13.5 * (10 ** (-6))
length_detector = 2048 * pixel_width  # 0.027648

geometryDict = {
    "crysPitch": -0.3445092569785031,
    "CrysRoll": 0.018058061625517555,
    "CamPitch": 0.7965766171590669,
    "CamRoll": -0.005201895746676618,
    "rcam": 0.08378234,
    "thetacam": 2.567,
}
