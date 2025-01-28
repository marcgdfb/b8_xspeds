import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tools import *

"""
Spectrum produced by plasma
-> deflection probability on crystal?
-> solid angle contributions
-> Fano factor + absorbtion probability in silicon / CCD
"""

def generate_spectrum(energy_start_eV=1090,energy_end_eV=1610,step=0.5,
                      energy_peaks_eV=None):
    """
    The Purpose of this function is to create a spectrum of
    X-Ray energies between energy_start_eV and energy_endeV with peaks at
    specific energies (given in eV)
    """
    if energy_peaks_eV is None:
        energy_peaks_eV = [E_Lalpha_eV, E_Lbeta_eV]

    range_energy = np.arange(start=energy_start_eV,stop=energy_end_eV,step=step)

    count_energy = []

    for energy in range_energy:
        nearPeak = False
        for peak in energy_peaks_eV:
            if abs(energy - peak) < 5:
                nearPeak = True

        if nearPeak:
            if energy in energy_peaks_eV:
                count_energy.append(100)
            else:
                count_energy.append(np.random.randint(50, 70))
        else:
            count_energy.append(np.random.randint(40, 50))

    df_spectrum = pd.DataFrame({'Energy': range_energy, 'Count': count_energy})

    sns.lineplot(data=df_spectrum, x="Energy",y="Count")
    plt.show()




# generate_spectrum()

class Simulation:
    def __init__(self,df_spectrum):
        self.df = df_spectrum
        self.photon_hit = None





    class Bragg:

        @staticmethod
        def randomPixel(energy_eV):
            """
            For a given energy there is a range of x y pixels that are allowed
            """

            theta = bragg_E_to_theta(energy_eV)
            phi_half = phiHalf(energy_eV)

            phi_val = random.uniform(-phi_half,phi_half)

            arrPixels = theta_phi_to_xy_observation(theta=theta,phi=phi_val)
            yPixel = arrPixels[0]
            xPixel = arrPixels[1]

            return xPixel,yPixel





phi_half = phiHalf(E_Lalpha_eV)

print(phi_half)





