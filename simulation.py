import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tools import *

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

    sns.lineplot(x=range_energy,y=count_energy)
    plt.show()


# generate_spectrum()

class Simulation:
    def __init__(self):
        self.photon_hit = None



