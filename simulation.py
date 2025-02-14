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
                      energy_peaks_eV=None, peak_width=5, peak_height=100,
                      plotIm=False):
    """
    The Purpose of this function is to create a spectrum of
    X-Ray energies between energy_start_eV and energy_endeV with peaks at
    specific energies (given in eV)

    energy_peaks_eV: list of energies at which there are known peaks
    peak_width,peak_height: parameters of the Gaussian peaks which are then added on top of the noise

    """
    if energy_peaks_eV is None:
        energy_peaks_eV = [E_Lalpha_eV, E_Lbeta_eV]
    # Generate energy range
    range_energy = np.arange(energy_start_eV, energy_end_eV, step)
    count_energy = np.random.uniform(30, 50, size=len(range_energy))  # Baseline noise

    # Add Gaussian peaks
    for peak in energy_peaks_eV:
        gaussian_peak = peak_height * np.exp(-((range_energy - peak) ** 2) / (2 * peak_width ** 2))
        count_energy += gaussian_peak

    # Create DataFrame for plotting
    df_spectrum = pd.DataFrame({'Energy': range_energy, 'Count': count_energy})

    if plotIm:
        # Plot
        sns.lineplot(data=df_spectrum, x="Energy", y="Count")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Intensity")
        plt.title("Smooth X-ray Spectrum")
        plt.show()

    return df_spectrum


# generate_spectrum()


class Simulation:
    def __init__(self, df_spectrum, relativeCount, experimental_geometry):
        self.df = df_spectrum
        self.relativeCount = relativeCount
        self.experimental_geometry = experimental_geometry

    def simulateHits(self):
        
        energyVals = self.df["Energy"]
        countVals = self.df["Count"]
        maxCount = np.max(countVals)
        # normalising for max count and rounding to the nearest integer
        countVals = (countVals / maxCount * self.relativeCount).astype(int)

        """Now to iterate for each energy, given the geometrical setup find the polar
        theta coordinate of the rays and find the resulting range of phi values that hit the detector
        and account for solid angle normalisation."""

        n_crystal, n_camera, r_camera = self.experimental_geometry[0], self.experimental_geometry[1], self.experimental_geometry[2]

        rotMatrix_cry = inverseRotation_matrix(n_crystal)

        for energyVal,countVal in zip(energyVals, countVals):

            theta = bragg_E_to_theta(energyVal)




    class Bragg:

        @staticmethod
        def randomPixel(energy_eV,n_crystal, n_camera, r_camera):
            """
            For a given energy there is a range of x y pixels that are allowed
            """
            rotMatrix_cry = inverseRotation_matrix(n_crystal)
            theta = bragg_E_to_theta(energy_eV)


Simulation(generate_spectrum(plotIm=True), 100, "Bragg").simulateHits()