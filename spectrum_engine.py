from calibrateGeometry import *
from spc_engine import *
from collections import Counter

import pandas as pd
import seaborn as sns

class Spectrum:
    def __init__(self,imageMatrix,crystal_pitch, crystal_roll, camera_pitch, camera_roll,
                 r_camera_spherical):
        self.imMat = imageMatrix
        self.crys_pitch = crystal_pitch
        self.crys_roll = crystal_roll
        self.cam_pitch = camera_pitch
        self.cam_roll = camera_roll
        self.r_cam_spherical = r_camera_spherical


    def singlePixelPhotonSpectrum(self, band_width=5,plotSpectrum=False,title='Photon Energy Spectrum with Single Photon Single Pixel Hits',
                                  logarithmic=False):
        spc_engine = PhotonCounting(self.imMat)
        list_countij = spc_engine.checKernelType(kernelType="single_pixel")
        # print(list_countij)

        # Find how many single,double etc photon hits we find
        count_occurrences = Counter(countij[0] for countij in list_countij)
        print(count_occurrences)

        braggEngine = Bragg(crystal_pitch=self.crys_pitch, crystal_roll=self.crys_roll,
                            camera_pitch=self.cam_pitch, camera_roll=self.cam_roll,
                            r_camera_spherical=self.r_cam_spherical)

        energyList = []

        for countij in list_countij:
            count = countij[0]
            iIndex = countij[1]
            jIndex = countij[2]

            yPixel = iIndex
            xPixel = jIndex

            x_0 = - braggEngine.xWidth / 2
            y_0 = + braggEngine.yWidth / 2

            x_coord = xPixel * braggEngine.pixelWidth + x_0  # x_o is such that the x coord for the exact center would be 0
            y_coord = y_0 - yPixel * braggEngine.pixelWidth

            energyVal = braggEngine.xyImagePlane_to_energy(x_coord, y_coord)

            energyList.extend([energyVal] * count)

        if plotSpectrum:
            energyBins = np.arange(min(energyList), max(energyList) + band_width, band_width)
            photonEnergies = np.array(energyList)
            count, bins_edges = np.histogram(photonEnergies, energyBins)

            # Find bin centers
            bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

            plt.figure(figsize=(10, 6))
            plt.plot(bin_centers, count, linestyle='-', color='b')
            plt.xlabel('Energy')
            plt.ylabel('Count')
            plt.title(title)
            if logarithmic:
                plt.yscale('log')
            plt.grid(True)
            plt.show()




    def singlePixelPhotonSpectrumOLD(self, band_width=5,plotSpectrum=False,logarithmic=False):

        pcEngine = PhotonCounting(self.imMat)
        indexIsolated = pcEngine.singlePhotonSinglePixelHits()

        braggEngine = Bragg(crystal_pitch=self.crys_pitch, crystal_roll=self.crys_roll,
                            camera_pitch=self.cam_pitch, camera_roll=self.cam_roll,
                            r_camera_spherical=self.r_cam_spherical)

        energyList = []
        xCoordList = []
        yCoordList = []

        print(len(indexIsolated))

        for indexRow in indexIsolated:
            yPixel = indexRow[0]  # i
            xPixel = indexRow[1]  # j

            x_0 = - braggEngine.xWidth / 2
            y_0 = + braggEngine.yWidth / 2

            x_coord = xPixel * braggEngine.pixelWidth + x_0   # x_o is such that the x coord for the exact center would be 0
            y_coord = y_0 - yPixel * braggEngine.pixelWidth

            energyVal = braggEngine.xyImagePlane_to_energy(x_coord, y_coord)

            energyList.append(energyVal)
            xCoordList.append(x_coord)
            yCoordList.append(y_coord)


        if plotSpectrum:
            energyBins = np.arange(min(energyList), max(energyList) + band_width, band_width)
            photonEnergies = np.array(energyList)
            count, bins_edges = np.histogram(photonEnergies, energyBins)

            # plt.figure(figsize=(10, 6))
            # plt.bar(bins_edges[:-1], count,width=band_width)
            # plt.ylabel('Count')
            # plt.xlabel('Energy')
            # plt.title('Photon Energy Spectrum with Single Photon Single Pixel Hits')
            # plt.show()

            # Compute bin centers
            bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

            plt.figure(figsize=(10, 6))
            plt.plot(bin_centers, count, linestyle='-', color='b')
            plt.xlabel('Energy')
            plt.ylabel('Count')
            plt.title('Photon Energy Spectrum with Single Photon Single Pixel Hits')
            if logarithmic:
                plt.yscale('log')
            plt.grid(True)
            plt.show()


        return energyList, xCoordList, yCoordList



if __name__ == "__main__":
    crysPitch = -0.3444672207603088
    CrysRoll = 0.018114148603524255
    CamPitch = 0.7950530342947064
    CamRoll = -0.005323879756451509
    rcam = 0.08395021
    thetacam = 2.567
    rcamSpherical = np.array([rcam,thetacam,np.pi])

    spectrum = Spectrum(high_intensity_points, crysPitch, CrysRoll, CamPitch, CamRoll, rcamSpherical)
    spectrum.singlePixelPhotonSpectrum(band_width=1,plotSpectrum=True,title="Photon Energy Spectrum with Multi Photon Single Pixel Hits Model 1"
                                       ,logarithmic=False)

    pass

