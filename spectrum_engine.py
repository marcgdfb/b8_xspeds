from calibrateGeometry import *
from spc_engine import *


class Spectrum:
    def __init__(self, indexOfInterest,
                 crystal_pitch, crystal_roll, camera_pitch, camera_roll,
                 r_camera_spherical, sp_adu_thr=180, dp_adu_thr=240, removeTopRows=0):
        print("Spectrum Class Initialised with:")
        print("indexOfInterest = ", indexOfInterest)
        print("crystal_pitch = ", crystal_pitch)
        print("crystal_roll = ", crystal_roll)
        self.indexOfInterest = indexOfInterest
        self.sp_thresh = sp_adu_thr
        self.dp_thresh = dp_adu_thr

        self.crys_pitch = crystal_pitch
        self.crys_roll = crystal_roll
        self.cam_pitch = camera_pitch
        self.cam_roll = camera_roll
        self.r_cam_spherical = r_camera_spherical

        self.removeTopRows = removeTopRows

    def multiPixelSpectrum(self, band_width=5, methodList=None,
                           spectrumTitle='Photon Energy Spectrum',
                           plotSpectrum=False, logarithmic=False, intensity_arb_unit=False,plotEachSubSpectrum=False):

        if methodList is None:
            methodList = ["single_pixel", "double_pixel", "triple_pixel", ]

        spc_engine = PhotonCounting(indexOfInterest=self.indexOfInterest, sp_adu_thr=self.sp_thresh, dp_adu_thr=self.dp_thresh,
                                    removeRows0To_=self.removeTopRows,)
        bragg_engine = Bragg(crystal_pitch=self.crys_pitch, crystal_roll=self.crys_roll,
                             camera_pitch=self.cam_pitch, camera_roll=self.cam_roll,
                             r_camera_spherical=self.r_cam_spherical)
        energyDict = {}

        for method in methodList:
            print("-" * 30)
            listCountij = spc_engine.checKernelType(method)
            count_occurrences = Counter(countij[0] for countij in listCountij)
            print(f"Count Occurences for {method} Hits")
            print(count_occurrences)

            energyList = self.lists_energy(listCountij, braggEngine_init=bragg_engine)
            energyDict[method] = energyList

        if plotEachSubSpectrum:
            for method in methodList:
                energyList = energyDict[method]
                self.plotSpectrum(energyList, band_width,
                                  f'Photon Energy Spectrum with {method} Hits',
                                  intensity_arb_unit,logarithmic)

        if plotSpectrum:
            energyList = []
            for method in methodList:
                energyList.extend(energyDict[method])
            self.plotSpectrum(energyList, band_width, spectrumTitle, intensity_arb_unit,logarithmic)

    def singlePixelPhotonSpectrum(self, band_width=5, plotSpectrum=False,
                                  spectrumTitle='Photon Energy Spectrum with Single Photon Single Pixel Hits',
                                  logarithmic=False):
        spc_engine = PhotonCounting(self.indexOfInterest, self.sp_thresh, self.dp_thresh)
        list_countij = spc_engine.checKernelType(kernelType="single_pixel")

        # Find how many single,double etc photon hits we find
        count_occurrences = Counter(countij[0] for countij in list_countij)
        print(count_occurrences)

        braggEngine = Bragg(crystal_pitch=self.crys_pitch, crystal_roll=self.crys_roll,
                            camera_pitch=self.cam_pitch, camera_roll=self.cam_roll,
                            r_camera_spherical=self.r_cam_spherical)

        energyList = self.lists_energy(list_countij, braggEngine)

        if plotSpectrum:
            self.plotSpectrum(energyList, band_width, spectrumTitle, logarithmic)

    def singlePixelPhotonSpectrumOLD(self, band_width=5, plotSpectrum=False, intensity_arbUnits=False,
                                     logarithmic=False):

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

            x_coord = xPixel * braggEngine.pixelWidth + x_0  # x_o is such that the x coord for the exact center would be 0
            y_coord = y_0 - yPixel * braggEngine.pixelWidth

            energyVal = braggEngine.xyImagePlane_to_energy(x_coord, y_coord)

            energyList.append(energyVal)
            xCoordList.append(x_coord)
            yCoordList.append(y_coord)

        if plotSpectrum:
            spectrumtitle = 'Photon Energy Spectrum with Single Photon Single Pixel Hits'
            self.plotSpectrum(energyList, band_width, spectrumtitle, intensity_arbUnits, logarithmic)

        return energyList, xCoordList, yCoordList

    def simpleSpectrum(self, band_width=5,
                       initialThreshold=90,
                       secondThreshold=150,
                       plotSpectrum=False, logarithmic=False):

        braggEngine = Bragg(crystal_pitch=self.crys_pitch, crystal_roll=self.crys_roll,
                            camera_pitch=self.cam_pitch, camera_roll=self.cam_roll,
                            r_camera_spherical=self.r_cam_spherical)

        energyList = []

        matrixInitialThreshold = (self.imMat > initialThreshold) & (self.imMat < secondThreshold)
        matrixSecondThreshold = self.imMat > secondThreshold

        for i in range(self.imMat.shape[0]):
            for j in range(self.imMat.shape[1]):
                if matrixInitialThreshold[i, j]:
                    count = 1
                if matrixSecondThreshold[i, j]:
                    count = 2
                else:
                    continue

                yPixel = i
                xPixel = j

                x_0 = - braggEngine.xWidth / 2
                y_0 = + braggEngine.yWidth / 2

                x_coord = xPixel * braggEngine.pixelWidth + x_0  # x_o is such that the x coord for the exact center would be 0
                y_coord = y_0 - yPixel * braggEngine.pixelWidth

                energyVal = braggEngine.xyImagePlane_to_energy(x_coord, y_coord)

                energyList.extend([energyVal] * count)

        if plotSpectrum:
            spectrumTitle = 'Photon Energy Spectrum with Multi Photon Single Pixel Hits Model Simple'
            self.plotSpectrum(energyList, band_width, spectrumTitle, logarithmic)

    @staticmethod
    def plotSpectrum(energyList, band_width, spectrumTitle, intensity_arb_unit=False, logarithmic=False):

        energyBins = np.arange(min(energyList), max(energyList) + band_width, band_width)
        photonEnergies = np.array(energyList)
        count, bins_edges = np.histogram(photonEnergies, energyBins)

        # Find bin centers
        bin_centers = (bins_edges[:-1] + bins_edges[1:]) / 2

        plt.figure(figsize=(10, 6))
        if intensity_arb_unit:
            countIntensity = []
            for countE, center in zip(count, bin_centers):
                countIntensity.append(countE * center)
            countIntensity = np.array(countIntensity)
            plt.plot(bin_centers, countIntensity, linestyle='-', color='b')
            plt.ylabel('Intensity (arb. unit)')

        else:
            plt.plot(bin_centers, count, linestyle='-', color='b')
            plt.ylabel('Count')
        plt.xlabel('Energy')
        plt.title(spectrumTitle)
        if logarithmic:
            plt.yscale('log')
        plt.grid(True)
        plt.show()

    @staticmethod
    def lists_energy(list_countij, braggEngine_init):
        energyList = []

        for countij in list_countij:
            count = countij[0]
            iIndex = countij[1]
            jIndex = countij[2]

            yPixel = iIndex
            xPixel = jIndex

            x_0 = - braggEngine_init.xWidth / 2
            y_0 = + braggEngine_init.yWidth / 2

            x_coord = xPixel * braggEngine_init.pixelWidth + x_0  # x_o is such that the x coord for the exact center would be 0
            y_coord = y_0 - yPixel * braggEngine_init.pixelWidth

            energyVal = braggEngine_init.xyImagePlane_to_energy(x_coord, y_coord)

            energyList.extend([energyVal] * count)

        return energyList


if __name__ == "__main__":
    crysPitch = -0.3444672207603088
    CrysRoll = 0.018114148603524255
    CamPitch = 0.7950530342947064
    CamRoll = -0.005323879756451509
    rcam = 0.08395021
    thetacam = 2.567
    rcamSpherical = np.array([rcam, thetacam, np.pi])

    spectrum = Spectrum(8, crysPitch, CrysRoll, CamPitch, CamRoll, rcamSpherical,removeTopRows=0)
    spectrum.multiPixelSpectrum(band_width=1, plotSpectrum=True, logarithmic=False,plotEachSubSpectrum=True,intensity_arb_unit=True,
                                spectrumTitle="Photon Energy Spectrum with Multi Photon Single Pixel Hits Model 2")

    # spectrum = Spectrum(imData[8], 100, crysPitch, CrysRoll, CamPitch, CamRoll, rcamSpherical)
    # spectrum.singlePixelPhotonSpectrumOLD(band_width=1, plotSpectrum=True, intensity_arbUnits=True, logarithmic=False)
    # spectrum.singlePixelPhotonSpectrum(band_width=1,plotSpectrum=True,title="Photon Energy Spectrum with Multi Photon Single Pixel Hits Model 1"
    #                                    ,logarithmic=False)
    # spectrum.simpleSpectrum(band_width=2,plotSpectrum=True,logarithmic=False)

    pass
