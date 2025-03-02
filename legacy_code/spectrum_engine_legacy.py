

class SpectrumEngineLegacy:

    def singlePixelPhotonSpectrum(self, band_width=5, plotSpectrum=False,
                                  spectrumTitle='Photon Energy Spectrum with Single Photon Single Pixel Hits',
                                  logarithmic=False):
        spc_engine = PhotonCounting(indexOfInterest=self.indexOfInterest,sp_adu_thr= self.sp_thresh,
                                    dp_adu_thr= self.dp_thresh)
        list_countij = spc_engine.checKernelType(kernelType="single_pixel")

        # Find how many single,double etc. photon hits we find
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

        pcEngine = PhotonCounting(indexOfInterest = self.indexOfInterest,
                                  sp_adu_thr = self.sp_thresh,
                                  dp_adu_thr = self.dp_thresh)
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

        imMat = loadData()[self.indexOfInterest]

        matrixInitialThreshold = (imMat > initialThreshold) & (imMat < secondThreshold)
        matrixSecondThreshold = imMat > secondThreshold

        for i in range(imMat.shape[0]):
            for j in range(imMat.shape[1]):
                if matrixInitialThreshold[i, j]:
                    count = 1
                elif matrixSecondThreshold[i, j]:
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
