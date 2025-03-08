from scipy.optimize import minimize, curve_fit
from imagePreProcessing import *
from tools import *
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import cma

# line_right_txt = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\data_logs\LineOnRightFitLog.txt"
# line_left_txt = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\data_logs\LineOnLeftFitLog.txt"
geometryLog = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\data_logs\geometryFitLog.txt"
quadLineLog = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\data_logs\quadraticLineFits.txt"


class Geometry:
    def __init__(self, crystal_pitch, crystal_roll, camera_pitch, camera_roll,
                 r_camera_spherical, xpixels=2048, ypixels=2048, pixelWidth=pixel_width, ):
        self.crystal_pitch = crystal_pitch
        self.crystal_roll = crystal_roll
        self.nxcrystal = np.sin(crystal_pitch) * np.cos(crystal_roll)
        self.nycrystal = np.sin(crystal_pitch) * np.sin(crystal_roll)
        self.nzcrystal = np.cos(crystal_pitch)
        self.nCrystal = np.array([self.nxcrystal, self.nycrystal, self.nzcrystal])

        self.camera_pitch = camera_pitch
        self.camera_roll = camera_roll
        self.nxcam = np.sin(camera_pitch) * np.cos(camera_roll)
        self.nycam = np.sin(camera_pitch) * np.sin(camera_roll)
        self.nzcam = np.cos(camera_pitch)
        self.nCam = np.array([self.nxcam, self.nycam, self.nzcam])

        self.r_camera_spherical = r_camera_spherical
        self.xpixels = xpixels
        self.ypixels = ypixels
        self.pixelWidth = pixelWidth
        self.xWidth = xpixels * pixelWidth
        self.yWidth = ypixels * pixelWidth
        self.r_cam_spherical = r_camera_spherical
        self.r_cam_cart = spherical_to_cartesian(r_camera_spherical)

    def __init__Old(self, n_crystal, n_camera, r_camera_spherical,
                    xpixels=2048, ypixels=2048, pixelWidth=pixel_width,
                    ):
        self.xpixels = xpixels
        self.ypixels = ypixels
        self.pixelWidth = pixelWidth
        self.xWidth = xpixels * pixelWidth
        self.yWidth = ypixels * pixelWidth
        self.nCrystal = n_crystal
        self.nCam = n_camera
        self.r_cam_spherical = r_camera_spherical
        self.r_cam_cart = spherical_to_cartesian(r_camera_spherical)

    def xy_coords_of_E(self, energy_eV, phiStepSize=0.0001):

        # Rotation matrix from crystal 001 to n crystal
        rotMatrix_crys = rotMatrixUsingEuler(self.crystal_pitch, self.crystal_roll)

        # angle between crystal plane and ray
        theta_E = bragg_E_to_theta(energy_eV)
        # polar coordinate of this angle (still as if crystal is 001)
        theta_E_polar = np.pi / 2 + theta_E

        list_xy = []

        phiDif = 0

        for phi in np.arange(3 * np.pi / 4, 5 * np.pi / 4, phiStepSize):
            v_rayPrime_spherical = np.array([1, theta_E_polar, phi])
            v_rayPrime_cart = spherical_to_cartesian(v_rayPrime_spherical)
            v_ray_cart = np.dot(rotMatrix_crys, v_rayPrime_cart)
            v_ray_cart_normalised = v_ray_cart / np.linalg.norm(v_ray_cart)

            v_ray_spherical = cartesian_to_spherical(v_ray_cart_normalised)
            # print("Energy eV = ", energy_eV, "Theta Polar in normal frame = ", v_ray_spherical[1])

            x_plane, y_plane = ray_in_planeCamera(v_ray_cart=v_ray_cart_normalised, n_camera_cart=self.nCam,
                                                  r_camera_cart=self.r_cam_cart)

            if ((abs(x_plane) < (self.xWidth - self.pixelWidth) / 2).all() and
                    (abs(y_plane) < (self.yWidth - self.pixelWidth) / 2).all()):
                if [x_plane, y_plane] not in list_xy:
                    list_xy.append([x_plane, y_plane])

                    phi_difference_to_pi = abs(abs(v_ray_spherical[2]) - np.pi)

                    if phi_difference_to_pi > phiDif:
                        phiDif = phi_difference_to_pi

        # print(f"The largest phi deviation from pi was {phiDif}")

        if not list_xy:
            print(f"No values for E = {energy_eV}")

        return list_xy

    def xy_coords_of_EAlphaBeta(self, ):

        listAlpha = self.xy_coords_of_E(E_Lalpha_eV)

        for i, alpharow in enumerate(listAlpha):
            print(f"Alpha row {i}: xPixel = {alpharow[0] / self.pixelWidth}, yPixel = {alpharow[1] / self.pixelWidth}")
        listBeta = self.xy_coords_of_E(E_Lbeta_eV)
        for i, betaarow in enumerate(listBeta):
            print(f"Beta row {i}: xPixel = {betaarow[0] / self.pixelWidth}, yPixel = {betaarow[1] / self.pixelWidth}")

        return listAlpha + listBeta

    def createLinesMatrix(self, imageMat, valMax, phiStepSize=0.0001):

        # If we are optimising r camera to be that going to the centre of the camera then the 0,0 pixel is displaced by:
        x_0 = - self.xWidth / 2
        y_0 = + self.yWidth / 2

        matrixLines = np.zeros((imageMat.shape[0], imageMat.shape[1]))

        for E in [E_Lalpha_eV, E_Lbeta_eV]:
            xy_E_list = self.xy_coords_of_E(E, phiStepSize)

            for row in xy_E_list:
                x_pixel = round((row[0] - x_0) / self.pixelWidth)
                y_pixel = round((y_0 - row[1]) / self.pixelWidth)

                matrixLines[y_pixel, x_pixel] = valMax

        return matrixLines

    def visualiseGeometry(self, list_energy_eV=None,
                          num_r_points=30, num_phi_points=30, ):

        # Get the rotation matrix from the crystal coordinate system to our coordinate system.
        if list_energy_eV is None:
            list_energy_eV = [E_Lalpha_eV, E_Lbeta_eV, E_min_eV, E_max_eV]
        rotMatrix_cry = inverseRotation_matrix(self.nCrystal)

        # Initialise Figure
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for energy_eV in list_energy_eV:
            # Compute the angle theta from the given energy.
            theta = bragg_E_to_theta(energy_eV)
            thetaPolar = np.pi / 2 + theta

            # Create a 3D grid of points. We input theta from above
            r_vals = np.linspace(0, 0.3, num_r_points)
            phi_vals = np.linspace(3 * np.pi / 4, 5 * np.pi / 4, num_phi_points)
            R, Phi = np.meshgrid(r_vals, phi_vals)

            # Finding the coordinates
            X_prime = R * np.sin(thetaPolar) * np.cos(Phi)
            Y_prime = R * np.sin(thetaPolar) * np.sin(Phi)
            Z_prime = R * np.cos(thetaPolar)

            # Rotate into normal coords
            points_prime = np.vstack((X_prime.ravel(), Y_prime.ravel(), Z_prime.ravel()))
            points_rotated = np.dot(rotMatrix_cry, points_prime)
            X_rot = points_rotated[0, :].reshape(X_prime.shape)
            Y_rot = points_rotated[1, :].reshape(Y_prime.shape)
            Z_rot = points_rotated[2, :].reshape(Z_prime.shape)

            # Plot the cone surface.
            ax.plot_surface(X_rot, Y_rot, Z_rot, alpha=0.5, edgecolor='none', label=f'{energy_eV:.2f} eV')
            # Plot the edge of the cone.
            ax.plot(X_rot[-1, :], Y_rot[-1, :], Z_rot[-1, :], linewidth=2, label=f'Edge {energy_eV:.2f} eV')

        # Mark the apex of the cones.
        ax.scatter(0, 0, 0, color='r', s=50, label='Origin')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Rotated Cones for Different Energies')
        ax.legend()
        plt.show()

    def rayBragg(self, energy_eV, phi_rad):

        # Test n crystal as nVectorFromEuler(-0.19,0)

        theta_E = bragg_E_to_theta(energy_eV)
        theta_E_polar = np.pi / 2 + theta_E

        v_rayPrime_spherical = np.array([1, theta_E_polar, phi_rad])
        print("v_rayPrime_spherical", v_rayPrime_spherical)
        v_rayPrime_cart = spherical_to_cartesian(v_rayPrime_spherical)
        print("v_rayPrime_cart", v_rayPrime_cart)

        v_ray_cart = np.dot(rotMatrixUsingEuler(self.crystal_pitch, self.crystal_roll), v_rayPrime_cart)
        v_ray_cart_normalised = v_ray_cart / np.linalg.norm(v_ray_cart)  # This works
        print("v_ray_cart_normalised", v_ray_cart_normalised)
        v_ray_spherical = cartesian_to_spherical(v_ray_cart_normalised)

        print(v_ray_spherical)


class Calibrate:
    def __init__(self, imageMatrix, logTextFile=None):
        self.imMat = imageMatrix
        self.log = logTextFile

    # The following code serves to compute quadratic curves that describe our lines of interest
    def computeLine(self, a, b, cBounds, plotGraph=False, cPlotVal=1450, plotResults=False,
                    adjacentWeight=1):
        """
        Assumes the lines can be parameterised as a quadratic X = A * (Y - B) ** 2 + C
        For a certain value of A and B the line integral is performed for different C values within the bounds

        :returns A list of lists [cval, totVal] which have the c value and the value given by its line integral
        """

        def quadraticPixelised(Y, A, B, C):
            return np.round(A * (Y - B) ** 2 + C, 0).astype(int)

        xWidth = self.imMat.shape[1]  # 2048
        yWidth = self.imMat.shape[0]  # 2048
        yCoords = np.arange(start=0, stop=yWidth, step=1)

        if plotGraph:
            xCoordsPlot = quadraticPixelised(yCoords, a, b, cPlotVal)

            testImMat = np.zeros((self.imMat.shape[0], self.imMat.shape[1]))

            for x, y in zip(xCoordsPlot, yCoords):
                if 0 <= x < xWidth:
                    testImMat[y, x] = np.max(im_very_clear)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1), plt.imshow(self.imMat, cmap='hot'), plt.title(
                'Original Image on which the quadratic is fitted')
            plt.subplot(1, 2, 2), plt.imshow(im_very_clear + testImMat, cmap='hot'), plt.title(
                'Clearer Image Matrix with quadratic {a}*(Y-{b})**2 + {cPlotVal}')
            plt.show()

            return

        def ComputeLineIntegral():
            list_c_integralVal = []
            for cval in np.arange(start=cBounds[0], stop=cBounds[1], step=1):
                # print(cval)
                xCoords = quadraticPixelised(yCoords, a, b, cval)

                totVal = 0

                for xL, yL in zip(xCoords, yCoords):
                    # print(x,y)

                    if 0 <= xL < xWidth:
                        totVal += self.imMat[yL, xL]
                    if xL + 1 < xWidth:
                        totVal += adjacentWeight * self.imMat[yL, xL + 1]
                    if 0 < xWidth:
                        totVal += adjacentWeight * self.imMat[yL, xL - 1]

                list_c_integralVal.append([cval, totVal])

            return list_c_integralVal

        lineintrgral = ComputeLineIntegral()

        if plotResults:
            plt.plot(np.array(lineintrgral)[:, 0], np.array(lineintrgral)[:, 1])
            plt.ylabel("Line Integral with width 3 pixels")
            plt.xlabel("C value in x = A(y-B)**2 + C")
            plt.title(f"Line Integral with values A = {a}, B = {b} as a function of C ")
            plt.show()

        return lineintrgral

    def fitGaussianToLineIntegral(self, a, b, cBounds, plotGauss=False):
        """
        For a given a, b in a quadratic of the form X =  * (Y - b) ** 2 + C we fit a gaussian to the peak.
        This peak is associtaed with one of the lines we see in the image.

        :param a:
        :param b:
        :param cBounds:
        :param plotGauss:
        :return: amp, sigma, cpeak of the gaussian

        Note if the c bounds is too small such that we cannot compuete a standard deviation sigma is set to 0.
        This is handlded in the next part of code. It is worth saying this is to be avoided
        """
        lineIntegralList = self.computeLine(a, b, cBounds)
        cVals = np.array(lineIntegralList)[:, 0]
        lineIntegralVals = np.array(lineIntegralList)[:, 1]

        def gaussian(X, amp_, xpeak, sigma_, c_offset):
            return amp_ * np.exp(-(X - xpeak) ** 2 / (2 * sigma_ ** 2)) + c_offset

        amp_guess = np.max(lineIntegralVals)
        xpeak_guess = cVals[np.argmax(lineIntegralVals)]
        sigma_guess = 20
        c_offset_guess = np.min(lineIntegralVals)

        try:
            popt, pcov = curve_fit(gaussian, cVals, lineIntegralVals,
                                   p0=[amp_guess, xpeak_guess, sigma_guess, c_offset_guess], maxfev=2000)
            if plotGauss:
                plt.plot(cVals, lineIntegralVals)
                plt.plot(cVals, gaussian(cVals, *popt))
                plt.ylabel("Line Integral with width 3 pixels")
                plt.xlabel("C value in x = A(y-B)**2 + C")
                plt.show()

            amp = popt[0]
            sigma = popt[2]
            cPeak = popt[1]

            return amp, sigma, cPeak
        except RuntimeError:
            print(RuntimeError)
            print("The fit could not be computed. A larger range for cBounds can often help with a better fit.")

            return amp_guess, 0, xpeak_guess

    def optimiseLines(self, aBounds, bBounds, cBounds, sigmaWeighting=1, plotGraph=False, plotResults=False,
                      plotGauss=False):
        """
        Using fitGaussianToLineIntegral we will now optimise to encourage a maximal peak with a minimum sigma.
        Recall the form X = A * (Y - B) ** 2 + C
        :param aBounds: bounds of A
        :param bBounds: bounds of B
        :param cBounds: bounds of C
        :param sigmaWeighting: A loss function of -amp + sigmaWeighting * sigma is used
        :param plotGraph:
        :param plotResults:
        :return:
        """

        def lossFunc(params, sigmaWeighting):

            print("-" * 40)
            Aval, Bval = params

            amp_, sigma, cPeak = self.fitGaussianToLineIntegral(Aval, Bval, cBounds)

            print(f"A = {Aval}, B = {Bval}, amplitude = {amp_},sigma = {sigma},cPeak = {cPeak}")
            # Aiming to maximise amp while minimising sigma
            print("Loss", -amp_ + sigmaWeighting * sigma)
            return -amp_ + sigmaWeighting * sigma

        bounds = [aBounds, bBounds]
        initial_guess = np.array([(aBounds[0] + aBounds[1]) / 2, (bBounds[0] + bBounds[1]) / 2])

        # result = differential_evolution(lossFunc, bounds, args=(sigmaWeighting,))

        result = minimize(lossFunc, initial_guess, args=(sigmaWeighting,), bounds=bounds, method='Nelder-Mead',
                          callback=callbackminimise, options={'maxiter': 30})

        A, B = result.x
        lossOptimised = lossFunc(result.x, sigmaWeighting)

        amp, sigma, cPeak = self.fitGaussianToLineIntegral(A, B, cBounds, plotGauss=plotGauss)

        def logResults():
            append_to_file(self.log, "-" * 30)
            append_to_file(self.log, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            append_to_file(self.log,
                           f"aBounds = {aBounds}" + "\n" + f"bBounds = {bBounds}" + "\n" + f"cBounds = {cBounds}")
            append_to_file(self.log, f"Optimised A = {A}, B = {B}, C = {cPeak}")
            append_to_file(self.log, f"Loss = {lossOptimised}")

            print("-" * 30)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"aBounds = {aBounds}" + "\n" + f"bBounds = {bBounds}" + "\n" + f"cBounds = {cBounds}")
            print(f"Optimised A = {A}, B = {B}, C = {cPeak}")
            print(f"Loss = {lossOptimised}")

        if self.log is not None:
            logResults()

        if plotResults:

            if sigma == 0:
                if self.log is not None:
                    append_to_file(self.log, "Gaussian not fitted due to RuntimeError")
            else:
                if self.log is not None:
                    append_to_file(self.log, "Gaussian fitted successfully")
            if plotGraph:
                self.computeLine(A, B, cBounds, plotGraph=True, cPlotVal=cPeak, plotResults=True)

        return A, B, cPeak

    def matrixWithLinesOptimisation(self, plotLines=False, returnABC=False, plotGauss=False,
                                    cBoundsLeft=(1200, 1360), aBoundsLeft=(0.00001, 0.0001), bBoundsLeft=(800, 950),
                                    cBoundsRight=(1380, 1460), aBoundsRight=(0.00001, 0.0001), bBoundsRight=(800, 950),
                                    ):

        # The following ranges are for image 8. They are expected to work for any image with data
        # The Line ends further along x at the bottom than it does y

        append_to_file(self.log, "-" * 30)
        append_to_file(self.log, "Left Line")
        print("Working on the left Line")
        ALeft, BLeft, cLeft = self.optimiseLines(aBoundsLeft, bBoundsLeft, cBoundsLeft, plotGraph=False,
                                                 plotResults=False, plotGauss=plotGauss)
        append_to_file(self.log, "-" * 30)
        append_to_file(self.log, "Right Line")
        print("Working on the right Line")
        ARight, BRight, cRight = self.optimiseLines(aBoundsRight, bBoundsRight, cBoundsRight, plotGraph=False,
                                                    plotResults=False, plotGauss=plotGauss)

        leftLineMat = self.matrixWithLines(ALeft, BLeft, cLeft, plotLines=False)
        rightLineMat = self.matrixWithLines(ARight, BRight, cRight, plotLines=False)

        if plotLines:
            plt.imshow(leftLineMat + rightLineMat, cmap='hot')
            plt.show()

        if returnABC:
            left_vars = [ALeft, BLeft, cLeft]
            right_vars = [ARight, BRight, cRight]
            return leftLineMat, rightLineMat, left_vars, right_vars
        else:
            return leftLineMat, rightLineMat

    def matrixWithLines(self, Aoptimised, Boptimised, Coptimised,
                        plotLines=False):

        def CreateMatrix():

            def quadraticPixelised(Y, A, B, C):
                return np.round(A * (Y - B) ** 2 + C, 0).astype(int)

            xWidth = self.imMat.shape[1]  # 2048
            yWidth = self.imMat.shape[0]  # 2048
            yCoords = np.arange(start=0, stop=yWidth, step=1)

            xCoordsPlot = quadraticPixelised(yCoords, Aoptimised, Boptimised, Coptimised)

            linesMat = np.zeros((self.imMat.shape[0], self.imMat.shape[1]))

            for xcoord, y in zip(xCoordsPlot, yCoords):
                if 0 <= xcoord < xWidth:
                    linesMat[y, xcoord] = 1

            return linesMat

        linesMatrix = CreateMatrix()

        if plotLines:
            plt.figure(figsize=(10, 5))
            linesMatrixForPlotting = np.where(linesMatrix > 0, np.max(im_very_clear), 0)

            plt.subplot(1, 2, 1), plt.imshow(self.imMat, cmap='hot'), plt.title(
                'Original Image on which the quadratic is fitted')
            plt.subplot(1, 2, 2), plt.imshow(im_very_clear + linesMatrixForPlotting, cmap='hot'), plt.title(
                'Clearer Image Matrix with curves')
            plt.show()

        return linesMatrix


class Bragg:
    def __init__(self, crystal_pitch, crystal_roll, camera_pitch, camera_roll,
                 r_camera_spherical, xpixels=2048, ypixels=2048, pixelWidth=pixel_width, ):
        self.crystal_pitch = crystal_pitch
        self.crystal_roll = crystal_roll
        self.nxcrystal = np.sin(crystal_pitch) * np.cos(crystal_roll)
        self.nycrystal = np.sin(crystal_pitch) * np.sin(crystal_roll)
        self.nzcrystal = np.cos(crystal_pitch)
        self.nCrystal = np.array([self.nxcrystal, self.nycrystal, self.nzcrystal])

        self.camera_pitch = camera_pitch
        self.camera_roll = camera_roll
        self.nxcam = np.sin(camera_pitch) * np.cos(camera_roll)
        self.nycam = np.sin(camera_pitch) * np.sin(camera_roll)
        self.nzcam = np.cos(camera_pitch)
        self.nCam = np.array([self.nxcam, self.nycam, self.nzcam])

        self.r_camera_spherical = r_camera_spherical
        self.xpixels = xpixels
        self.ypixels = ypixels
        self.pixelWidth = pixelWidth
        self.xWidth = xpixels * pixelWidth
        self.yWidth = ypixels * pixelWidth
        self.r_cam_spherical = r_camera_spherical
        self.r_cam_cart = spherical_to_cartesian(r_camera_spherical)

    def xyImagePlane_to_energy(self, x_imPlane, y_imPlane, ):
        """
        The origin of the following coordinates is at the center of the CCD
        :param x_imPlane: The x coordinate in meters in the image plane.
        :param y_imPlane: The y coordinate in meters in the image plane.
        :return: The energy in eV of this point
        """

        # The x_imPlane,y_imPlane are those within the plane of the ccd.
        # The r_spherical coordinate goes to the center of the image

        # This is v_ray_cam from before
        r_to_point = xyPlane_to_ray(x_imPlane, y_imPlane, camera_ptich=self.camera_pitch, camera_roll=self.camera_roll,
                                    r_camera_cart=self.r_cam_cart)
        v_ray_cart_normalised = r_to_point / np.linalg.norm(r_to_point)

        nCrysNormalised = self.nCrystal / np.linalg.norm(self.nCrystal)

        sinthetaBragg = abs(np.dot(nCrysNormalised, v_ray_cart_normalised))
        thetaBragg = np.arcsin(sinthetaBragg)

        energy_eV = bragg_theta_to_E(theta_rad=thetaBragg)

        return energy_eV


def calibrateSaveQuadratics(indexOfInterest, thr=100, pltGauss=False, plot_finalLines=False, save=False,
                            logFile=quadLineLog,
                            bBounds=(790, 870),
                            folderpath_to_save=r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\stored_variables\ABC_lines\unsupervised"
                            ):
    print(indexOfInterest)
    image_mat = loadData()[indexOfInterest]
    image_mat = np.where(image_mat > thr, image_mat, 0)
    imMatVeryClear = imVeryClear(image_mat, 0, (21, 5))

    cal = Calibrate(image_mat, logFile)
    append_to_file(cal.log, "-" * 30)
    append_to_file(cal.log, f"Index of Interest {indexOfInterest}, thresholded above {thr}")
    leftLineMat, rightLineMat, left_vars, right_vars = cal.matrixWithLinesOptimisation(False, returnABC=True,
                                                                                       plotGauss=pltGauss,
                                                                                       cBoundsLeft=(1220, 1340))
    Aleft = left_vars[0]
    Bleft = left_vars[1]
    cLeft = left_vars[2]
    ARight = right_vars[0]
    Bright = right_vars[1]
    cRight = right_vars[2]

    LinesMatLeft = np.where(leftLineMat > 0, thr, 0)
    LinesMatRight = np.where(rightLineMat > 0, thr, 0)

    if plot_finalLines:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.imshow(image_mat, cmap='hot'), plt.title(
            f"Image {indexOfInterest} thresholded above {thr}")
        plt.subplot(1, 2, 2), plt.imshow(imMatVeryClear + LinesMatLeft + LinesMatRight, cmap='hot'), plt.title(
            f"Average Pooled Image with geometric lines")
        plt.show()
        if save:
            if input("Type 1 to save") == "1":
                print("saving...")
                vals = np.array([[Aleft, Bleft, cLeft],
                                 [ARight, Bright, cRight]])
                filename = f"{indexOfInterest}"
                np.save(f"{folderpath_to_save}/{filename}.npy", vals)
    else:
        if save:
            print("saving...")
            vals = np.array([[Aleft, Bleft, cLeft],
                             [ARight, Bright, cRight]])
            filename = f"{indexOfInterest}"
            np.save(f"{folderpath_to_save}/{filename}.npy", vals)


def optimiseGeometryToCalibratedLines(indexOfInterest, initialGuess, bounds, useSavedVals=True, thr=100,
                                      r_thetaval=2.567, logTextFile=None, saveResults=False,
                                      weight_ofsettedPoints=0.5, iterations=30, plot=False):
    print(indexOfInterest)
    image_mat = loadData()[indexOfInterest]
    image_mat = np.where(image_mat > thr, image_mat, 0)
    # imMatVeryClear = imVeryClear(image_mat, 0, (21, 5))

    if useSavedVals:
        folderpath = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\stored_variables\ABC_lines\unsupervised"
        filename = f"{indexOfInterest}"
        vals = np.load(f"{folderpath}/{filename}.npy")
        Aleft = vals[0, 0]
        Bleft = vals[0, 1]
        Cleft = vals[0, 2]
        Aright = vals[1, 0]
        Bright = vals[1, 1]
        Cright = vals[1, 2]

        print("Using Saved Values")
        print("Left Line:")
        print(f"Aleft: {Aleft}, Bleft: {Bleft}, Cleft: {Cleft}")
        print("Right Line:")
        print(f"Aright: {Aright}, Bright: {Bright}, Cright: {Cright}")

        linesMatLeft = Calibrate(image_mat, None).matrixWithLines(Aoptimised=Aleft, Boptimised=Bleft,
                                                                  Coptimised=Cleft, plotLines=False)
        linesMatRight = Calibrate(image_mat, None).matrixWithLines(Aoptimised=Aright, Boptimised=Bright,
                                                                   Coptimised=Cright, plotLines=False)

    else:
        cal = Calibrate(image_mat, quadLineLog)
        append_to_file(cal.log, "-" * 30)
        append_to_file(cal.log, "Lines fits used for geometric fitting at a similar time")
        linesMatLeft, linesMatRight, left_vars, right_vars = cal.matrixWithLinesOptimisation(plotLines=False,
                                                                                             returnABC=False,
                                                                                             cBoundsLeft=(1220, 1340),
                                                                                             )

        print("Left Line:")
        print(f"Aleft: {left_vars[0]}, Bleft: {left_vars[1]}, Cleft: {left_vars[2]}")
        print("Right Line:")
        print(f"Aright: {right_vars[0]}, Bright: {right_vars[1]}, Cright: {right_vars[2]}")

        if saveResults:
            print("saving...")
            vals = np.array([[left_vars[0], left_vars[1], left_vars[2]],
                             [right_vars[0], right_vars[1], right_vars[2]]])
            folderpath = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\stored_variables\ABC_lines\unsupervised"
            filename = f"{indexOfInterest}"
            np.save(f"{folderpath}/{filename}.npy", vals)

    def lossFunction(params):
        p = params

        geo = Geometry(crystal_pitch=p[0], crystal_roll=p[1],
                       camera_pitch=p[2], camera_roll=p[3],
                       r_camera_spherical=np.array([p[4], r_thetaval, np.pi]))

        alphaLineCoords = geo.xy_coords_of_E(E_Lalpha_eV)  # More Right / right line
        betaLineCoords = geo.xy_coords_of_E(E_Lbeta_eV)  # More Left / left line

        # If we are optimising r camera to be that going to the centre of the camera then the 0,0 pixel is displaced by:
        x_0 = - geo.xWidth / 2
        y_0 = + geo.yWidth / 2

        lossPositive = 0

        # Treat each Line Separately as not to over encourage curvature
        for row in alphaLineCoords:
            x_pixel = round((row[0] - x_0) / geo.pixelWidth)
            y_pixel = round((y_0 - row[1]) / geo.pixelWidth)

            lossPositive += linesMatRight[y_pixel, x_pixel]

            if x_pixel + 1 < geo.xWidth:
                lossPositive += weight_ofsettedPoints * linesMatRight[y_pixel, x_pixel + 1]
            if x_pixel - 1 >= 0:
                lossPositive += weight_ofsettedPoints * linesMatRight[y_pixel, x_pixel - 1]

        for row in betaLineCoords:
            x_pixel = round((row[0] - x_0) / geo.pixelWidth)
            y_pixel = round((y_0 - row[1]) / geo.pixelWidth)

            lossPositive += linesMatLeft[y_pixel, x_pixel]

            if x_pixel + 1 < geo.xWidth:
                lossPositive += weight_ofsettedPoints * linesMatLeft[y_pixel, x_pixel + 1]
            if x_pixel - 1 >= 0:
                lossPositive += weight_ofsettedPoints * linesMatLeft[y_pixel, x_pixel - 1]

        # We wish to maximise the integral along these lines
        loss = - lossPositive

        def printParamsAndLoss():
            print("-" * 30)
            print("Params:")
            print("crysPitch = ", p[0], "CrysRoll = ", p[1])
            print("CamPitch = ", p[2], "CamRoll = ", p[3])
            print("rcamSpherical = ", np.array([p[4], r_thetaval, np.pi]))
            print("Loss = ", loss)

        printParamsAndLoss()

        return loss

    result = minimize(lossFunction, initialGuess, bounds=bounds, method='Nelder-Mead', options={'maxiter': iterations},
                      callback=callbackminimise)

    optimisedParams = result.x

    crystal_pitch = optimisedParams[0]
    crystal_roll = optimisedParams[1]
    camera_pitch = optimisedParams[2]
    camera_roll = optimisedParams[3]
    rcamSphericalOptimised = np.array([optimisedParams[4], r_thetaval, np.pi])

    ncrysOptimised = nVectorFromEuler(crystal_pitch, crystal_roll)
    ncamOptimised = nVectorFromEuler(camera_pitch, camera_roll)
    rcamSphericalOptimised = np.array([optimisedParams[4], r_thetaval, np.pi])
    lossOptimised = lossFunction(optimisedParams)

    geoOptimised = Geometry(crystal_pitch=crystal_pitch, crystal_roll=crystal_roll,
                            camera_pitch=camera_pitch, camera_roll=camera_roll,
                            r_camera_spherical=rcamSphericalOptimised, )
    linesMatGeoOptimised = geoOptimised.createLinesMatrix(image_mat, 1)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.imshow(linesMatLeft + linesMatRight, cmap='hot'), plt.title('Quadratic Lines')
        plt.subplot(1, 2, 2), plt.imshow(linesMatGeoOptimised + linesMatLeft + linesMatRight, cmap='hot'), plt.title(
            'Geometic Lines plotted Over')
        plt.show()

    def logResults():
        append_to_file(logTextFile, "-" * 30)
        append_to_file(logTextFile, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        append_to_file(logTextFile, f"crystal pitch bounds = {bounds[0]}, crystal roll bounds = {bounds[1]}")
        append_to_file(logTextFile, f"camera pitch bounds = {bounds[2]}, camera roll bounds = {bounds[3]}")
        append_to_file(logTextFile, f"rcamBounds = {bounds[4]}")

        append_to_file(logTextFile, f"optimised crystal pitch = {crystal_pitch}, n crystal roll = {crystal_roll}")
        append_to_file(logTextFile, f"optimised camera pitch = {camera_pitch}, n camera roll = {camera_roll}")
        append_to_file(logTextFile, f"optimised rcam spherical = {rcamSphericalOptimised}")

        append_to_file(logTextFile, f"Optimised n crystal = {ncrysOptimised}")
        append_to_file(logTextFile, f"Optimised n camera = {ncamOptimised}")
        append_to_file(logTextFile, f"Optimised r camera = {rcamSphericalOptimised}")
        append_to_file(logTextFile, f"Loss = {lossOptimised}")

        print("-" * 30)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"Optimised crystal pitch = {crystal_pitch}, n crystal roll = {crystal_roll}")
        print(f"optimised camera pitch = {camera_pitch}, n camera roll = {camera_roll}")
        print(f"optimised rcam spherical = {rcamSphericalOptimised}")
        print(f"Optimised n crystal = {ncrysOptimised}")
        print(f"Optimised n camera = {ncamOptimised}")
        print(f"Optimised r camera = {rcamSphericalOptimised}")
        print(f"Loss = {lossOptimised}")

    if logTextFile is not None:
        logResults()

    if saveResults:
        folderpath = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\stored_variables\geometry"
        filename = f"{indexOfInterest}"
        np.save(f"{folderpath}/{filename}.npy", optimisedParams)


def optimiseGeometryToCalibratedLibesCMA(initialGuess, bounds, sigma0=0.001, r_thetaval=2.62, logTextFile=None):
    # For Image 8
    linesMatLeft = Calibrate(imTest, None).matrixWithLines(Aoptimised=6.613794078473409e-05, Boptimised=862,
                                                           Coptimised=1278, plotLines=False)
    linesMatRight = Calibrate(imTest, None).matrixWithLines(Aoptimised=7.063102423636962e-05, Boptimised=862,
                                                            Coptimised=1418, plotLines=False)

    def lossFunction(params):

        weight_ofsettedPoints = 0.5

        p = params

        geo = Geometry(crystal_pitch=p[0], crystal_roll=p[1],
                       camera_pitch=p[2], camera_roll=p[3],
                       r_camera_spherical=np.array([p[4], r_thetaval, np.pi]), )

        alphaLineCoords = geo.xy_coords_of_E(E_Lalpha_eV)  # More Right / right line
        betaLineCoords = geo.xy_coords_of_E(E_Lbeta_eV)  # More Left / left line

        # If we are optimising r camera to be that going to the centre of the camera then the 0,0 pixel is displaced by:
        x_0 = - geo.xWidth / 2
        y_0 = + geo.yWidth / 2

        lossPositive = 0
        # Treat each Line Separately as not to over encourage curvature
        for row in alphaLineCoords:
            x_pixel = round((row[0] - x_0) / geo.pixelWidth)
            y_pixel = round((y_0 - row[1]) / geo.pixelWidth)

            lossPositive += linesMatRight[y_pixel, x_pixel]

            if x_pixel + 1 < geo.xWidth:
                lossPositive += weight_ofsettedPoints * linesMatRight[y_pixel, x_pixel + 1]
            if x_pixel - 1 >= 0:
                lossPositive += weight_ofsettedPoints * linesMatRight[y_pixel, x_pixel - 1]

        for row in betaLineCoords:
            x_pixel = round((row[0] - x_0) / geo.pixelWidth)
            y_pixel = round((y_0 - row[1]) / geo.pixelWidth)

            lossPositive += linesMatLeft[y_pixel, x_pixel]

            if x_pixel + 1 < geo.xWidth:
                lossPositive += weight_ofsettedPoints * linesMatLeft[y_pixel, x_pixel + 1]
            if x_pixel - 1 >= 0:
                lossPositive += weight_ofsettedPoints * linesMatLeft[y_pixel, x_pixel - 1]

        # We wish to maximise the integral along these lines
        loss = - lossPositive

        def printParamsAndLoss():
            print("-" * 30)
            print("Params:")
            print("crysPitch = ", p[0], "CrysRoll = ", p[1])
            print("CamPitch = ", p[2], "CamRoll = ", p[3])
            print("rcamSpherical = ", np.array([p[4], r_thetaval, np.pi]))
            print("Loss = ", loss)

        printParamsAndLoss()

        return loss

    # Set up the options dictionary with the bounds
    opts = {
        'bounds': bounds,
        # Other options can be added here, like 'maxfevals' or 'popsize'
    }

    result = cma.fmin(lossFunction, initialGuess, sigma0, opts)

    optimisedParams = result[0]

    crystal_pitch = optimisedParams[0]
    crystal_roll = optimisedParams[1]
    camera_pitch = optimisedParams[2]
    camera_roll = optimisedParams[3]
    rcamSphericalOptimised = np.array([optimisedParams[4], r_thetaval, np.pi])

    ncrysOptimised = nVectorFromEuler(crystal_pitch, crystal_roll)
    ncamOptimised = nVectorFromEuler(camera_pitch, camera_roll)
    rcamSphericalOptimised = np.array([optimisedParams[4], r_thetaval, np.pi])
    lossOptimised = lossFunction(optimisedParams)

    def logResults():
        append_to_file(logTextFile, "-" * 30)
        append_to_file(logTextFile, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        append_to_file(logTextFile, f"crystal pitch bounds = {bounds[0]}, crystal roll bounds = {bounds[1]}")
        append_to_file(logTextFile, f"camera pitch bounds = {bounds[2]}, camera roll bounds = {bounds[3]}")
        append_to_file(logTextFile, f"rcamBounds = {bounds[4]}")

        append_to_file(logTextFile, f"optimised crystal pitch = {crystal_pitch}, n crystal roll = {crystal_roll}")
        append_to_file(logTextFile, f"optimised camera pitch = {camera_pitch}, n camera roll = {camera_roll}")
        append_to_file(logTextFile, f"optimised rcam spherical = {rcamSphericalOptimised}")

        append_to_file(logTextFile, f"Optimised n crystal = {ncrysOptimised}")
        append_to_file(logTextFile, f"Optimised n camera = {ncamOptimised}")
        append_to_file(logTextFile, f"Optimised r camera = {rcamSphericalOptimised}")
        append_to_file(logTextFile, f"Loss = {lossOptimised}")

        print("-" * 30)
        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"Optimised n crystal = {ncrysOptimised}")
        print(f"Optimised n camera = {ncamOptimised}")
        print(f"Optimised r camera = {rcamSphericalOptimised}")
        print(f"Loss = {lossOptimised}")

    if logTextFile is not None:
        logResults()

def visualiseQuadParams(list_indexOI):
    vals_dict = {
        "A Left":[],
        "B Left":[],
        "C Left":[],
        "A Right": [],
        "B Right": [],
        "C Right": [],
    }

    for iOI in list_indexOI:
        folderpath_quad = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\stored_variables\ABC_lines\unsupervised_2"
        filename = f"{iOI}"

        vals = np.load(f"{folderpath_quad}/{filename}.npy")
        Aleft = vals[0, 0]
        Bleft = vals[0, 1]
        Cleft = vals[0, 2]
        Aright = vals[1, 0]
        Bright = vals[1, 1]
        Cright = vals[1, 2]

        optimised_params = [Aleft, Bleft, Cleft, Aright, Bright, Cright]

        for idx_param, key in enumerate(vals_dict.keys()):
            vals_dict[key].append(optimised_params[idx_param])

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))

    for idx, (key, ax) in enumerate(zip(vals_dict.keys(), axes)):
        sns.violinplot(data=vals_dict[key],
                       inner="point",
                       color="#00BFFF", linewidth=0, ax=ax)

        sns.stripplot(data=vals_dict[key], color="black", alpha=0.7, ax=ax)
        ax.set_title(f"{key}")
        ax.grid(True)
        ax.tick_params(axis='y', labelsize=5)
        # ax.set_ylabel(f"{key}")

    plt.tight_layout()
    plt.show()

def visualiseGeometryFitParams(list_indexOI):

    vals_dict = {
        "crystal pitch (rad)": [],
        "crystal roll (rad)": [],
        "camera pitch (rad)": [],
        "camera roll (rad)": [],
        "r cam (m)": [], }
    iOI_labels = []

    for iOI in list_indexOI:
        folderpath_geo = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\stored_variables\geometry"
        filename = f"{iOI}"
        optimised_geoParams = np.load(f"{folderpath_geo}/{filename}.npy")

        iOI_labels.append(f"Image {iOI}")

        for idx_param, key in enumerate(vals_dict.keys()):
            vals_dict[key].append(optimised_geoParams[idx_param])

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for idx, (key, ax) in enumerate(zip(vals_dict.keys(), axes)):
        sns.violinplot(data=vals_dict[key],
                       inner="point",
                       color="#00BFFF", linewidth=0, ax=ax)

        sns.stripplot(data=vals_dict[key], color="black", alpha=0.7, ax=ax)
        ax.set_title(f"{key}")
        ax.grid(True)
        ax.tick_params(axis='y', labelsize=5)
        # ax.set_ylabel(f"{key}")

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':

    # visualiseGeometryFitParams(list_data)
    visualiseQuadParams(list_data)

    def fitMinimise(indexOfInterest, useSavedVals):
        crysPitch = -0.3444672207603088
        CrysRoll = 0.018114148603524255
        CamPitch = 0.7950530342947064
        CamRoll = -0.005323879756451509
        rcam = 0.08395021
        thetacam = 2.567

        initialGuess = np.array([crysPitch,  # crystal pitch
                                 CrysRoll,  # crystal roll
                                 CamPitch,  # Camera pitch, pi/4 is ~ 0.785
                                 CamRoll,  # camera roll
                                 rcam  # r camera
                                 ])

        bounds = [(None, None),  # crystal pitch Bounds
                  (None, None),  # crystal roll Bounds
                  (0, None),  # Camera pitch Bounds
                  (None, None),  # camera roll Bounds
                  (0.07, 0.15),  # rcamBounds
                  ]

        optimiseGeometryToCalibratedLines(indexOfInterest, initialGuess, bounds, useSavedVals=useSavedVals, thr=100,
                                          r_thetaval=thetacam, logTextFile=None, saveResults=True,
                                          weight_ofsettedPoints=0.5, iterations=30, )


    # for i in range(len(imData)):
    #     if i not in list_darkData:
    #         fitMinimise(i,True)

    # fitMinimise(8,True)

    def fitCMA():
        crysPitch = -0.3444672207603088
        CrysRoll = 0.018114148603524255
        CamPitch = 0.7950530342947064
        CamRoll = -0.005323879756451509
        rcam = 0.08395021
        thetacam = 2.567
        rcamSpherical = np.array([rcam, thetacam, np.pi])
        initial_guess = np.array([crysPitch,  # crystal pitch
                                  CrysRoll,  # crystal roll
                                  CamPitch,  # Camera pitch, pi/4 is ~ 0.785
                                  CamRoll,  # camera roll
                                  rcam  # r camera
                                  ])
        bounds = [[None, None, None, None, 0.08],  # Lower bounds
                  [None, None, None, None, 0.25]]  # Upper bounds

        optimiseGeometryToCalibratedLibesCMA(initial_guess, bounds, sigma0=0.001, r_thetaval=thetacam,
                                             logTextFile=geometryLog)


    def testPlotGeometryLines(indexOfInterest, thr=100):
        print(indexOfInterest)
        image_mat = loadData()[indexOfInterest]
        image_mat = np.where(image_mat > thr, image_mat, 0)
        imMatVeryClear = imVeryClear(image_mat, 0, (21, 5))

        folderpath_geo = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\stored_variables\geometry"
        filename = f"{indexOfInterest}"
        optimised_geoParams = np.load(f"{folderpath_geo}/{filename}.npy")

        crysPitch = optimised_geoParams[0]
        CrysRoll = optimised_geoParams[1]
        CamPitch = optimised_geoParams[2]
        CamRoll = optimised_geoParams[3]
        rcam = optimised_geoParams[4]
        thetacam = 2.567
        rcamSpherical = np.array([rcam, thetacam, np.pi])

        print("crysPitch = ", crysPitch, "CrysRoll = ", CrysRoll)
        print("CamPitch = ", CamPitch, "CamRoll = ", CamRoll)
        print("rcamSpherical = ", rcamSpherical)

        geo = Geometry(crysPitch, CrysRoll, CamPitch, CamRoll, rcamSpherical)
        geolinesMat = geo.createLinesMatrix(imTest, np.max(im_very_clear), phiStepSize=0.0001)

        def get_quadLinesMat():
            folderpath_quad = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\stored_variables\ABC_lines\unsupervised"
            filename_quad = f"{indexOfInterest}"
            vals = np.load(f"{folderpath_quad}/{filename_quad}.npy")
            Aleft = vals[0, 0]
            Bleft = vals[0, 1]
            Cleft = vals[0, 2]
            Aright = vals[1, 0]
            Bright = vals[1, 1]
            Cright = vals[1, 2]

            linesMatLeft = Calibrate(image_mat, None).matrixWithLines(Aoptimised=Aleft, Boptimised=Bleft,
                                                                      Coptimised=Cleft, plotLines=False)
            linesMatRight = Calibrate(image_mat, None).matrixWithLines(Aoptimised=Aright, Boptimised=Bright,
                                                                       Coptimised=Cright, plotLines=False)

            return linesMatLeft, linesMatRight

        plt.imshow(imMatVeryClear + geolinesMat, cmap="hot")
        title_l1 = f"Lα and Lβ lines for image {indexOfInterest}"
        title_l2 = f"\ncrystal: Pitch = {crysPitch}, Roll = {CrysRoll}"
        title_l3 = f"\ncamera: Pitch = {CamPitch}, Roll = {CamRoll}"
        title_l4 = f"\nr_camera = {rcam}"
        plt.title(title_l1 + title_l2 + title_l3 + title_l4)
        plt.show()


    # testPlotGeometryLines(1)

    def testPlotQuadLines(indexOfInterest, thr=100):
        print(indexOfInterest)
        image_mat = loadData()[indexOfInterest]
        image_mat = np.where(image_mat > thr, image_mat, 0)
        imMatVeryClear = imVeryClear(image_mat, 0, (21, 5))
        mat_plot = image_mat

        folderpath_quad = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\stored_variables\ABC_lines\unsupervised"
        filename = f"{indexOfInterest}"
        vals = np.load(f"{folderpath_quad}/{filename}.npy")
        Aleft = vals[0, 0]
        Bleft = vals[0, 1]
        Cleft = vals[0, 2]
        Aright = vals[1, 0]
        Bright = vals[1, 1]
        Cright = vals[1, 2]

        mat_quadLeft = Calibrate(image_mat, None).matrixWithLines(Aoptimised=Aleft, Boptimised=Bleft,
                                                                  Coptimised=Cleft, plotLines=False)
        mat_quadRight = Calibrate(image_mat, None).matrixWithLines(Aoptimised=Aright, Boptimised=Bright,
                                                                   Coptimised=Cright, plotLines=False)

        val_line = np.max(mat_plot) / 4

        mat_quadLeft = np.where(mat_quadLeft > 0, val_line, 0)
        mat_quadRight = np.where(mat_quadRight > 0, val_line, 0)

        plt.imshow(mat_plot + mat_quadLeft + mat_quadRight, cmap="hot")
        titleL1 = f"Image {indexOfInterest} with saved A(y-B)**2 + C coefficients"
        titleL2 = f"\nLeft: A={Aleft:.2f}, B={Bleft:.2f}, C={Cleft:.2f}"
        titleL3 = f"\nRight: A={Aright:.2f}, B={Bright:.2f}, C={Cright:.2f}"
        plt.title(titleL1 + titleL2 + titleL3)
        plt.show()


    # testPlotQuadLines(1)

    def showTwoLines(index_of_interest, thr=90, ):
        ImMat = imData[index_of_interest]
        ImMat = np.where(ImMat > thr, ImMat, 0)

        imMatVeryClear = imVeryClear(ImMat, 0, (21, 5))
        # cal = Calibrate(imTest, None)
        # cal.optimiseLines(aBounds=[0,1e-04],bBounds=[861,863],cBounds=[1100,1400],plotGraph=True,plotResults=True)

        linesMatLeft = Calibrate(imTest, None).matrixWithLines(Aoptimised=6.613794078473409e-05, Boptimised=862,
                                                               Coptimised=1278, plotLines=False)
        LinesMatLeft = np.where(linesMatLeft > 0, thr, 0)
        linesMatRight = Calibrate(imTest, None).matrixWithLines(Aoptimised=7.063102423636962e-05, Boptimised=862,
                                                                Coptimised=1418, plotLines=False)
        LinesMatRight = np.where(linesMatRight > 0, thr, 0)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.imshow(ImMat, cmap='hot'), plt.title(
            f"Image {index_of_interest} thresholded above {thr}")
        plt.subplot(1, 2, 2), plt.imshow(imMatVeryClear + LinesMatLeft + LinesMatRight, cmap='hot'), plt.title(
            f"Average Pooled Image with geometric lines")
        plt.show()


    # showTwoLines(1)

    # calibrateSaveQuadratics(8,pltGauss=False,plot_finalLines=True)

    # for i in range(len(imData)):
    #     if i not in list_darkData:
    #         calibrateSaveQuadratics(i,save=True,
    #                                 logFile=r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\stored_variables\ABC_lines\unsupervised_2\quad_line_fits.txt",
    #                                 folderpath_to_save=r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\stored_variables\ABC_lines\unsupervised_2")

    def testBragg():
        crysPitch = -0.3444672207603088
        CrysRoll = 0.018114148603524255
        CamPitch = 0.7950530342947064
        CamRoll = -0.005323879756451509
        rcam = 0.08395021
        thetacam = 2.567
        rcamSpherical = np.array([rcam, thetacam, np.pi])

        bragg = Bragg(crystal_pitch=crysPitch, crystal_roll=CrysRoll,
                      camera_pitch=CamPitch, camera_roll=CamRoll,
                      r_camera_spherical=rcamSpherical)

        # geoCheck = Geometry(crystal_pitch=crysPitch,crystal_roll=CrysRoll,
        #               camera_pitch=CamPitch, camera_roll=CamRoll,
        #               r_camera_spherical=rcamSpherical)
        # geoCheck.xy_coords_of_EAlphaBeta()

        xpixel_fromCenter = 404
        ypixel_fromCenter = 519

        eVal = bragg.xyImagePlane_to_energy(xpixel_fromCenter * pixel_width, ypixel_fromCenter * pixel_width)

        print(eVal)


    # testBragg()

    pass
