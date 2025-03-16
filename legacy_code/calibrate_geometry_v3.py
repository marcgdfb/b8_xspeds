from scipy.optimize import minimize, curve_fit
from imagePreProcessing import *
from pedestal_engine_v2 import *
from tools import *
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import os
import json

geometryLog = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\data_logs\geometryFitLog.txt"
quadLineLog = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\data_logs\quadraticLineFits.txt"



class Geometry:
    def __init__(self, crystal_pitch, crystal_roll, camera_pitch, camera_roll,
                 r_cam, r_theta=2.567, xpixels=2048, ypixels=2048, pixelWidth=pixel_width, ):
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

        self.r_cam = r_cam
        self.r_camera_spherical = np.array([r_cam, r_theta, np.pi])
        self.xpixels = xpixels
        self.ypixels = ypixels
        self.pixelWidth = pixelWidth
        self.xWidth = xpixels * pixelWidth
        self.yWidth = ypixels * pixelWidth
        self.r_cam_cart = spherical_to_cartesian(self.r_camera_spherical)

    def xy_coords_of_E(self, energy_eV, phiStepSize=0.0001,
                       findMaxPhi=False):
        """
        :param energy_eV: Energy in eV of photon for which we are producing the cone of directions allowed by the bragg
        crystal
        :param phiStepSize: Step size over which phi is iterated, smaller will run quicker but not produce the whole line.
        This is not optimised
        :param findMaxPhi: If true the largest phi value on the CCD will be found and printed
        :return: A list containing further lists of [x_meters, y_meters]
        """

        # Rotation matrix from crystal 001 to n crystal
        rotMatrix_crys = rotMatrixUsingEuler(self.crystal_pitch, self.crystal_roll)

        # angle between crystal plane and ray
        theta_E = bragg_E_to_theta(energy_eV)
        # polar coordinate of this angle (still as if crystal is 001)
        theta_E_polar = np.pi / 2 + theta_E

        list_xy = []

        if findMaxPhi:
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

                    if findMaxPhi:

                        phi_difference_to_pi = abs(abs(v_ray_spherical[2]) - np.pi)

                        if phi_difference_to_pi > phiDif:
                            phiDif = phi_difference_to_pi

        if findMaxPhi:
            print(f"The largest phi deviation from pi was {phiDif}")

        if not list_xy:
            print(f"No values for E = {energy_eV}")

        return list_xy

    def xy_pixelCoords_of_E(self, energy_eV, phiStepSize=0.0001):
        list_xy_coords = self.xy_coords_of_E(energy_eV, phiStepSize)
        list_xy_pixel_coords = []

        for row in list_xy_coords:
            # E.G. row[0] = 3.5 pixel widths (i.e. center of 4th away from center)
            # We want if x is within 3-4 pixel widths that this is in the 4th pixel

            x_pixel, y_pixel = xy_meters_to_xyPixel(x_meters=row[0], y_meters=row[1])

            # Check if any existing coordinate has the same y_pixel value
            if not any(y == y_pixel for _, y in list_xy_pixel_coords):
                list_xy_pixel_coords.append([x_pixel, y_pixel])

        return list_xy_pixel_coords

    def xy_pixelCoords_of_E_old(self, energy_eV, phiStepSize=0.0001):
        list_xy_coords = self.xy_coords_of_E(energy_eV, phiStepSize)
        list_xy_pixel_coords = []

        for row in list_xy_coords:
            # E.G. row[0] = 3.5 pixel widths (i.e. center of 4th away from center)
            # We want if x is within 3-4 pixel widths that this is in the 4th pixel

            x_pixel, y_pixel = xy_meters_to_xyPixel(x_meters=row[0], y_meters=row[1])

            list_xy_pixel_coords.append([x_pixel, y_pixel])

        return list_xy_pixel_coords

    def createLinesMatrix(self, imageMat, valMax, phiStepSize=0.0001):

        matrixLines = np.zeros((imageMat.shape[0], imageMat.shape[1]))

        for E in [E_Lalpha_eV, E_Lbeta_eV]:
            xy_E_list_pixel = self.xy_pixelCoords_of_E(E, phiStepSize)

            for row in xy_E_list_pixel:
                x_pixel = row[0]
                y_pixel = row[1]

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
    def __init__(self, imageMatrix, logTextFile=None, adjacentWeight=1.0, width_lineIntegral_5=False):
        self.imMat = imageMatrix
        self.log = logTextFile
        self.adjacentWeight = adjacentWeight
        self.width_lineIntegral_5 = width_lineIntegral_5

    # The following code serves to compute quadratic curves that describe our lines of interest
    def computeLine(self, a, b, cBounds, plotGraph=False, cPlotVal=1450, plotResults=False, ):
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
                            totVal += self.adjacentWeight * self.imMat[yL, xL + 1]
                        if 0 < xL - 1:
                            totVal += self.adjacentWeight * self.imMat[yL, xL - 1]

                    if self.width_lineIntegral_5:
                        if xL + 2 < xWidth:
                            totVal += self.adjacentWeight / 2 * self.imMat[yL, xL + 2]
                        if 0 < xL - 2:
                            totVal += self.adjacentWeight / 2 * self.imMat[yL, xL - 2]

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

    # noinspection PyTupleAssignmentBalance
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

        count_runtimeError = 0

        lineIntegralList = self.computeLine(a, b, cBounds)
        cVals = np.array(lineIntegralList)[:, 0]
        lineIntegralVals = np.array(lineIntegralList)[:, 1]

        def gaussian(X, amp_, sigma_, xpeak, c_offset):
            return amp_ * np.exp(-(X - xpeak) ** 2 / (2 * sigma_ ** 2)) + c_offset

        amp_guess = np.max(lineIntegralVals)
        xpeak_guess = cVals[np.argmax(lineIntegralVals)]
        sigma_guess = 20
        c_offset_guess = np.min(lineIntegralVals)

        try:
            amp_sigma_cPeak, pcov = curve_fit(gaussian, cVals, lineIntegralVals,
                                              p0=[amp_guess, xpeak_guess, sigma_guess, c_offset_guess], maxfev=2000)
            if plotGauss:
                plt.plot(cVals, lineIntegralVals)
                plt.plot(cVals, gaussian(cVals, *amp_sigma_cPeak))
                plt.ylabel("Line Integral with width 3 pixels")
                plt.xlabel("C value in x = A(y-B)**2 + C")
                plt.show()

            amp_sigma_cPeak_errors = np.sqrt(np.diag(pcov))

            return amp_sigma_cPeak, amp_sigma_cPeak_errors
        except RuntimeError:
            print(RuntimeError)
            count_runtimeError += 1
            if count_runtimeError > 10:
                raise RuntimeError("Too many RuntimeErrors, stopping function execution.")

            else:
                print("The fit could not be computed. Retrying with a larger range of C")

                clower = cBounds[0]
                cHigher = cBounds[1]
                c_dif = (cHigher - clower) / 4

                cBoundsNew = (int(clower - c_dif), int(cHigher + c_dif))

                print("New C bounds: ", cBoundsNew)

                self.fitGaussianToLineIntegral(a, b, cBoundsNew, plotGauss)

    def optimiseLines(self, aBounds, bBounds, cBounds, sigmaWeighting=1, plotGraph=False,
                      plotGauss=False):
        """
        Using fitGaussianToLineIntegral we will now optimise to encourage a maximal peak with a minimum sigma.
        Recall the form X = A * (Y - B) ** 2 + C
        :param aBounds: bounds of A
        :param bBounds: bounds of B
        :param cBounds: bounds of C
        :param sigmaWeighting: A loss function of -amp + sigmaWeighting * sigma is used
        :param plotGraph: plot the curved lines matrix
        :param plotGauss: Plot the gaussian curve and fitting for the final data
        :return:
        """

        def lossFunc(params, sigmaWeighting_):

            print("-" * 40)
            Aval, Bval = params

            amp_sigma_cPeak_, amp_sigma_cPeak_errors_ = self.fitGaussianToLineIntegral(Aval, Bval, cBounds)

            print(
                f"A = {Aval}, B = {Bval}, amplitude = {amp_sigma_cPeak_[0]},sigma = {amp_sigma_cPeak_[1]},cPeak = {amp_sigma_cPeak_[2]}")
            # Aiming to maximise amp while minimising sigma
            print("Loss", -amp_sigma_cPeak_[0] + sigmaWeighting_ * amp_sigma_cPeak_[1])
            return -amp_sigma_cPeak_[0] + sigmaWeighting_ * amp_sigma_cPeak_[1]

        bounds = [aBounds, bBounds]
        initial_guess = np.array([(aBounds[0] + aBounds[1]) / 2, (bBounds[0] + bBounds[1]) / 2])

        # result = differential_evolution(lossFunc, bounds, args=(sigmaWeighting,))

        result = minimize(lossFunc, initial_guess, args=(sigmaWeighting,), bounds=bounds, method='Nelder-Mead',
                          callback=callbackminimise, options={'maxiter': 30})

        A, B = result.x
        lossOptimised = lossFunc(result.x, sigmaWeighting)

        amp_sigma_cPeak, amp_sigma_cPeak_errors = self.fitGaussianToLineIntegral(A, B, cBounds, plotGauss=plotGauss)
        cPeak = amp_sigma_cPeak[2]
        cPeak_unc = amp_sigma_cPeak_errors[2]

        def logResults():
            append_to_file(self.log, "-" * 30)
            append_to_file(self.log, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            append_to_file(self.log,
                           f"aBounds = {aBounds}" + "\n" + f"bBounds = {bBounds}" + "\n" + f"cBounds = {cBounds}")
            append_to_file(self.log, f"Optimised A = {A}, B = {B}, C = {cPeak} +- {cPeak_unc}")
            append_to_file(self.log, f"Loss = {lossOptimised}")

            print("-" * 30)
            print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"aBounds = {aBounds}" + "\n" + f"bBounds = {bBounds}" + "\n" + f"cBounds = {cBounds}")
            print(f"Optimised A = {A}, B = {B}, C = {cPeak} +- {cPeak_unc}")
            print(f"Loss = {lossOptimised}")

        if self.log is not None:
            logResults()

        if plotGraph:
            self.computeLine(A, B, cBounds, plotGraph=True, cPlotVal=cPeak, plotResults=True)

        return A, B, cPeak, cPeak_unc

    def matrixWithLinesOptimisation(self, plotLines=False, returnABC=False, plotGauss=False,
                                    cBoundsLeft=(1220, 1340), aBoundsLeft=(0.00001, 0.0001), bBoundsLeft=(800, 950),
                                    cBoundsRight=(1380, 1460), aBoundsRight=(0.00001, 0.0001), bBoundsRight=(800, 950),
                                    ):

        # The following ranges are for image 8. They are expected to work for any image with data
        # The Line ends further along x at the bottom than it does y

        append_to_file(self.log, "-" * 30)
        append_to_file(self.log, "Left Line")
        print("Working on the left Line")
        ALeft, BLeft, cLeft, cLeft_Peak_unc = self.optimiseLines(aBoundsLeft, bBoundsLeft, cBoundsLeft, plotGraph=False,
                                                                 plotGauss=plotGauss)
        append_to_file(self.log, "-" * 30)
        append_to_file(self.log, "Right Line")
        print("Working on the right Line")
        ARight, BRight, cRight, cRight_Peak_unc = self.optimiseLines(aBoundsRight, bBoundsRight, cBoundsRight,
                                                                     plotGraph=False, plotGauss=plotGauss)

        leftLineMat = self.matrixWithLines(ALeft, BLeft, cLeft, plotLines=False)
        rightLineMat = self.matrixWithLines(ARight, BRight, cRight, plotLines=False)

        if plotLines:
            plt.imshow(leftLineMat + rightLineMat, cmap='hot')
            plt.show()

        if returnABC:
            left_vars = [ALeft, BLeft, cLeft, cLeft_Peak_unc]
            right_vars = [ARight, BRight, cRight, cRight_Peak_unc]
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

        # The x_imPlane,y_imPlane are those within the plane of the CCD.
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

    def xyPixelImagePlane_to_energy(self, xPixel_imPlane, yPixel_imPlane, ):
        x_coord, y_coord = xyPixel_to_xyMeters(xPixel_imPlane, yPixel_imPlane)

        return self.xyImagePlane_to_energy(x_coord, y_coord)


class SolidAngle:
    def __init__(self, geo_engine=None):
        if geo_engine is not None:
            self.geo_engine = geo_engine
        else:
            self.geo_engine = geo_engine_withSavedParams()

    def d_omega_dS(self, x_inPlane_meters, y_inPlane_meters):
        """
        Solid angle in spherical polar coordinates is r_hat dot n_area_hat * d_area / r^2
        :param x_inPlane_meters:
        :param y_inPlane_meters:
        :return:
        """

        # Ensuring normalised normal to the camera
        n_cam = self.geo_engine.nCam / np.linalg.norm(self.geo_engine.nCam)
        # r vector to the center of the camera
        r_cam = self.geo_engine.r_cam_cart

        r_inPlane_prime = np.array([x_inPlane_meters, y_inPlane_meters, 0])
        # The rotation matrix to convert the primed vector to normal coordinates
        rotMat_camera = rotMatrixUsingEuler(pitch_rad=self.geo_engine.camera_pitch,
                                            roll_rad=self.geo_engine.camera_roll)
        r_inplane = np.dot(rotMat_camera, r_inPlane_prime)

        r = r_cam + r_inplane
        r_magnitude = np.linalg.norm(r)
        r_hat = r / r_magnitude

        return abs(np.dot(r_hat, n_cam) / (r_magnitude ** 2))

    def integrate_d_omega(self, x_min, x_max, y_min, y_max, number_points=3):

        # The maximum points are the bounds of the sqaure meaning if we want 3 points between these
        # => 0-1 should have 3 squares with lengh dx = dy = 1/3 and centers at

        dx = (x_max - x_min) / number_points
        x_points = np.linspace(x_min + dx / 2, x_max - dx / 2, number_points)
        dy = (y_max - y_min) / number_points
        y_points = np.linspace(y_min + dy / 2, y_max - dy / 2, number_points)

        integral = 0
        for x in x_points:
            for y in y_points:
                integral += self.d_omega_dS(x, y) * dx * dy

        return integral

    def solidAngle_pixelij(self, i_idx, j_idx, pixelWidth=pixel_width, number_points=100):

        xMeters, yMeters = xyPixel_to_xyMeters(x_pixel=j_idx, y_pixel=i_idx, )

        xMax = xMeters + pixelWidth / 2
        xMin = xMeters - pixelWidth / 2
        yMax = yMeters + pixelWidth / 2
        yMin = yMeters - pixelWidth / 2

        return self.integrate_d_omega(xMin, xMax, yMin, yMax, number_points=number_points)


# ---------- Calibration and saving of parameters ----------

def mat_thr_aboveNsigma(index_of_interest, how_many_sigma, ):
    image_mat = loadData()[index_of_interest]
    # ----------pedestal mean and sigma----------
    ped_mean, ped_sigma = pedestal_mean_sigma_awayFromLines(image_mat, index_of_interest)
    thr = ped_mean + how_many_sigma * ped_sigma

    image_mat = np.where(image_mat > thr, image_mat, 0)

    return image_mat, thr


# ---------- Quadratic:

def calibrate_and_save_quadratics(indexOfInterest,
                                  howManySigma=2, adjacent_pixel_weighting=0.5, pixelwidth_lineintegral_5=True,
                                  bBounds=None,
                                  folderpath="stored_variables", saveData=True,
                                  plot_Results=False,
                                  ):
    # initialise subfolder if it doesn't exist
    index_folder = os.path.join(folderpath, str(indexOfInterest))
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    logFile = os.path.join(index_folder, "quadratic_fits_log.txt")

    print("Index: ", indexOfInterest)

    # ----------pedestal mean and sigma----------
    image_mat, thr = mat_minusMean_thr_aboveNsigma(indexOfInterest, howManySigma)
    imMatVeryClear = imVeryClear(image_mat, 0, (21, 5))

    # ---------- Initialise Calibrate Engine and log the initialising parameters ----------

    cal = Calibrate(image_mat, logFile, adjacentWeight=adjacent_pixel_weighting,
                    width_lineIntegral_5=pixelwidth_lineintegral_5)
    append_to_file(cal.log, "-" * 30)
    append_to_file(cal.log, f"Index of Interest {indexOfInterest}, thresholded above {howManySigma} sigma")
    append_to_file(cal.log, f"Adjacent pixels weighted by {adjacent_pixel_weighting}")
    if pixelwidth_lineintegral_5:
        append_to_file(cal.log, "Line integral used pixel width of 5")
    else:
        append_to_file(cal.log, "Line integral used pixel width of 3")

    # Note the values of A,B,C are appended to the log file within the following functions:
    if bBounds is None:
        leftLineMat, rightLineMat, left_vars, right_vars = cal.matrixWithLinesOptimisation(False, returnABC=True,
                                                                                           plotGauss=plot_Results,
                                                                                           bBoundsLeft=(800, 950),
                                                                                           bBoundsRight=(800, 950), )
    else:
        leftLineMat, rightLineMat, left_vars, right_vars = cal.matrixWithLinesOptimisation(False, returnABC=True,
                                                                                           plotGauss=plot_Results,
                                                                                           bBoundsLeft=bBounds,
                                                                                           bBoundsRight=bBounds)

    LinesMatLeft = np.where(leftLineMat > 0, thr, 0)
    LinesMatRight = np.where(rightLineMat > 0, thr, 0)


    if saveData:
        filepath = os.path.join(index_folder, "quadratic_fits.npy")
        quad_vars = np.array(left_vars + right_vars)
        np.save(filepath, quad_vars)


    if plot_Results:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.imshow(image_mat, cmap='hot'), plt.title(
            f"Image {indexOfInterest} thresholded above {thr}")
        plt.subplot(1, 2, 2), plt.imshow(imMatVeryClear + LinesMatLeft + LinesMatRight, cmap='hot'), plt.title(
            f"Average Pooled Image with geometric lines")
        plt.show()


def access_saved_quadratics(indexOfInterest, folderpath="stored_variables"):
    index_folder = os.path.join(folderpath, str(indexOfInterest))
    filepath = os.path.join(index_folder, "quadratic_fits.npy")

    saved_variables = np.load(filepath)

    left_vars = saved_variables[0:4]
    print(left_vars)
    right_vars = saved_variables[4:8]
    print(right_vars)

    return left_vars, right_vars


def testPlotQuadLines(indexOfInterest, how_many_sigma=2, plot_gauss=True, ):
    left_vars, right_vars = access_saved_quadratics(indexOfInterest)
    Aleft = left_vars[0]
    Bleft = left_vars[1]
    Cleft = left_vars[2]

    Aright = right_vars[0]
    Bright = right_vars[1]
    Cright = right_vars[2]

    print(indexOfInterest)
    image_mat, thr = mat_thr_aboveNsigma(indexOfInterest, how_many_sigma)
    mat_plot = image_mat

    calibrate_tpql = Calibrate(image_mat, None, adjacentWeight=0.5, width_lineIntegral_5=True)
    if plot_gauss:
        calibrate_tpql.fitGaussianToLineIntegral(a=Aleft, b=Bleft, cBounds=(Cleft - 60, Cleft + 60),
                                                 plotGauss=True)
        calibrate_tpql.fitGaussianToLineIntegral(a=Aright, b=Bright, cBounds=(Cright - 60, Cright + 60),
                                                 plotGauss=True, )

    mat_quadLeft = calibrate_tpql.matrixWithLines(Aoptimised=Aleft, Boptimised=Bleft,
                                                  Coptimised=Cleft, plotLines=False)
    mat_quadRight = calibrate_tpql.matrixWithLines(Aoptimised=Aright, Boptimised=Bright,
                                                   Coptimised=Cright, plotLines=False, )

    val_line = np.max(mat_plot) / 4

    mat_quadLeft = np.where(mat_quadLeft > 0, val_line, 0)
    mat_quadRight = np.where(mat_quadRight > 0, val_line, 0)

    # plt.imshow(mat_plot + mat_quadLeft + mat_quadRight, cmap="hot")
    plt.imshow(mat_quadLeft + mat_quadRight, cmap="hot")
    titleL1 = f"Image {indexOfInterest} with saved A(y-B)**2 + C coefficients"
    titleL2 = f"\nLeft: A={Aleft:.2f}, B={Bleft:.2f}, C={Cleft:.2f}"
    titleL3 = f"\nRight: A={Aright:.2f}, B={Bright:.2f}, C={Cright:.2f}"
    plt.title(titleL1 + titleL2 + titleL3)
    plt.show()


# ---------- Geometric:


def optimiseGeometryToCalibratedLines(indexOfInterest,
                                      initialGuess, bounds, r_thetaval=2.567,
                                      folderpath="stored_variables", useSavedData=True, saveData=True,
                                      how_many_sigma=2,
                                      weight_ofsettedPoints=0.5, iterations=30, plot=False):
    # Ensuring the folder is initialised
    index_folder = os.path.join(folderpath, str(indexOfInterest))
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)
    logTextFile = os.path.join(index_folder, "geometric_fits_log.txt")

    def logStart():
        log_ = Append_to_file(logTextFile)
        log_.append("-" * 30)
        log_.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        log_.append(f"crystal pitch bounds = {bounds[0]}, crystal roll bounds = {bounds[1]}")
        log_.append(f"camera pitch bounds = {bounds[2]}, camera roll bounds = {bounds[3]}")
        log_.append(f"rcamBounds = {bounds[4]}")
        log_.append(f"Initial Guess = {initialGuess}")
        log_.append(f"r_thetaval = {r_thetaval}")
        log_.append(f"how many sigma = {how_many_sigma}")
        log_.append(f"weight_ofsettedPoints = {weight_ofsettedPoints}")
        log_.append(f"iterations = {iterations}")

    logStart()

    print(indexOfInterest)
    image_mat = loadData()[indexOfInterest]

    # Getting threshold using pedetsal engine
    ped_mean, ped_sigma = pedestal_mean_sigma_awayFromLines(image_mat, indexOfInterest)
    thr = ped_mean + how_many_sigma * ped_sigma

    image_mat = np.where(image_mat > thr, image_mat, 0)
    # imMatVeryClear = imVeryClear(image_mat, 0, (21, 5))

    if useSavedData:

        left_vars_, right_vars_ = access_saved_quadratics(indexOfInterest, folderpath)

        print("Using Saved Values")
        print("Left Line:")
        print(f"Aleft: {left_vars_[0]}, Bleft: {left_vars_[1]}, Cleft: {left_vars_[2]} +- {left_vars_[3]}")
        print("Right Line:")
        print(f"Aright: {right_vars_[0]}, Bright: {right_vars_[1]}, Cright: {right_vars_[2]} +- {right_vars_[3]}")

        linesMatLeft = Calibrate(image_mat, None, adjacentWeight=0.5, width_lineIntegral_5=True).matrixWithLines(
            Aoptimised=left_vars_[0], Boptimised=left_vars_[1],
            Coptimised=left_vars_[2], plotLines=False)
        linesMatRight = Calibrate(image_mat, None, adjacentWeight=0.5, width_lineIntegral_5=True).matrixWithLines(
            Aoptimised=right_vars_[0], Boptimised=right_vars_[1],
            Coptimised=right_vars_[2], plotLines=False)

    else:
        cal = Calibrate(image_mat, quadLineLog, adjacentWeight=0.5, width_lineIntegral_5=True)
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

    def lossFunction(params):
        p = params

        geo = Geometry(crystal_pitch=p[0], crystal_roll=p[1],
                       camera_pitch=p[2], camera_roll=p[3],
                       r_cam=p[4], r_theta=r_thetaval, )

        alphaLineCoords_pixel = geo.xy_pixelCoords_of_E(E_Lalpha_eV)  # More Right / right line
        betaLineCoords_pixel = geo.xy_pixelCoords_of_E(E_Lbeta_eV)  # More Left / left line


        def computeLossOld():
            lossPositive = 0
            # Treat each Line Separately as not to over encourage curvature
            for row in alphaLineCoords_pixel:
                x_pixel = row[0]
                y_pixel = row[1]

                lossPositive += linesMatRight[y_pixel, x_pixel]

                if x_pixel + 1 < geo.xpixels:
                    lossPositive += weight_ofsettedPoints * linesMatRight[y_pixel, x_pixel + 1]
                if x_pixel - 1 >= 0:
                    lossPositive += weight_ofsettedPoints * linesMatRight[y_pixel, x_pixel - 1]

            for row in betaLineCoords_pixel:
                x_pixel = row[0]
                y_pixel = row[1]

                lossPositive += linesMatLeft[y_pixel, x_pixel]

                if x_pixel + 1 < geo.xWidth:
                    lossPositive += weight_ofsettedPoints * linesMatLeft[y_pixel, x_pixel + 1]
                if x_pixel - 1 >= 0:
                    lossPositive += weight_ofsettedPoints * linesMatLeft[y_pixel, x_pixel - 1]

            return -lossPositive

        def computeLossNew():
            loss_to_be_minimised = 0

            left = [betaLineCoords_pixel,linesMatLeft]
            right = [alphaLineCoords_pixel,linesMatRight]

            for pixel_mat_list in [left,right]:
                geometric_line_pixels = pixel_mat_list[0]
                linesMat = pixel_mat_list[1]

                for row in geometric_line_pixels:
                    x_pixel = row[0]
                    y_pixel = row[1]

                    # now we want to find the pixel distance
                    # Finding the indices in the given row that are non-zero
                    nonzero_indices = np.nonzero(linesMat[y_pixel])[0]
                    mean_position = np.mean(nonzero_indices)

                    difference = abs(x_pixel-mean_position)

                    # print("y_pixel", y_pixel, "x_pixel", x_pixel)
                    # print("nonzero_indices", nonzero_indices)
                    # print("mean_position", mean_position)
                    # print(difference)


                    loss_to_be_minimised += difference

            return loss_to_be_minimised

        # We wish to maximise the integral along these lines
        loss = computeLossNew()

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
                            r_cam=optimisedParams[4], r_theta=r_thetaval, )
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

    if saveData:
        filepath = os.path.join(index_folder, "geometric_fits.npy")
        geometric_vars = np.array(optimisedParams)
        np.save(filepath, geometric_vars)

    return lossOptimised


def geometry_fitMinimise(indexOfInterest, useSavedVals=True, iterations_=50, weight_ofsettedPoints=0.5):
    crysPitch = -0.3445
    CrysRoll = 0.0184
    CamPitch = 0.814
    CamRoll = -0.00537
    rcam = 0.0839

    initialGuess = np.array([crysPitch,  # crystal pitch
                             CrysRoll,  # crystal roll
                             CamPitch,  # Camera pitch, pi/4 is ~ 0.785
                             CamRoll,  # camera roll
                             rcam  # r camera
                             ])

    bounds = [(-0.346, -0.343),  # crystal pitch Bounds
              (None, None),  # crystal roll Bounds
              (0.7, 0.9),  # Camera pitch Bounds
              (None, None),  # camera roll Bounds
              (0.0820, 0.0860),  # rcamBounds
              ]

    optimisedLoss = optimiseGeometryToCalibratedLines(indexOfInterest,
                                                      initialGuess, bounds, r_thetaval=2.567,
                                                      folderpath="stored_variables", useSavedData=useSavedVals,
                                                      saveData=True,
                                                      how_many_sigma=2,
                                                      weight_ofsettedPoints=weight_ofsettedPoints,
                                                      iterations=iterations_, plot=False)

    if optimisedLoss < 4000:
        print("loss is not sufficiently high, Consider trying again with more iterations")



def access_saved_geometric(indexOfInterest, folderpath="stored_variables"):
    index_folder = os.path.join(folderpath, str(indexOfInterest))
    filepath = os.path.join(index_folder, "geometric_fits.npy")

    saved_variables = np.load(filepath)

    crys_pitch = saved_variables[0]
    crys_roll = saved_variables[1]
    cam_pitch = saved_variables[2]
    cam_roll = saved_variables[3]
    r_cam = saved_variables[4]

    return crys_pitch, crys_roll, cam_pitch, cam_roll, r_cam


def testPlotGeometryLines(indexOfInterest, how_many_sigma=2, r_theta=2.567):
    print(indexOfInterest)

    image_mat, thr = mat_thr_aboveNsigma(indexOfInterest, how_many_sigma)
    imMatVeryClear = imVeryClear(image_mat, 0, (21, 5))

    crysPitch, CrysRoll, CamPitch, CamRoll, rcam = access_saved_geometric(indexOfInterest)

    rcamSpherical = np.array([rcam, r_theta, np.pi])

    print("crysPitch = ", crysPitch, "CrysRoll = ", CrysRoll)
    print("CamPitch = ", CamPitch, "CamRoll = ", CamRoll)
    print("rcamSpherical = ", rcamSpherical)

    geo = Geometry(crysPitch, CrysRoll, CamPitch, CamRoll, r_cam=rcam, r_theta=r_theta)
    geolinesMat = geo.createLinesMatrix(imTest, np.max(im_very_clear), phiStepSize=0.0001)

    def get_quadLinesMat():
        left_vars, right_vars = access_saved_quadratics(indexOfInterest)
        Aleft = left_vars[0]
        Bleft = left_vars[1]
        Cleft = left_vars[2]

        Aright = right_vars[0]
        Bright = right_vars[1]
        Cright = right_vars[2]

        linesMatLeft = Calibrate(image_mat, None, adjacentWeight=0.5, width_lineIntegral_5=True).matrixWithLines(
            Aoptimised=Aleft, Boptimised=Bleft,
            Coptimised=Cleft, plotLines=False)
        linesMatRight = Calibrate(image_mat, None, adjacentWeight=0.5, width_lineIntegral_5=True).matrixWithLines(
            Aoptimised=Aright, Boptimised=Bright,
            Coptimised=Cright, plotLines=False, )

        return np.where(linesMatLeft > 0, np.max(im_very_clear) / 2, 0), np.where(linesMatRight > 0,
                                                                                  np.max(im_very_clear) / 2, 0)

    matLeft, matRight = get_quadLinesMat()

    plt.imshow(imMatVeryClear + geolinesMat, cmap="hot")
    # plt.imshow(geolinesMat, cmap="hot")
    title_l1 = f"Lα and Lβ lines for image {indexOfInterest}"
    title_l2 = f"\ncrystal: Pitch = {crysPitch}, Roll = {CrysRoll}"
    title_l3 = f"\ncamera: Pitch = {CamPitch}, Roll = {CamRoll}"
    title_l4 = f"\nr_camera = {rcam}"
    plt.title(title_l1 + title_l2 + title_l3 + title_l4)
    plt.show()

    plt.imshow(matLeft + matRight + geolinesMat, cmap="jet")
    plt.title("")
    plt.show()


# ---------- Visualisations of Params
# Each crystal must be accounted for individually to avoid x-ray jitter error

def violinPlot_QuadParams(list_indexOI, fp_excel="stored_variables/quadratic_fits.xlsx"):
    vals_dict = {
        "A Left": [],
        "B Left": [],
        "C Left": [],
        "A Right": [],
        "B Right": [],
        "C Right": [],
    }

    for iOI in list_indexOI:

        Aleft, Bleft, Cleft, Cleft_unc, Aright, Bright, Cright, Cright_unc = access_saved_quadratics(iOI, fp_excel)

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



def geo_engine_withSavedParams(index_oI, printVals=False):
    crys_pitch, crys_roll, cam_pitch, cam_roll, r_cam = access_saved_geometric(index_oI)

    geo_engine = Geometry(crystal_pitch=crys_pitch, crystal_roll=crys_roll,
                          camera_pitch=cam_pitch, camera_roll=cam_roll,
                          r_cam=r_cam, )

    return geo_engine


if __name__ == '__main__':

    # visualiseGeometryFitParams(list_data)
    # visualiseQuadParams(list_data)

    def plotBinWidthLines(bin_width, index_of_interest):

        crys_pitch, crys_roll, cam_pitch, cam_roll, r_cam = access_saved_geometric(index_of_interest)

        energyBins = np.arange(1100, 1600 + bin_width, bin_width)

        geo_engine = Geometry(crystal_pitch=crys_pitch, crystal_roll=crys_roll,
                              camera_pitch=cam_pitch, camera_roll=cam_roll,
                              r_cam=r_cam, )

        fig, ax = plt.subplots(figsize=(8, 8))

        # intialising the colours of the lines to better visibility
        colors = itertools.cycle(['b', 'r', 'g', 'm', 'c', 'y'])
        for energy in energyBins:
            xy_E = geo_engine.xy_pixelCoords_of_E(energy)
            x, y = zip(*xy_E)

            ax.plot(x, y, color=next(colors), linestyle='-')

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"Energy Lines for bin width = {bin_width}")
        ax.grid(True)

        plt.show()


    def plotBinWidthLinesMatrix(bin_width, index_of_interest):

        crys_pitch, crys_roll, cam_pitch, cam_roll, r_cam = access_saved_geometric(index_of_interest)

        energyBins = np.arange(1100, 1600 + bin_width, bin_width)

        geo_engine = Geometry(crystal_pitch=crys_pitch, crystal_roll=crys_roll,
                              camera_pitch=cam_pitch, camera_roll=cam_roll,
                              r_cam=r_cam, )

        mat_lines = np.zeros((length_detector_pixels, length_detector_pixels))
        # intialising the colours of the lines to better visibility
        for energy in energyBins:
            xy_E = geo_engine.xy_pixelCoords_of_E(energy)
            x, y = zip(*xy_E)

            mat_lines[y, x] += 1

        plt.imshow(mat_lines, cmap='gray')
        plt.show()


    def plotMatrixWithEnergyOfPixel(savefile=False):

        geo_eng = geo_engine_withSavedParams()
        bragg_eng = Bragg(crystal_pitch=geo_eng.crystal_pitch, crystal_roll=geo_eng.crystal_roll,
                          camera_pitch=geo_eng.camera_pitch, camera_roll=geo_eng.camera_roll,
                          r_camera_spherical=geo_eng.r_camera_spherical, )

        mat_Energy = np.zeros((length_detector_pixels, length_detector_pixels))

        for i in range(length_detector_pixels):
            for j in range(length_detector_pixels):
                energy_of_pixel = bragg_eng.xyPixelImagePlane_to_energy(xPixel_imPlane=j, yPixel_imPlane=i)
                mat_Energy[i, j] = energy_of_pixel

        if savefile:
            folder_path_to_save = r"/stored_variables"
            filename = "energy_of_pixel"
            np.save(f"{folder_path_to_save}/{filename}.npy", mat_Energy)

            print(mat_Energy)

        plt.imshow(mat_Energy, cmap='jet')
        plt.title("Energy Mapping of CCD image Plane")
        plt.colorbar(label="Energy (eV)")
        plt.show()


    def calibrate_quadratics(list_indices=list_data):

        print("calibrate_quadratics")
        for indexOI in list_indices:
            calibrate_and_save_quadratics(indexOI,
                                          howManySigma=2, adjacent_pixel_weighting=0.5, pixelwidth_lineintegral_5=True,
                                          bBounds=(700, 950),
                                          folderpath="stored_variables", saveData=True,
                                          plot_Results=True,
                                          )


    def calibrate_geometric(list_indices=list_data):

        print("calibrate_geometric")
        for indexOI in list_indices:
            geometry_fitMinimise(indexOI, useSavedVals=True, )


    calibrate_quadratics()

    # calibrate_geometric()
    # testPlotGeometryLines(1)

    # testPlotQuadLines(1,plot_gauss=False)


    pass
