from scipy.optimize import minimize
from pedestal_engine_v2 import *
from tools import *
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import os
import pandas as pd




class Geometry:
    def __init__(self, crystal_pitch, crystal_roll, camera_pitch, camera_roll,
                 r_cam, r_theta=2.567, xpixels=2048, ypixels=2048, pixelWidth=pixel_width,declareVars=True ):

        def print_initialised_geo_params():
            print("\n" + "-"*30)
            print("Initialised geometry parameters:")
            print("crystal_pitch:", crystal_pitch)
            print("crystal_roll:", crystal_roll)
            print("camera_pitch:", camera_pitch)
            print("camera_roll:", camera_roll)
            print("r_cam:", r_cam)

        if declareVars:
            print_initialised_geo_params()

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

    def line_xyCoords(self,energy_line_eV):

        xy_pixel = self.xy_pixelCoords_of_E(energy_line_eV)
        x_coords_pixel = []
        y_coords_pixel = []
        for row in xy_pixel:
            x_pixel = row[0]
            y_pixel = row[1]

            if x_pixel < length_detector_pixels and y_pixel < length_detector_pixels:
                x_coords_pixel.append(x_pixel)
                y_coords_pixel.append(y_pixel)

        return np.array(x_coords_pixel), np.array(y_coords_pixel)


class Quadratic_Fit:
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

        def gaussian_(X, amp_, sigma_, xpeak, c_offset):
            return amp_ * np.exp(-(X - xpeak) ** 2 / (2 * sigma_ ** 2)) + c_offset

        amp_guess = np.max(lineIntegralVals)
        xpeak_guess = cVals[np.argmax(lineIntegralVals)]
        sigma_guess = 20
        c_offset_guess = np.min(lineIntegralVals)

        try:
            amp_sigma_cPeak, pcov = curve_fit(gaussian_, cVals, lineIntegralVals,
                                              p0=[amp_guess, xpeak_guess, sigma_guess, c_offset_guess], maxfev=2000)
            if plotGauss:
                plt.plot(cVals, lineIntegralVals)
                plt.plot(cVals, gaussian_(cVals, *amp_sigma_cPeak))
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

ellipse_latex_string = r"C + A - A \cdot \sqrt{1 - \frac{(y - y_0)^2}{B^2}}"

class Ellipse_Fit:
    def __init__(self, imageMatrix, logTextFile=None, adjacentWeight=0.5,
                 loss_sigma_weight=1,
                 how_many_adjacent_pixels_each_side = 3 ):
        self.imMat = imageMatrix
        self.log = logTextFile
        self.adjacentWeight = adjacentWeight
        self.how_many_adjacent_pixels_each_side = how_many_adjacent_pixels_each_side
        self.loss_sigma_weight = loss_sigma_weight

        self.x_pixel_width = imageMatrix.shape[1]
        self.y_pixel_width = imageMatrix.shape[0]

    @staticmethod
    def ellipse_of_y(Y, c_, Y0, A, B):
        """
        For an ellipse of the form 1 = (x-x0)**2 / A **2 + (y-y0)**2 / B **2
        I want to iterate over the leftmost part of the ellipse which will be X0-A
        I will hence call this parameter C
        => x = C+A-A sqrt(1- (y-y0)**2 / B**2)
        """
        return c_ + A - A * np.sqrt(1 - (Y - Y0) ** 2 / B ** 2)

    @staticmethod
    def pixelised_ellipse(Y, c_, Y0, A, B):
        return np.round(c_ + A - A * np.sqrt(1 - (Y - Y0) ** 2 / B ** 2), 0).astype(int)

    def line_integral_across_CBounds(self, params_Y0_A_B, c_Bounds):

        # I will perform this using the natural indices that the matrix offers you which will result in
        # an effective reflection in the y=0 axis

        y_coords_pixel = np.arange(start=0, stop=self.y_pixel_width, step=1)

        list_integrals_of_c = []

        for cval in np.arange(start=c_Bounds[0], stop=c_Bounds[1], step=1):
            x_coord_pixel = self.pixelised_ellipse(y_coords_pixel, cval, *params_Y0_A_B)
            c_totVal = 0

            for x_pixel, y_pixel in zip(x_coord_pixel, y_coords_pixel):
                # print(x_pixel,y_pixel)

                if 0 <= x_pixel < self.x_pixel_width:
                    c_totVal += self.imMat[y_pixel, x_pixel]
                    sigma = 1
                    for d in range(1, self.how_many_adjacent_pixels_each_side):  # Adjust range to control width
                        weight = np.exp(-d ** 2 / (2 * sigma ** 2))

                        # Right side
                        if x_pixel + d < self.x_pixel_width:
                            c_totVal += weight * self.imMat[y_pixel, x_pixel + d]

                        # Left side
                        if x_pixel - d >= 0:
                            c_totVal += weight * self.imMat[y_pixel, x_pixel - d]



            # print([cval,c_totVal])
            list_integrals_of_c.append([cval, c_totVal])

        return list_integrals_of_c

    def fitGaussian(self, params_Y0_A_B, c_Bounds, plot_gauss=False, title=None):
        count_runtimeError = 0

        line_integrals_of_c = self.line_integral_across_CBounds(params_Y0_A_B, c_Bounds)
        cVals = np.array(line_integrals_of_c)[:, 0]
        lineIntegralVals = np.array(line_integrals_of_c)[:, 1]

        def gaussian_(c_, amp_, sigma_, c_0, offset_):
            return amp_ * np.exp(-(c_ - c_0) ** 2 / (2 * sigma_ ** 2)) + offset_

        amp_guess = np.max(line_integrals_of_c)
        c_0_guess = cVals[np.argmax(lineIntegralVals)]
        sigma_guess = 5
        c_offset_guess = np.min(lineIntegralVals)

        gauss_guess = [amp_guess, c_0_guess, sigma_guess, c_offset_guess]
        # print(gauss_guess)

        try:
            params_gauss, pcov = curve_fit(gaussian_, cVals, lineIntegralVals, p0=gauss_guess, maxfev=2000)

            if plot_gauss:
                plt.figure(figsize=(12, 8))
                plt.plot(cVals, lineIntegralVals)
                plt.plot(cVals, gaussian_(cVals, *params_gauss))

                if title is not None:
                    plt.title(title + " with gaussian weighted line width of 5 pixels")
                    plt.ylabel("Line Integral (ADU)")
                    plt.xlabel("C value in "+r"$" + ellipse_latex_string + r"$")

                else:
                    plt.ylabel("Line Integral a gaussian weighted width of 5 pixels")
                    plt.xlabel("C value in "+r"$" + ellipse_latex_string + r"$")
                plt.show()

            params_unc = np.sqrt(np.diag(pcov))

            return params_gauss, params_unc

        except RuntimeError:
            print(RuntimeError)
            count_runtimeError += 1
            if count_runtimeError > 10:
                raise RuntimeError("Too many RuntimeErrors, stopping function execution.")
            else:
                print("The fit could not be computed. Retrying with a larger range of C")
                clower = c_Bounds[0]
                cHigher = c_Bounds[1]
                c_dif = (cHigher - clower) / 4
                cBoundsNew = (int(clower - c_dif), int(cHigher + c_dif))

                print("New C bounds: ", cBoundsNew)

                self.fitGaussian(params_Y0_A_B, cBoundsNew, plot_gauss)

    def optimise_ellipse(self, y0_bounds, a_bounds, b_bounds, c_bounds,
                         plot_optimised_gaussian=False,iterations=30):

        def loss_func_cIntegral_gauss(params, sigmaWeighting_):
            # print("params", params)
            # print(f"y0 = {params[0]}, a = {params[1]}, b = {params[2]}")

            params_gauss_, params_unc_ = self.fitGaussian(params_Y0_A_B=params, c_Bounds=c_bounds, plot_gauss=False)

            # print(f"c_0 = {params_gauss_[2]}")

            # loss minimising encourages higher amplitude with minimal width
            loss = -params_gauss_[0] + sigmaWeighting_ * params_gauss_[1]

            # print(f"loss = {loss}")
            return loss

        bounds = [y0_bounds, a_bounds, b_bounds]
        initial_guess = np.array(
            [(y0_bounds[0] + y0_bounds[1]) / 2,
             (a_bounds[0] + a_bounds[1]) / 2,
             (b_bounds[0] + b_bounds[1]) / 2,
             ])

        result = minimize(loss_func_cIntegral_gauss, initial_guess, args=(self.loss_sigma_weight,), bounds=bounds,
                          method='Nelder-Mead',
                          callback=callbackminimise, options={'maxiter': iterations})

        y0, a, b = result.x
        lossOptimised = loss_func_cIntegral_gauss(result.x, self.loss_sigma_weight)

        params_gauss, params_unc = self.fitGaussian(params_Y0_A_B=result.x, c_Bounds=c_bounds,
                                                    plot_gauss=plot_optimised_gaussian)
        cPeak = params_gauss[2]
        cPeak_unc = params_unc[2]

        if self.log is not None:
            app = Append_to_file(self.log).append_and_print
            app("-" * 30)
            app(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            app(f"y0_bounds = {y0_bounds}" + "\n" + f"a_bounds = {a_bounds}" + "\n" + f"b_bounds = {b_bounds}" + "\n" + f"c_bounds = {c_bounds}")
            app(f"Optimised y0 = {y0}, a = {a}, b = {b}, C = {cPeak} +- {cPeak_unc}")
            app(f"Loss = {lossOptimised}")

        params_ellipse = [y0, a, b, cPeak]

        return params_ellipse, cPeak_unc

    def fit_image_lines(self, left_bounds, right_bounds,
                        plot_optimised_gaussian=False,iterations=30):

        keys = ["left", "right"]
        dict_params = {}

        for key, bounds in zip(keys, [left_bounds, right_bounds]):
            if self.log is not None:
                app = Append_to_file(self.log).append_and_print
                app("-" * 30)
                app(f"{key} Line: ")
            y0_bounds = bounds[0]
            a_bounds = bounds[1]
            b_bounds = bounds[2]
            c_bounds = bounds[3]

            params_ellipse, cPeak_unc = self.optimise_ellipse(y0_bounds, a_bounds, b_bounds, c_bounds,
                                                              plot_optimised_gaussian,iterations=iterations)
            # params_ellipse is [y0,a,b,c]

            dict_params[key] = {"params_ellipse": params_ellipse,
                                "cPeak_unc": cPeak_unc, }

        return dict_params

    def fitted_lines_image_matrix(self, optimised_ellipse_params,
                                  value_of_line_points=1):

        y_coords_pixel = np.arange(start=0, stop=self.y_pixel_width, step=1)

        y0, a, b, cPeak = optimised_ellipse_params
        x_coords_pixel = self.pixelised_ellipse(y_coords_pixel, cPeak, y0, a, b)

        linesMat = np.zeros((self.x_pixel_width, self.y_pixel_width))

        for xp, yp in zip(x_coords_pixel, y_coords_pixel):
            if 0 <= xp < self.x_pixel_width:
                linesMat[yp, xp] = value_of_line_points

        return linesMat



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
    def __init__(self, index_of_interest, geo_engine=None):
        if geo_engine is not None:
            self.geo_engine = geo_engine
        else:
            self.geo_engine = geo_engine_withSavedParams(index_of_interest)

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

# ---------- Calibration and saving of Quadratic Parameters:

def fit_quadratics(indexOfInterest,
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
    app = Append_to_file(logFile).append

    print("Index: ", indexOfInterest)

    # ----------pedestal mean and sigma----------
    image_mat, thr = mat_thr_aboveNsigma(indexOfInterest, howManySigma)
    imMatVeryClear = imVeryClear(image_mat, 0, (21, 5))

    # ---------- Initialise Quadratic Engine and log the initialising parameters ----------

    cal = Quadratic_Fit(image_mat, logFile, adjacentWeight=adjacent_pixel_weighting,
                        width_lineIntegral_5=pixelwidth_lineintegral_5)
    app("-" * 30)
    app(f"Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    app(f"Index of Interest {indexOfInterest}, thresholded above {howManySigma} sigma")
    app(f"Adjacent pixels weighted by {adjacent_pixel_weighting}")
    if pixelwidth_lineintegral_5:
        app("Line integral used pixel width of 5")
    else:
        app("Line integral used pixel width of 3")

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

    app(f"Finish: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

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
    print("left line variables: ", left_vars)
    right_vars = saved_variables[4:8]
    print("right line variables: ", right_vars)

    return left_vars, right_vars

# ---------- Calibration and saving of Ellipse Parameters:

def fit_ellipse(indexOfInterest, left_bounds=None, right_bounds=None,
                howManySigma=2, adjacent_pixel_weighting=0.5, pixelwidth_lineintegral_5=True,
                folderpath="stored_variables", saveData=True,
                plot_Results=False, ):
    # initialise subfolder if it doesn't exist
    index_folder = os.path.join(folderpath, str(indexOfInterest))
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    logFile = os.path.join(index_folder, "ellipse_fits_log.txt")
    app = Append_to_file(logFile).append

    print("Index: ", indexOfInterest)

    # ----------pedestal mean and sigma----------
    image_mat, thr = mat_thr_aboveNsigma(indexOfInterest, howManySigma)
    imMatVeryClear = imVeryClear(image_mat, 0, (21, 5))

    # ---------- Initialise Ellipse Engine and log the initialising parameters ----------

    cal_ellipse = Ellipse_Fit(image_mat, logFile, adjacentWeight=adjacent_pixel_weighting,
                              how_many_adjacent_pixels_each_side=3)
    app("-" * 30)
    app(f"Start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")
    app(f"Index of Interest {indexOfInterest}, thresholded above {howManySigma} sigma")
    app(f"Adjacent pixels weighted by {adjacent_pixel_weighting}")
    if pixelwidth_lineintegral_5:
        app("Line integral used pixel width of 5")
    else:
        app("Line integral used pixel width of 3")

    if left_bounds is None:
        y0_bounds = (700, 950)
        a_bounds = (6000, 8000)
        b_bounds = (6000, 8000)  # Note a_b similarity ==> encourages near circular like face
        c_bounds = (1220, 1340)

        left_bounds = [y0_bounds, a_bounds, b_bounds, c_bounds]

    if right_bounds is None:
        y0_bounds = (700, 950)
        a_bounds = (6000, 8000)
        b_bounds = (6000, 8000)
        c_bounds = (1380, 1460)

        right_bounds = [y0_bounds, a_bounds, b_bounds, c_bounds]

    params_dict_ = cal_ellipse.fit_image_lines(left_bounds=left_bounds, right_bounds=right_bounds,
                                               plot_optimised_gaussian=plot_Results)
    left_dict = params_dict_["left"]
    left_params = left_dict["params_ellipse"]
    left_c_unc = left_dict["cPeak_unc"]

    right_params = params_dict_["right"]["params_ellipse"]
    right_c_unc = params_dict_["right"]["cPeak_unc"]

    app(f"Finish: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}")

    if saveData:
        filepath = os.path.join(index_folder, "ellipse_fits.npy")
        left_all = left_params + [left_c_unc]
        right_all = right_params + [right_c_unc]
        ellipse_vars = np.array([
            left_all,
            right_all
        ])
        np.save(filepath, ellipse_vars)

    if plot_Results:
        matLeft = cal_ellipse.fitted_lines_image_matrix(left_params)
        matRight = cal_ellipse.fitted_lines_image_matrix(right_params)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.imshow(image_mat, cmap='hot'), plt.title(
            f"Image {indexOfInterest} thresholded above {howManySigma} sigma")
        plt.subplot(1, 2, 2), plt.imshow(imMatVeryClear + matLeft + matRight, cmap='hot'), plt.title(
            f"Average Pooled Image with geometric lines")
        plt.show()


def access_saved_ellipse(indexOfInterest, folderpath="stored_variables"):
    index_folder = os.path.join(folderpath, str(indexOfInterest))
    filepath = os.path.join(index_folder, "ellipse_fits.npy")

    saved_variables = np.load(filepath)

    left_vars = saved_variables[0][:-1]
    left_c_unc = saved_variables[0][-1]
    # print(left_vars)
    right_vars = saved_variables[1][:-1]
    right_c_unc = saved_variables[1][-1]
    # print(right_vars)

    return left_vars, right_vars, left_c_unc, right_c_unc

def saved_ellipse_gaussPlot(indexOfInterest, folderpath="stored_variables",howManySigma=2):
    left_vars, right_vars, left_c_unc, right_c_unc = access_saved_ellipse(indexOfInterest, folderpath)
    image_mat, thr = mat_thr_aboveNsigma(indexOfInterest, howManySigma)
    ellipse_eng = Ellipse_Fit(image_mat,)
    ellipse_eng.fitGaussian(left_vars[:-1],c_Bounds=(1220, 1340),plot_gauss=True,title=f"Image {indexOfInterest} Left Line")
    ellipse_eng.fitGaussian(right_vars[:-1], c_Bounds=(1380, 1460), plot_gauss=True,title=f"Image {indexOfInterest} Right Line")

# ---------- Calibration and saving of Geometric Parameters:


def fit_geometry_to_ellipse(indexOfInterest,
                            initialGuess=None, bounds=None, r_thetaval=2.567,
                            folderpath="stored_variables", saveData=True,
                            how_many_sigma=2,
                            weight_ofsettedPoints=0.5, iterations=30, plot=False,
                            lossMethodNew=True, useQuadratic=False):
    if initialGuess is None:
        initialGuess = np.array([-0.3445,  # crystal pitch
                                 0.0184,  # crystal roll
                                 0.814,  # Camera pitch, pi/4 is
                                 -0.00537,  # camera roll
                                 0.0839  # r camera
                                 ])
    if bounds is None:
        bounds = [(-0.346, -0.343),  # crystal pitch Bounds
                  (0.018, 0.019),  # crystal roll Bounds
                  (0.75, 0.85),  # Camera pitch Bounds
                  (-0.006, -0.005),  # camera roll Bounds
                  (0.0830, 0.0860),  # rcamBounds
                  ]

    # Ensuring the folder is initialised
    index_folder = os.path.join(folderpath, str(indexOfInterest))
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    if useQuadratic:
        logTextFile = os.path.join(index_folder, "geometric_fits_usingQuad_log.txt")
    else:
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
        log_.append(f"New loss method: {lossMethodNew}")


    logStart()

    print(indexOfInterest)
    image_mat = loadData()[indexOfInterest]

    # Getting threshold using pedetsal engine
    ped_mean, ped_sigma = pedestal_mean_sigma_awayFromLines(image_mat, indexOfInterest)
    thr = ped_mean + how_many_sigma * ped_sigma

    image_mat = np.where(image_mat > thr, image_mat, 0)
    # imMatVeryClear = imVeryClear(image_mat, 0, (21, 5))


    if useQuadratic:
        cal_quadratic = Quadratic_Fit(image_mat,None,adjacentWeight=0.5,width_lineIntegral_5=True)
        leftvars, rightvars = access_saved_quadratics(indexOfInterest,folderpath)
        linesMatLeft = cal_quadratic.matrixWithLines(Aoptimised=leftvars[0], Boptimised=leftvars[1],
                                                     Coptimised=leftvars[2], plotLines=False)
        linesMatRight = cal_quadratic.matrixWithLines(Aoptimised=rightvars[0], Boptimised=rightvars[1],
                                                     Coptimised=rightvars[2], plotLines=False)

    else:
        cal_ellipse = Ellipse_Fit(image_mat, None, adjacentWeight=0.5, how_many_adjacent_pixels_each_side=3)
        left_vars_y0ABc, right_vars_y0ABc, left_c_unc, right_c_unc = access_saved_ellipse(indexOfInterest, folderpath)
        linesMatLeft = cal_ellipse.fitted_lines_image_matrix(optimised_ellipse_params=left_vars_y0ABc)
        linesMatRight = cal_ellipse.fitted_lines_image_matrix(optimised_ellipse_params=right_vars_y0ABc, )

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

            left = [betaLineCoords_pixel, linesMatLeft]
            right = [alphaLineCoords_pixel, linesMatRight]

            for pixel_mat_list in [left, right]:
                geometric_line_pixels = pixel_mat_list[0]
                linesMat = pixel_mat_list[1]

                for row in geometric_line_pixels:
                    x_pixel = row[0]
                    y_pixel = row[1]

                    # now we want to find the pixel distance
                    # Finding the indices in the given row that are non-zero
                    nonzero_indices = np.nonzero(linesMat[y_pixel])[0]
                    mean_position = np.mean(nonzero_indices)

                    difference = abs(x_pixel - mean_position)

                    # print("y_pixel", y_pixel, "x_pixel", x_pixel)
                    # print("nonzero_indices", nonzero_indices)
                    # print("mean_position", mean_position)
                    # print(difference)

                    loss_to_be_minimised += difference

            return loss_to_be_minimised

        # We wish to maximise the integral along these lines
        if lossMethodNew:
            loss = computeLossNew()
        else:
            loss = computeLossOld()

        def printParamsAndLoss():
            print("-" * 30)
            print("Params:")
            print("crysPitch = ", p[0], "CrysRoll = ", p[1])
            print("CamPitch = ", p[2], "CamRoll = ", p[3])
            print("rcamSpherical = ", np.array([p[4], r_thetaval, np.pi]))
            print("Loss = ", loss)

        if abs(loss) < 8000:
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

    def logResults():
        app = Append_to_file(logTextFile).append_and_print
        app("-" * 30)
        app(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        app(f"optimised crystal pitch = {crystal_pitch}, n crystal roll = {crystal_roll}")
        app(f"optimised camera pitch = {camera_pitch}, n camera roll = {camera_roll}")
        app(f"optimised rcam spherical = {rcamSphericalOptimised}")

        app(f"Optimised n crystal = {ncrysOptimised}")
        app(f"Optimised n camera = {ncamOptimised}")
        app(f"Optimised r camera = {rcamSphericalOptimised}")
        app(f"Loss = {lossOptimised}")

    if logTextFile is not None:
        logResults()

    if saveData:
        if useQuadratic:
            filepath = os.path.join(index_folder, "geometric_fits_usingQuad.npy")
        else:
            filepath = os.path.join(index_folder, "geometric_fits.npy")

        geometric_vars = np.array(optimisedParams)
        np.save(filepath, geometric_vars)

    if plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.imshow(linesMatLeft + linesMatRight, cmap='hot'), plt.title('Quadratic Lines')
        plt.subplot(1, 2, 2), plt.imshow(linesMatGeoOptimised + linesMatLeft + linesMatRight, cmap='hot'), plt.title(
            'Geometic Lines plotted Over')
        plt.show()

    return lossOptimised


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


def geo_engine_withSavedParams(index_oI,declareVars=True):
    crys_pitch, crys_roll, cam_pitch, cam_roll, r_cam = access_saved_geometric(index_oI)

    geo_engine = Geometry(crystal_pitch=crys_pitch, crystal_roll=crys_roll,
                          camera_pitch=cam_pitch, camera_roll=cam_roll,
                          r_cam=r_cam, declareVars=declareVars )

    return geo_engine


# ---------- Generate energy and solid angle matrices

def save_energy_and_solid_angle_matrix(indexOfInterest, savefile=True, folderpath="stored_variables", ifplot=False, solid_angle_grid_width=10 ):
    # Ensuring the folder is initialised
    print("-"*30)
    print("Saving Energy and Solid Angle Matrices")
    index_folder = os.path.join(folderpath, str(indexOfInterest))
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    geo_eng = geo_engine_withSavedParams(indexOfInterest)
    bragg_eng = Bragg(crystal_pitch=geo_eng.crystal_pitch, crystal_roll=geo_eng.crystal_roll,
                      camera_pitch=geo_eng.camera_pitch, camera_roll=geo_eng.camera_roll,
                      r_camera_spherical=geo_eng.r_camera_spherical, )
    solidAngle_engine = SolidAngle(indexOfInterest, geo_eng)

    mat_Energy = np.zeros((length_detector_pixels, length_detector_pixels))
    mat_Solid_Angle = np.zeros((length_detector_pixels, length_detector_pixels))

    for i in range(length_detector_pixels):
        for j in range(length_detector_pixels):
            energy_of_pixel = bragg_eng.xyPixelImagePlane_to_energy(xPixel_imPlane=j, yPixel_imPlane=i)
            solidAngle_of_pixel = solidAngle_engine.solidAngle_pixelij(i, j, number_points=solid_angle_grid_width,)
            mat_Energy[i, j] = energy_of_pixel
            mat_Solid_Angle[i, j] = solidAngle_of_pixel

    if savefile:
        energy_filepath = os.path.join(index_folder, "energy_of_pixel.npy")
        np.save(energy_filepath, mat_Energy)
        solid_angle_filepath = os.path.join(index_folder, "solid_angle_of_pixel.npy")
        np.save(solid_angle_filepath, mat_Solid_Angle)

    if ifplot:
        plt.imshow(mat_Energy, cmap='jet')
        plt.title("Energy Mapping of CCD image Plane")
        plt.colorbar(label="Energy (eV)")
        plt.show()

        plt.imshow(mat_Solid_Angle, cmap='jet')
        plt.title("Solid Angle Mapping of CCD image Plane")
        plt.colorbar(label="Solid Angle")
        plt.show()

    return mat_Energy, mat_Solid_Angle

def save_energy_mat(indexOfInterest, savefile=True, folderpath="stored_variables", ifplot=False, ):
    # Ensuring the folder is initialised
    index_folder = os.path.join(folderpath, str(indexOfInterest))

    geo_eng = geo_engine_withSavedParams(indexOfInterest)
    bragg_eng = Bragg(crystal_pitch=geo_eng.crystal_pitch, crystal_roll=geo_eng.crystal_roll,
                      camera_pitch=geo_eng.camera_pitch, camera_roll=geo_eng.camera_roll,
                      r_camera_spherical=geo_eng.r_camera_spherical, )

    mat_Energy = np.zeros((length_detector_pixels, length_detector_pixels))

    for i in range(length_detector_pixels):
        for j in range(length_detector_pixels):
            energy_of_pixel = bragg_eng.xyPixelImagePlane_to_energy(xPixel_imPlane=j, yPixel_imPlane=i)
            mat_Energy[i, j] = energy_of_pixel

    if savefile:
        energy_filepath = os.path.join(index_folder, "energy_of_pixel.npy")
        np.save(energy_filepath, mat_Energy)

    if ifplot:
        plt.imshow(mat_Energy, cmap='jet')
        plt.title("Energy Mapping of CCD image Plane")
        plt.colorbar(label="Energy (eV)")
        plt.show()

def save_solidAngle_mat(indexOfInterest, savefile=True, folderpath="stored_variables", ifplot=False, ):
    # Ensuring the folder is initialised
    index_folder = os.path.join(folderpath, str(indexOfInterest))
    if not os.path.exists(index_folder):
        os.makedirs(index_folder)

    geo_eng = geo_engine_withSavedParams(indexOfInterest)
    solidAngle_engine = SolidAngle(indexOfInterest, geo_eng)

    mat_Solid_Angle = np.zeros((length_detector_pixels, length_detector_pixels))

    for i in range(length_detector_pixels):
        for j in range(length_detector_pixels):
            solidAngle_of_pixel = solidAngle_engine.solidAngle_pixelij(i, j, number_points=1)
            mat_Solid_Angle[i, j] = solidAngle_of_pixel

    if savefile:
        solid_angle_filepath = os.path.join(index_folder, "solid_angle_of_pixel.npy")
        np.save(solid_angle_filepath, mat_Solid_Angle)

    if ifplot:
        plt.imshow(mat_Solid_Angle, cmap='jet')
        plt.title("Solid Angle Mapping of CCD image Plane")
        plt.colorbar(label="Solid Angle")
        plt.show()


# -------- main export functions:


class Calibrate:
    def __init__(self, list_indices=list_good_data, folderpath="stored_variables"):
        self.list_indices = list_indices
        self.folderpath = folderpath

    def calibrate_quadratic(self):
        print("calibrate_quadratic")
        for indexOI in self.list_indices:
            fit_quadratics(indexOI, folderpath=self.folderpath)

    def calibrate_ellipse(self):
        print("calibrate_ellipse")
        for indexOI in self.list_indices:
            fit_ellipse(indexOI, plot_Results=False, folderpath=self.folderpath)

    def calibrate_geometric(self):
        print("calibrate_geometric")
        for indexOI in self.list_indices:
            fit_geometry_to_ellipse(indexOI, folderpath=self.folderpath)

    def calibrate_geometric_usingQuad(self):
        print("calibrate_geometric_usingQuad")
        for indexOI in self.list_indices:
            fit_geometry_to_ellipse(indexOI, folderpath=self.folderpath,useQuadratic=True)

    def calibrate_energy_solidAngle(self):
        print("calibrate_energy_solidAngle")
        for indexOI in self.list_indices:
            print(indexOI)
            save_energy_and_solid_angle_matrix(indexOI, savefile=True, folderpath=self.folderpath)

    def calibrate_energy_mat(self):
        print("calibrate_energy_mat")
        for indexOI in self.list_indices:
            print(indexOI)
            save_energy_mat(indexOI, savefile=True, folderpath=self.folderpath)


    def calibrate_solid_angle_mat(self):
        print("calibrate_solid_angle_mat")
        for indexOI in self.list_indices:
            print(indexOI)
            save_solidAngle_mat(indexOI, savefile=True, folderpath=self.folderpath)

# -------- Visualisation Classes

class TestPlot:
    def __init__(self,indexOfInterest, how_many_sigma=2,):
        self.indexOfInterest = indexOfInterest
        self.how_many_sigma = how_many_sigma

    def testPlotQuadLines(self,plot_gauss=True, ):
        left_vars, right_vars = access_saved_quadratics(self.indexOfInterest)
        Aleft = left_vars[0]
        Bleft = left_vars[1]
        Cleft = left_vars[2]

        Aright = right_vars[0]
        Bright = right_vars[1]
        Cright = right_vars[2]

        print(self.indexOfInterest)
        image_mat, thr = mat_minusMean_thr_aboveNsigma(self.indexOfInterest, self.how_many_sigma)
        mat_plot = image_mat

        calibrate_tpql = Quadratic_Fit(image_mat, None, adjacentWeight=0.5, width_lineIntegral_5=True)
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
        titleL1 = f"Image {self.indexOfInterest} with saved A(y-B)**2 + C coefficients"
        titleL2 = f"\nLeft: A={Aleft:.2f}, B={Bleft:.2f}, C={Cleft:.2f}"
        titleL3 = f"\nRight: A={Aright:.2f}, B={Bright:.2f}, C={Cright:.2f}"
        plt.title(titleL1 + titleL2 + titleL3)
        plt.show()

    def testPlot_ellipse_lines(self,plot_gauss=True, ):
        left_vars_y0ABc, right_vars_y0ABc, left_c_unc, right_c_unc = access_saved_ellipse(self.indexOfInterest)

        print(self.indexOfInterest)
        image_mat, thr = mat_minusMean_thr_aboveNsigma(self.indexOfInterest, self.how_many_sigma)
        mat_plot = image_mat

        cal_ellipse = Ellipse_Fit(image_mat, None, adjacentWeight=0.5,
                                  how_many_adjacent_pixels_each_side=3)
        if plot_gauss:
            params_Y0_A_B_left = left_vars_y0ABc[:-1]
            c_left = left_vars_y0ABc[-1]
            params_Y0_A_B_right = right_vars_y0ABc[:-1]
            c_right = right_vars_y0ABc[-1]

            cal_ellipse.fitGaussian(params_Y0_A_B=params_Y0_A_B_left, c_Bounds=(c_left - 60, c_left + 60))
            cal_ellipse.fitGaussian(params_Y0_A_B=params_Y0_A_B_right, c_Bounds=(c_right - 60, c_right + 60))

        val_line = np.max(mat_plot) / 4

        mat_left = cal_ellipse.fitted_lines_image_matrix(optimised_ellipse_params=left_vars_y0ABc,
                                                         value_of_line_points=val_line)
        mat_right = cal_ellipse.fitted_lines_image_matrix(optimised_ellipse_params=right_vars_y0ABc,
                                                          value_of_line_points=val_line)

        plt.imshow(mat_plot + mat_left + mat_right, cmap="hot")
        # plt.imshow(mat_left + mat_right, cmap="hot")
        titleL1 = f"Image {self.indexOfInterest} with saved x = c + A - A*np.sqrt(1 - (Y-Y0)**2 / B**2) coefficients"
        titleL2 = f"\nLeft: c={left_vars_y0ABc[-1]:.2f}, A={left_vars_y0ABc[1]:.2f}, y0={left_vars_y0ABc[0]:.2f}, B={left_vars_y0ABc[2]}"
        titleL3 = f"\nRight: c={right_vars_y0ABc[-1]:.2f}, A={right_vars_y0ABc[1]:.2f}, y0={right_vars_y0ABc[0]:.2f}, B={right_vars_y0ABc[2]}"
        plt.title(titleL1 + titleL2 + titleL3)
        plt.show()

    def testPlotGeometryLines(self, r_theta=2.567,testVars=None, plotEllipse_vs_geometric=False):
        print(self.indexOfInterest)

        image_mat, thr = mat_minusMean_thr_aboveNsigma(self.indexOfInterest, self.how_many_sigma)
        imMatVeryClear = imVeryClear(image_mat, 0, (21, 5))

        if testVars is None:
            crysPitch, CrysRoll, CamPitch, CamRoll, rcam = access_saved_geometric(self.indexOfInterest)
        else:
            crysPitch, CrysRoll, CamPitch, CamRoll, rcam = testVars

        rcamSpherical = np.array([rcam, r_theta, np.pi])

        print("crysPitch = ", crysPitch, "CrysRoll = ", CrysRoll)
        print("CamPitch = ", CamPitch, "CamRoll = ", CamRoll)
        print("rcamSpherical = ", rcamSpherical)

        geo = Geometry(crysPitch, CrysRoll, CamPitch, CamRoll, r_cam=rcam, r_theta=r_theta)
        geolinesMat = geo.createLinesMatrix(imTest, np.max(im_very_clear), phiStepSize=0.0001)

        title_l1 = f"Lα and Lβ lines for image {self.indexOfInterest}"
        title_l2 = f"\ncrystal: Pitch = {crysPitch:.5g}, Roll = {CrysRoll:.5g}"
        title_l3 = f"\ncamera: Pitch = {CamPitch:.5g}, Roll = {CamRoll:.5g}"
        title_l4 = f"\nr_camera = {rcam:.5g}"

        # plt.imshow(imMatVeryClear + geolinesMat, cmap="hot")
        # plt.title(title_l1 + title_l2 + title_l3 + title_l4)
        # plt.show()

        plt.imshow(image_mat + geolinesMat, cmap="jet")
        plt.title(title_l1 + title_l2 + title_l3 + title_l4)
        plt.show()

        if plotEllipse_vs_geometric:
            left_vars_y0ABc, right_vars_y0ABc, left_c_unc, right_c_unc = access_saved_ellipse(self.indexOfInterest)

            cal_ellipse = Ellipse_Fit(image_mat, None, adjacentWeight=0.5,
                                      how_many_adjacent_pixels_each_side=3)

            val_line = np.max(im_very_clear) / 4

            matLeft = cal_ellipse.fitted_lines_image_matrix(optimised_ellipse_params=left_vars_y0ABc,
                                                            value_of_line_points=val_line)
            matRight = cal_ellipse.fitted_lines_image_matrix(optimised_ellipse_params=right_vars_y0ABc,
                                                             value_of_line_points=val_line)

            plt.imshow(matLeft + matRight + geolinesMat, cmap="jet")
            plt.title("")
            plt.show()

    def plot_energy_mats(self,folderpath="stored_variables"):
        index_folder = os.path.join(folderpath, str(self.indexOfInterest))
        energy_filepath = os.path.join(index_folder, "energy_of_pixel.npy")
        mat_Energy = np.load(energy_filepath)

        plt.imshow(mat_Energy, cmap='jet')
        plt.title(f"Image {self.indexOfInterest}: Energy Mapping of CCD image Plane")
        plt.colorbar(label="Energy (eV)")
        plt.xlabel("j index")
        plt.ylabel("i index")
        plt.show()

    def plot_solidAngle_mats(self, folderpath="stored_variables"):
        index_folder = os.path.join(folderpath, str(self.indexOfInterest))
        solid_angle_filepath = os.path.join(index_folder, "solid_angle_of_pixel.npy")
        mat_Solid_Angle = np.load(solid_angle_filepath)

        plt.imshow(mat_Solid_Angle, cmap='jet')
        plt.title(f"Image {self.indexOfInterest} Solid Angle Mapping of CCD image Plane")
        plt.colorbar(label="Solid Angle (sr)")
        plt.show()

    def plotBinWidthLines(self, bin_width):
        crys_pitch, crys_roll, cam_pitch, cam_roll, r_cam = access_saved_geometric(self.indexOfInterest)

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
        ax.set_title(f"Image {self.indexOfInterest} Energy Lines for bin width = {bin_width}")
        ax.grid(True)

        plt.show()

class Violin:
    def __init__(self, list_indexOI=list_good_data):
        self.list_indexOI = list_indexOI

    @staticmethod
    def plotViolins_curve(left_dictionary, right_dictionary, title_Equation):

        number_of_variables = len(left_dictionary)

        for str_side, dictionary in zip(["Left", "Right"], [left_dictionary, right_dictionary]):
            print(str_side)
            fig, axes = plt.subplots(1, number_of_variables, figsize=(18, 3))

            for idx, (key, ax) in enumerate(zip(dictionary.keys(), axes)):
                sns.violinplot(data=dictionary[key],
                               inner="point",
                               color="#00BFFF", linewidth=0, ax=ax)

                sns.stripplot(data=dictionary[key], color="black", alpha=0.7, ax=ax)
                ax.set_title(f"{key}")
                ax.grid(True)
                ax.tick_params(axis='y', labelsize=5)
                # ax.set_ylabel(f"{key}")

            # Add a title to the whole figure
            fig.suptitle(f"{str_side} Line : {title_Equation}", fontsize=16,
                         # y=1.05
                         )
            plt.tight_layout()
            plt.show()

    def quadratic_params(self):
        vals_dict_left = {
            "A": [],
            "B": [],
            "C": [],
        }

        vals_dict_right = {
            "A": [],
            "B": [],
            "C": [],
        }

        for iOI in self.list_indexOI:
            left_vars, right_vars = access_saved_quadratics(iOI)

            vals_dict_left["A"].append(left_vars[0])
            vals_dict_left["B"].append(left_vars[1])
            vals_dict_left["C"].append(left_vars[2])

            vals_dict_right["A"].append(right_vars[0])
            vals_dict_right["B"].append(right_vars[1])
            vals_dict_right["C"].append(right_vars[2])

        self.plotViolins_curve(left_dictionary=vals_dict_left, right_dictionary=vals_dict_right,
                               title_Equation="x = A(y-B)**2 + C")


    def ellipse_params(self, ):
        vals_dict_left = {
            "y0": [],
            "A": [],
            "B": [],
            "C": [],
        }

        vals_dict_right = {
            "y0": [],
            "A": [],
            "B": [],
            "C": [],
        }

        for iOI in self.list_indexOI:
            left_vars_y0ABc, right_vars_y0ABc, left_c_unc, right_c_unc = access_saved_ellipse(iOI)

            vals_dict_left["y0"].append(left_vars_y0ABc[0])
            vals_dict_left["A"].append(left_vars_y0ABc[1])
            vals_dict_left["B"].append(left_vars_y0ABc[2])
            vals_dict_left["C"].append(left_vars_y0ABc[3])

            vals_dict_right["y0"].append(right_vars_y0ABc[0])
            vals_dict_right["A"].append(right_vars_y0ABc[1])
            vals_dict_right["B"].append(right_vars_y0ABc[2])
            vals_dict_right["C"].append(right_vars_y0ABc[3])

        self.plotViolins_curve(left_dictionary=vals_dict_left, right_dictionary=vals_dict_right,
                               title_Equation="x = C+A-A sqrt(1- (y-y0)**2 / B**2)")

    def geo_params(self):
        vals_dict = {
            "crystal pitch (rad)": [],
            "crystal roll (rad)": [],
            "camera pitch (rad)": [],
            "camera roll (rad)": [],
            "r cam (m)": [], }

        for iOI in self.list_indexOI:
            crysPitch, CrysRoll, CamPitch, CamRoll, rcam = access_saved_geometric(iOI)
            optimised_geoParams = [crysPitch, CrysRoll, CamPitch, CamRoll, rcam]

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

        fig.suptitle(f"Geometric Fittings using Elliptical Curves", fontsize=16,
                     # y=1.05
                     )

        plt.tight_layout()
        plt.show()


    def geo_params_usingQuad(self,folderpath="stored_variables"):
        vals_dict = {
            "crystal pitch (rad)": [],
            "crystal roll (rad)": [],
            "camera pitch (rad)": [],
            "camera roll (rad)": [],
            "r cam (m)": [], }

        for iOI in self.list_indexOI:
            index_folder = os.path.join(folderpath, str(iOI))
            filepath = os.path.join(index_folder, "geometric_fits_usingQuad.npy")

            saved_variables = np.load(filepath)

            crysPitch = saved_variables[0]
            CrysRoll = saved_variables[1]
            CamPitch = saved_variables[2]
            CamRoll = saved_variables[3]
            rcam = saved_variables[4]
            optimised_geoParams = [crysPitch, CrysRoll, CamPitch, CamRoll, rcam]

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
            ax.tick_params(axis='y', labelsize=6)

        fig.suptitle(f"Geometric Fittings using Quadratic Curves", fontsize=16,
                     # y=1.05
                     )
        plt.tight_layout()
        plt.show()

    def r_cam_ellipseVSquad(self,folderpath="stored_variables"):

        r_cam_ellipse = []
        r_cam_quad = []


        for iOI in self.list_indexOI:
            crysPitch, CrysRoll, CamPitch, CamRoll, rcam = access_saved_geometric(iOI)
            r_cam_ellipse.append(rcam)

            index_folder = os.path.join(folderpath, str(iOI))
            filepath = os.path.join(index_folder, "geometric_fits_usingQuad.npy")

            saved_variables = np.load(filepath)
            r_cam_quad.append(saved_variables[4])

        data = [r_cam_quad,r_cam_ellipse]

        plt.figure(figsize=(7, 5))

        sns.violinplot(data=data,inner=None)
        sns.swarmplot(data=data, color='k', alpha=0.5, size=6)
        plt.ylabel(r"$|\mathbf{r_{cam}}|$ (m)",fontsize=14)
        plt.xticks([0, 1], ['Quadratic', 'Elliptical'],fontsize=14)

        plt.title(r"$|\mathbf{r_{cam}}|$ Fitted Using Elliptical and Quadratic Lines", fontsize=16,)
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def crys_pitch_ellipseVsQuad(self,folderpath="stored_variables"):

        crys_pitch_ellipse = []
        crys_pitch_quad = []


        for iOI in self.list_indexOI:
            crysPitch, CrysRoll, CamPitch, CamRoll, rcam = access_saved_geometric(iOI)
            crys_pitch_ellipse.append(crysPitch)

            index_folder = os.path.join(folderpath, str(iOI))
            filepath = os.path.join(index_folder, "geometric_fits_usingQuad.npy")

            saved_variables = np.load(filepath)
            crys_pitch_quad.append(saved_variables[0])

        data = [crys_pitch_quad,crys_pitch_ellipse]

        plt.figure(figsize=(7, 5))

        sns.violinplot(data=data,inner=None)
        sns.swarmplot(data=data, color='k', alpha=0.5, size=6)
        plt.ylabel("Crystal Pitch (rad)")
        plt.xticks([0, 1], ['Quadratic', 'Elliptical'])

        plt.title("Crystal Pitch Fitted Using Elliptical and Quadratic Lines", fontsize=16,)
        plt.tight_layout()
        plt.grid(True)
        plt.show()

    def ellipse_c_vals(self,folderpath="stored_variables"):
        vals_dict = {
            "Left": [],
            "Right": [],
        }

        for iOI in self.list_indexOI:
            left_vars_y0ABc, right_vars_y0ABc, left_c_unc, right_c_unc = access_saved_ellipse(iOI,folderpath)
            vals_dict["Left"].append(left_vars_y0ABc[-1])
            vals_dict["Right"].append(right_vars_y0ABc[-1])

        data = [vals_dict["Left"], vals_dict["Right"]]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), )
        sns.violinplot(data=data[0], inner=None, ax=axes[0])
        sns.swarmplot(data=data[0], color='k', alpha=0.5, size=6, ax=axes[0])
        axes[0].set_title("Left Line")
        axes[0].set_ylabel("C (pixels)")
        axes[0].grid(True)

        sns.violinplot(data=data[1], inner=None, ax=axes[1])
        sns.swarmplot(data=data[1], color='k', alpha=0.5, size=6, ax=axes[1])
        axes[1].set_title("Right Line")
        axes[1].grid(True)

        fig.suptitle(r"$C$ in " + "\n" + r"$" + ellipse_latex_string + r"$", fontsize=16)
        plt.tight_layout()
        plt.show()

    def ellipse_error_c_vals(self,folderpath="stored_variables"):
        vals_dict = {
            "Left": [],
            "Right": [],
        }

        for iOI in self.list_indexOI:
            left_vars_y0ABc, right_vars_y0ABc, left_c_unc, right_c_unc = access_saved_ellipse(iOI,folderpath)
            vals_dict["Left"].append(left_c_unc)
            vals_dict["Right"].append(right_c_unc)

        data = [vals_dict["Left"], vals_dict["Right"]]

        plt.figure(figsize=(7, 5))

        sns.violinplot(data=data, inner=None)
        sns.swarmplot(data=data, color='k', alpha=0.5, size=6)
        plt.ylabel("C Uncertainty (pixels)", fontsize=14)
        plt.xticks([0, 1], ['Left Line', 'Right Line'], fontsize=14)

        # title = r"Uncertainty in $c$" + "\n" + r"$" + ellipse_latex_string + r"$"

        # plt.title(title, fontsize=16, )
        plt.tight_layout()
        plt.grid(True)
        plt.show()

# -------- Create Unit Test

class Geo_UnitTest:
    def __init__(self,bounds=None):
        if bounds is None:
            bounds = [(-0.346, -0.3435),  # crystal pitch Bounds
                      (0.018, 0.019),  # crystal roll Bounds
                      (0.75, 0.85),  # Camera pitch Bounds
                      (-0.006, -0.005),  # camera roll Bounds
                      (0.083, 0.0841),  # rcamBounds
                      ]

        self.bounds = bounds

        random_params = []

        for bound in bounds:
            rand_val = np.random.uniform(bound[0], bound[1])
            random_params.append(rand_val)

        self.random_params = random_params

        geo_engine_rand = Geometry(crystal_pitch=random_params[0],crystal_roll=random_params[1],
                                   camera_pitch=random_params[2], camera_roll=random_params[3],
                                   r_cam=random_params[4])

        self.rdm_geo_engine = geo_engine_rand

    def rdm_xy_pixel_mat(self, noise_level_left=0.2,noise_level_right=0.1, testPlotRaw_noisy=False, testPlot_mat=False,phiStepSize=0.0005):

        xy_meter_alpha = self.rdm_geo_engine.xy_coords_of_E(E_Lalpha_eV,phiStepSize=phiStepSize)
        xy_meter_beta = self.rdm_geo_engine.xy_coords_of_E(E_Lbeta_eV,phiStepSize=phiStepSize)

        # list of values with [ [x1,y1], [x2,y2], ]

        dictLines = {}

        matLines = np.zeros((length_detector_pixels,length_detector_pixels))

        for list_xy,name, noise_level in zip([xy_meter_alpha,xy_meter_beta],["Alpha","Beta"],[noise_level_right,noise_level_left]):
            xvals = []
            yvals = []
            for row in list_xy:
                xvals.append(row[0])
                yvals.append(row[1])

            x_array = np.array(xvals)
            y_array = np.array(yvals)

            # Generate random noise scaled to the value range
            x_noise = np.random.uniform(-noise_level, noise_level, size=x_array.shape) * (x_array.max() - x_array.min())
            y_noise = np.random.uniform(-noise_level, noise_level, size=y_array.shape) * (y_array.max() - y_array.min())

            noisy_x = x_array + x_noise
            noisy_y = y_array + y_noise

            noisy_x_pixel = []
            noisy_y_pixel = []

            for x_val, y_val in zip(noisy_x, noisy_y):

                x_pixel,y_pixel = xy_meters_to_xyPixel(x_meters=x_val,y_meters=y_val)

                # print("x,y: ", (x_pixel,y_pixel))

                if x_pixel < length_detector_pixels and y_pixel < length_detector_pixels:

                    matLines[y_pixel, x_pixel] = np.random.uniform(50,150)

                    noisy_x_pixel.append(x_pixel)
                    noisy_y_pixel.append(y_pixel)

            if testPlotRaw_noisy:

                noisy_x_pixel_array = np.array(noisy_x_pixel)
                noisy_y_pixel_array = np.array(noisy_y_pixel)

                x_pixel_array = []
                y_pixel_array = []

                for x_val,y_val in zip(x_array, y_array):
                    x_pixel, y_pixel = xy_meters_to_xyPixel(x_meters=x_val, y_meters=y_val)
                    x_pixel_array.append(x_pixel)
                    y_pixel_array.append(y_pixel)

                plt.scatter(np.array(x_pixel_array), np.array(y_pixel_array), label='Original', s=5)
                plt.scatter(noisy_x_pixel_array, noisy_y_pixel_array, label='Noisy', s=5)
                plt.legend()
                plt.show()


            dictLines[name] = {
                "arr_x_noise": noisy_x,
                "arr_y_noise": noisy_y
            }

        if testPlot_mat:
            plt.imshow(matLines)
            plt.show()


        return matLines

    @staticmethod
    def pedestal_thresholded_mat(index_mat_sigma=11,how_many_sigma=2):
        meanPedestal, sigmaPedestal = pedestal_mean_sigma_awayFromLines(imMatrix=loadData()[index_mat_sigma],indexOfInterest=index_mat_sigma)
        ped_mat = generate_pedestal_mat(sigma=2*sigmaPedestal,x0=0)

        # Mirroring the double thresholding
        ped_mat_thresholded = np.where(ped_mat > how_many_sigma *sigmaPedestal,ped_mat,0)

        return ped_mat_thresholded

    def UnitTest_mat(self,noise_level_left=0.2,noise_level_right=0.1,index_mat_sigma=11,testPlot_mat=False,how_many_sigma=2,
                     phi_step_size=0.0005):

        rdm_mat = self.rdm_xy_pixel_mat(noise_level_left,noise_level_right,phiStepSize=phi_step_size)
        ped_mat = self.pedestal_thresholded_mat(index_mat_sigma,how_many_sigma)

        unit_test_mat = rdm_mat + ped_mat

        if testPlot_mat:
            plt.imshow(unit_test_mat)
            plt.title(f"Unit Test for Geometry Fitting\nLeft Noise Level = {noise_level_left}; Right Noise Level = {noise_level_right}; Phi Step Size = {phi_step_size}")
            plt.show()

        return unit_test_mat


    def correct_line_xyCoords(self,energy_line_eV):
        xy_pixel = self.rdm_geo_engine.xy_pixelCoords_of_E(energy_line_eV)
        x_coords_pixel = []
        y_coords_pixel = []
        for row in xy_pixel:
            x_pixel = row[0]
            y_pixel = row[1]

            if x_pixel < length_detector_pixels and y_pixel < length_detector_pixels:
                x_coords_pixel.append(x_pixel)
                y_coords_pixel.append(y_pixel)

        return np.array(x_coords_pixel), np.array(y_coords_pixel)


def test_geo_fitting(noise_level_left=0.05,noise_level_right=0.025,phi_step_size=0.0003,
                     left_bounds_ellipse=None, right_bounds_ellipse=None,
                     initialGuess=None,
                     plot_comparison=True):

    geo_ut = Geo_UnitTest()
    ut_mat = geo_ut.UnitTest_mat(noise_level_left,noise_level_right,phi_step_size=phi_step_size)

    cal_ellipse = Ellipse_Fit(ut_mat, logTextFile=None, adjacentWeight=0.5,
                              how_many_adjacent_pixels_each_side=3)

    if left_bounds_ellipse is None:
        y0_bounds = (700, 950)
        a_bounds = (6000, 8000)
        b_bounds = (6000, 8000)  # Note a_b similarity ==> encourages near circular like face
        c_bounds = (1220, 1360)

        left_bounds_ellipse = [y0_bounds, a_bounds, b_bounds, c_bounds]

    if right_bounds_ellipse is None:
        y0_bounds = (700, 950)
        a_bounds = (6000, 8000)
        b_bounds = (6000, 8000)
        c_bounds = (1360, 1500)

        right_bounds_ellipse = [y0_bounds, a_bounds, b_bounds, c_bounds]

    params_dict_ = cal_ellipse.fit_image_lines(left_bounds=left_bounds_ellipse, right_bounds=right_bounds_ellipse,
                                               plot_optimised_gaussian=False,iterations=20)
    left_params = params_dict_["left"]["params_ellipse"]
    right_params = params_dict_["right"]["params_ellipse"]

    linesMatLeft = cal_ellipse.fitted_lines_image_matrix(optimised_ellipse_params=left_params)
    linesMatRight = cal_ellipse.fitted_lines_image_matrix(optimised_ellipse_params=right_params, )

    def lossFunction(params_):
        p = params_

        geo = Geometry(crystal_pitch=p[0], crystal_roll=p[1],
                       camera_pitch=p[2], camera_roll=p[3],
                       r_cam=p[4], )

        alphaLineCoords_pixel = geo.xy_pixelCoords_of_E(E_Lalpha_eV)  # More Right / right line
        betaLineCoords_pixel = geo.xy_pixelCoords_of_E(E_Lbeta_eV)  # More Left / left line

        def computeLossNew():
            loss_to_be_minimised = 0

            left = [betaLineCoords_pixel, linesMatLeft]
            right = [alphaLineCoords_pixel, linesMatRight]

            for pixel_mat_list in [left, right]:
                geometric_line_pixels = pixel_mat_list[0]
                linesMat = pixel_mat_list[1]

                for row in geometric_line_pixels:
                    x_pixel = row[0]
                    y_pixel = row[1]

                    # now we want to find the pixel distance
                    # Finding the indices in the given row that are non-zero
                    nonzero_indices = np.nonzero(linesMat[y_pixel])[0]
                    mean_position = np.mean(nonzero_indices)

                    difference = abs(x_pixel - mean_position)

                    # print("y_pixel", y_pixel, "x_pixel", x_pixel)
                    # print("nonzero_indices", nonzero_indices)
                    # print("mean_position", mean_position)
                    # print(difference)

                    loss_to_be_minimised += difference

            return loss_to_be_minimised

        # We wish to maximise the integral along these lines
        loss = computeLossNew()

        return loss

    if initialGuess is None:
        initialGuess = np.array([-0.3445,  # crystal pitch
                                 0.0184,  # crystal roll
                                 0.814,  # Camera pitch, pi/4 is
                                 -0.00537,  # camera roll
                                 0.0839  # r camera
                                 ])

    result = minimize(lossFunction, initialGuess, bounds=geo_ut.bounds, method='Nelder-Mead', options={'maxiter': 20},
                      callback=callbackminimise)

    optimisedParams = result.x

    print("Random Params: ", geo_ut.random_params)
    print("Fitted Params: ", optimisedParams)

    geoOptimised = Geometry(crystal_pitch=optimisedParams[0], crystal_roll=optimisedParams[1],
                            camera_pitch=optimisedParams[2], camera_roll=optimisedParams[3],
                            r_cam=optimisedParams[4], )
    geo_x_coords_alpha,geo_y_coords_alpha = geoOptimised.line_xyCoords(E_Lalpha_eV)
    geo_x_coords_beta, geo_y_coords_beta = geoOptimised.line_xyCoords(E_Lbeta_eV)
    ut_x_coords_alp,ut_y_coords_alp = geo_ut.correct_line_xyCoords(E_Lalpha_eV)
    ut_x_coords_bet, ut_y_coords_bet = geo_ut.correct_line_xyCoords(E_Lbeta_eV)

    if plot_comparison:
        plt.imshow(ut_mat)
        plt.plot(geo_x_coords_alpha, geo_y_coords_alpha, color="red", linewidth=2, linestyle="--",label="Fitted Line (Alpha)",alpha=0.8)
        plt.plot(geo_x_coords_beta, geo_y_coords_beta, color="red", linewidth=2, linestyle="-",label="Fitted Line (Beta)",alpha=0.5)
        plt.plot(ut_x_coords_bet, ut_y_coords_bet, color="blue", linewidth=2, linestyle="--", label="Unit Test Line (Beta)",alpha=0.8)
        plt.plot(ut_x_coords_alp, ut_y_coords_alp, color="blue", linewidth=2, linestyle="-", label="Unit Test Line (Alpha)",alpha=0.5)
        plt.legend()
        plt.title("Unit Test for Geometrical Fitting Engine",fontsize=16)
        # plt.title(f"Unit Test for Geometrical Fitting Engine\nLeft Noise Level = {noise_level_left}; Right Noise Level = {noise_level_right}; Phi Step Size = {phi_step_size}")
        plt.ylabel("i index (pixels)",fontsize=14)
        plt.xlabel("j index (pixels)",fontsize=14)
        plt.show()


if __name__ == '__main__':

    # TestPlot(8,2).plot_energy_mats()

    # Violin().r_cam_ellipseVSquad()


    # Violin().ellipse_c_vals()

    # saved_ellipse_gaussPlot(8)
    # saved_ellipse_gaussPlot(11)

    # Violin().crys_pitch_ellipseVsQuad()
    # Violin().ellipse_error_c_vals()

    # TestPlot(8,2).plot_energy_mats()

    def geo_unit_test_tests():
        geo_utest = Geo_UnitTest()
        ut_mat_ = geo_utest.UnitTest_mat(testPlot_mat=True,noise_level_left=0.05,noise_level_right=0.025,phi_step_size=0.0003)


    # geo_unit_test_tests()

    test_geo_fitting(plot_comparison=True)

    def calibrate_all(list_indices=list_good_data, folder_path="stored_variables"):

        cal = Calibrate(list_indices,folder_path)
        cal.calibrate_quadratic()
        cal.calibrate_ellipse()
        cal.calibrate_geometric()
        cal.calibrate_geometric_usingQuad()
        cal.calibrate_energy_solidAngle()


    def calibrate_oneType(list_indices=list_good_data, folder_path="stored_variables"):
        cal = Calibrate(list_indices,folder_path)
        # cal.calibrate_geometric_usingQuad()
        cal.calibrate_quadratic()

    # calibrate_oneType()

    def plotAllGeometryLines(list_indices=list_good_data, folder_path="stored_variables"):
        print("plotAllGeometryLines")
        for index_ in list_indices:
            testPlot = TestPlot(index_,2)
            testPlot.testPlotGeometryLines()

    # plotAllGeometryLines()

    # calibrate_all()


    # calibrate_ellipse([11])
    # testPlot_ellipse_lines(11)

    # calibrate_geometric([11])
    # testPlotGeometryLines(7, )
    # testPlotGeometryLines(8, )

    # calibrate_solid_angle_mat()
    # calibrate_energy_solidAngle()

    # Violin().geo_params_usingQuad()
    # Violin().geo_params()

    # violinPlot_geo_params(list_data)
    # violinPlot_ellipse_params(list_data)
    # violinPlot_quadratic_params(list_data)

    # print(access_saved_quadratics(1,))

    # plot_energy_solidAngle_mats(1)

    pass
