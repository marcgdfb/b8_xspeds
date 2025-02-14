from scipy.optimize import minimize, curve_fit
from imagePreProcessing import *
from tools import *
from datetime import datetime
import cma

line_right_txt = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\data_logs\LineOnRightFitLog.txt"
line_left_txt = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\data_logs\LineOnLeftFitLog.txt"
geometryLog = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\data_logs\geometryFitLog.txt"


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


    def xy_coords_of_E(self, energy_eV,phiStepSize = 0.0001):

        # Rotation matrix from crystal 001 to n crystal
        rotMatrix_crys = rotMatrixUsingEuler(self.crystal_pitch,self.crystal_roll)

        # angle between crystal plane and ray
        theta_E = bragg_E_to_theta(energy_eV)
        # polar coordinate of this angle (still as if crystal is 001)
        theta_E_polar = np.pi / 2 + theta_E

        list_xy = []

        for phi in np.arange(3* np.pi / 4, 5 * np.pi / 4, phiStepSize):
            v_rayPrime_spherical = np.array([1, theta_E_polar, phi])
            v_rayPrime_cart = spherical_to_cartesian(v_rayPrime_spherical)
            v_ray_cart = np.dot(rotMatrix_crys, v_rayPrime_cart)
            v_ray_cart_normalised = v_ray_cart / np.linalg.norm(v_ray_cart)

            v_ray_spherical = cartesian_to_spherical(v_ray_cart_normalised)
            # print("Energy eV = ", energy_eV, "Theta Polar in normal frame = ", v_ray_spherical[1])

            x_plane, y_plane = ray_in_planeCamera(v_ray_cart=v_ray_cart_normalised, n_camera_cart=self.nCam, r_camera_cart=self.r_cam_cart)

            if ((abs(x_plane) < (self.xWidth - self.pixelWidth) / 2).all() and
                    (abs(y_plane) < (self.yWidth - self.pixelWidth) / 2).all()):
                list_xy.append([x_plane, y_plane])

        if not list_xy:
            print(f"No values for E = {energy_eV}")

        return list_xy

    def xy_coords_of_EAlphaBeta(self, ):

        listAlpha = self.xy_coords_of_E(E_Lalpha_eV)
        listBeta = self.xy_coords_of_E(E_Lbeta_eV)

        return listAlpha + listBeta

    def createLinesMatrix(self, imageMat, valMax,phiStepSize=0.0001):

        # If we are optimising r camera to be that going to the centre of the camera then the 0,0 pixel is displaced by:
        x_0 = - self.xWidth / 2
        y_0 = + self.yWidth / 2

        matrixLines = np.zeros((imageMat.shape[0], imageMat.shape[1]))

        for E in [E_Lalpha_eV, E_Lbeta_eV]:
            xy_E_list = self.xy_coords_of_E(E,phiStepSize)

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

    def rayBragg(self,energy_eV,phi_rad):

        # Test n crystal as nVectorFromEuler(-0.19,0)

        theta_E = bragg_E_to_theta(energy_eV)
        theta_E_polar = np.pi / 2 + theta_E

        v_rayPrime_spherical = np.array([1, theta_E_polar, phi_rad])
        print("v_rayPrime_spherical", v_rayPrime_spherical)
        v_rayPrime_cart = spherical_to_cartesian(v_rayPrime_spherical)
        print("v_rayPrime_cart",v_rayPrime_cart)

        v_ray_cart = np.dot(rotMatrixUsingEuler(self.crystal_pitch,self.crystal_roll), v_rayPrime_cart)
        v_ray_cart_normalised = v_ray_cart / np.linalg.norm(v_ray_cart)  # This works
        print("v_ray_cart_normalised",v_ray_cart_normalised)
        v_ray_spherical = cartesian_to_spherical(v_ray_cart_normalised)

        print(v_ray_spherical)

    def visualiseLines(self):
        linesMatLeft = Calibrate(imTest, None).matrixWithLines(Aoptimised=6.613794078473409e-05, Boptimised=862,
                                                               Coptimised=1278, plotLines=False)
        linesMatRight = Calibrate(imTest, None).matrixWithLines(Aoptimised=7.063102423636962e-05, Boptimised=862,
                                                                Coptimised=1418, plotLines=False)

        alphaLineCoords = self.xy_coords_of_E(E_Lalpha_eV)  # More Right / right line
        betaLineCoords = self.xy_coords_of_E(E_Lbeta_eV)  # More Left / left line

        matAlpha = np.zeros((self.ypixels, self.xpixels))
        matBeta = np.zeros((self.ypixels, self.xpixels))
        for row in alphaLineCoords:
            x_pixel = round((row[0] + self.xWidth / 2) / self.pixelWidth)
            y_pixel = round((self.yWidth / 2 - row[1]) / self.pixelWidth)

            matAlpha[y_pixel, x_pixel] = 1

        for row in betaLineCoords:
            x_pixel = round((row[0] + self.xWidth / 2) / self.pixelWidth)
            y_pixel = round((self.yWidth / 2 - row[1]) / self.pixelWidth)

            matBeta[y_pixel, x_pixel] = 1

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1), plt.imshow(matAlpha, cmap='hot'), plt.title('Alpha Line')
        plt.subplot(1, 2, 2), plt.imshow(matBeta, cmap='hot'), plt.title('Beta Line')
        plt.show()


class Calibrate:
    def __init__(self, imageMatrix, logTextFile=None):
        self.imMat = imageMatrix
        self.log = logTextFile

    # The following code serves to compute quadratic curves that describe our lines of interest
    def computeLine(self, a, b, cBounds, plotGraph=False, cPlotVal=1450, plotResults=False):

        """
        The aim of this function will be to paramaterise a curve and then perform a line integral over it
        perhaps with some thickness of pixels and iterate this over the image. This will return gaussian forms
        where the lines are which we will the aim to maximise the peak and minimise the standard deviation
        """

        # Assuming the curve to be parabolic of the form x = ay**2 + b*y + c

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
                    testImMat[y, x] = np.max(imVeryClear)

            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1), plt.imshow(self.imMat, cmap='hot'), plt.title(
                'Original Image on which the quadratic is fitted')
            plt.subplot(1, 2, 2), plt.imshow(imVeryClear + testImMat, cmap='hot'), plt.title(
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
                        totVal += self.imMat[yL, xL + 1]
                    if 0 < xWidth:
                        totVal += self.imMat[yL, xL - 1]

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

        lineIntegralList = self.computeLine(a, b, cBounds)
        cVals = np.array(lineIntegralList)[:, 0]
        lineIntegralVals = np.array(lineIntegralList)[:, 1]

        def gaussian(X, amp, xpeak, sigma, c_offset):
            return amp * np.exp(-(X - xpeak) ** 2 / (2 * sigma ** 2)) + c_offset

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

            return amp_guess, 0, xpeak_guess

    def optimiseLines(self, aBounds, bBounds, cBounds, sigmaWeighting=1, plotGraph=False, plotResults=False):

        # Taking Compute Line we wish to fit a gaussian to the curve produced then
        # maximise the amplitude while minimising the deviation

        def lossFunc(params, sigmaWeighting):

            print("-" * 40)
            A, B = params

            amp, sigma, cPeak = self.fitGaussianToLineIntegral(A, B, cBounds)
            print("Amplitude", amp)
            print("Sigma", sigma)
            print("Cpeak", cPeak)

            # Aiming to maximise amp while minimising sigma
            print("Loss", -amp + sigmaWeighting * sigma)
            return -amp + sigmaWeighting * sigma

        bounds = [aBounds, bBounds]
        initial_guess = np.array([(aBounds[0] + aBounds[1]) / 2, (bBounds[0] + bBounds[1]) / 2])

        result = minimize(lossFunc, initial_guess, args=(sigmaWeighting,), bounds=bounds, method='L-BFGS-B')

        A, B = result.x
        lossOptimised = lossFunc(result.x, sigmaWeighting)

        amp, sigma, cPeak = self.fitGaussianToLineIntegral(A, B, cBounds, plotGauss=True)

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

    def matrixWithLinesOptimisation(self, plotLines=False):

        # The following ranges are for image 8. They are expected to work for any image with data
        cBoundsLeft = (1200, 1360)
        aBoundsLeft = (0.00001, 0.0001)
        bBoundsLeft = (700, 1024)

        cBoundsRight = (1380, 1460)
        aBoundsRight = (0.00001, 0.0001)
        # The Line ends further along x at the bottom than it does y
        bBoundsRight = (700, 1024)

        ALeft, BLeft, cLeft = self.optimiseLines(aBoundsLeft, bBoundsLeft, cBoundsLeft, plotGraph=False,
                                                 plotResults=False)
        ARight, BRight, cRight = self.optimiseLines(aBoundsRight, bBoundsRight, cBoundsRight, plotGraph=False,
                                                    plotResults=False)

        leftLineMat = self.matrixWithLines(ALeft, BLeft, cLeft, plotLines=False)
        rightLineMat = self.matrixWithLines(ARight, BRight, cRight, plotLines=False)

        if plotLines:
            plt.imshow(leftLineMat + rightLineMat, cmap='hot')
            plt.show()

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
            linesMatrixForPlotting = np.where(linesMatrix > 0, np.max(imVeryClear), 0)

            plt.subplot(1, 2, 1), plt.imshow(self.imMat, cmap='hot'), plt.title(
                'Original Image on which the quadratic is fitted')
            plt.subplot(1, 2, 2), plt.imshow(imVeryClear + linesMatrixForPlotting, cmap='hot'), plt.title(
                'Clearer Image Matrix with curves')
            plt.show()

        return linesMatrix




def optimiseGeometryToCalibratedLines(imageMat, r_thetaval=2.618,logTextFile=None, iterations=30,):
    # linesMat = Calibrate(imageMat).matrixWithLinesOptimisation()

    # For Image 8

    linesMatLeft = Calibrate(imTest, None).matrixWithLines(Aoptimised=6.613794078473409e-05, Boptimised=862,
                                                           Coptimised=1278, plotLines=False)
    linesMatRight = Calibrate(imTest, None).matrixWithLines(Aoptimised=7.063102423636962e-05, Boptimised=862,
                                                            Coptimised=1418, plotLines=False)

    def lossFunction(params):
        p = params

        geo = Geometry(crystal_pitch=p[0], crystal_roll=p[1],
                       camera_pitch=p[2], camera_roll=p[3],
                       r_camera_spherical=np.array([p[4], r_thetaval, np.pi]))

        alphaLineCoords = geo.xy_coords_of_E(E_Lalpha_eV)  # More Right / right line
        betaLineCoords = geo.xy_coords_of_E(E_Lbeta_eV)    # More Left / left line

        # If we are optimising r camera to be that going to the centre of the camera then the 0,0 pixel is displaced by:
        x_0 = - geo.xWidth / 2
        y_0 = + geo.yWidth / 2

        lossPositive = 0

        # Treat each Line Separately as not to over encourage curvature
        for row in alphaLineCoords:
            x_pixel = round((row[0] - x_0) / geo.pixelWidth)
            y_pixel = round((y_0 - row[1]) / geo.pixelWidth)

            lossPositive += linesMatRight[y_pixel, x_pixel]

        for row in betaLineCoords:
            x_pixel = round((row[0] - x_0) / geo.pixelWidth)
            y_pixel = round((y_0 - row[1]) / geo.pixelWidth)

            lossPositive += linesMatLeft[y_pixel, x_pixel]

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


    initialGuess = np.array([-0.7141186367791281,   # crystal pitch
                             -0.0025759119106860723,   # crystal roll
                             1.22,   # Camera pitch, pi/4 is ~ 0.785
                             -0.00509203896479344,   # camera roll
                             0.19  # r camera
                             ])

    bounds = [(None, None),  # crystal pitch Bounds
              (-0.002, 0.003),  # crystal roll Bounds
              (0, None),  # Camera pitch Bounds
              (0, 0),  # camera roll Bounds
              (0.18, 0.21),  # rcamBounds
              ]

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
    linesMatGeoOptimised = geoOptimised.createLinesMatrix(imageMat, 1)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1), plt.imshow(linesMatLeft+linesMatRight, cmap='hot'), plt.title('Quadratic Lines')
    plt.subplot(1, 2, 2), plt.imshow(linesMatGeoOptimised + linesMatGeoOptimised, cmap='hot'), plt.title(
        'Geometic Lines plotted Over')
    plt.show()

    def logResults():
        append_to_file(logTextFile, "-" * 30)
        append_to_file(logTextFile, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        append_to_file(logTextFile, f"crystal pitch bounds = {bounds[0]}, crystal roll bounds = {bounds[1]}")
        append_to_file(logTextFile, f"camera pitch bounds = {bounds[2]}, camera roll bounds = {bounds[3]}")
        append_to_file(logTextFile, f"rcamBounds = {bounds[4]}")

        append_to_file(logTextFile,f"optimised rystal pitch = {crystal_pitch}, n crystal roll = {crystal_roll}")
        append_to_file(logTextFile,f"optimised camera pitch = {camera_pitch}, n camera roll = {camera_roll}")
        append_to_file(logTextFile,f"optimised rcam spherical = {rcamSphericalOptimised}")

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


def optimiseGeometryToCalibratedLibesCMA(imageMat,r_thetaval=2.62, logTextFile=None):
    # For Image 8
    linesMatLeft = Calibrate(imTest, None).matrixWithLines(Aoptimised=6.613794078473409e-05, Boptimised=862,
                                                           Coptimised=1278, plotLines=False)
    linesMatRight = Calibrate(imTest, None).matrixWithLines(Aoptimised=7.063102423636962e-05, Boptimised=862,
                                                            Coptimised=1418, plotLines=False)

    def lossFunction(params):
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

        for row in betaLineCoords:
            x_pixel = round((row[0] - x_0) / geo.pixelWidth)
            y_pixel = round((y_0 - row[1]) / geo.pixelWidth)

            lossPositive += linesMatLeft[y_pixel, x_pixel]

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

    initialGuess = np.array([-6.120306912807198e-05,  # crystal pitch
                             -1.2852834454814307e-05,  # crystal roll
                             0.780098495349171,  # Camera pitch, pi/4 is ~ 0.785
                             0,  # camera roll
                             0.20008739  # r camera
                             ])

    sigma0 = 0.005

    bounds = [[None, None, None, None,  0.08],  # Lower bounds
              [None, None, None, None,  0.25]]  # Upper bounds

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

        append_to_file(logTextFile,f"optimised rystal pitch = {crystal_pitch}, n crystal roll = {crystal_roll}")
        append_to_file(logTextFile,f"optimised camera pitch = {camera_pitch}, n camera roll = {camera_roll}")
        append_to_file(logTextFile,f"optimised rcam spherical = {rcamSphericalOptimised}")

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



# optimiseGeometryToCalibratedLines(imTest,r_thetaval=2.595
#                                   ,logTextFile=geometryLog,iterations=30)
# optimiseGeometryToCalibratedLibesCMA(imTest)

# geo = Geometry(crystal_pitch=-0.19, crystal_roll=0,camera_pitch=0,camera_roll=0,r_camera_spherical=np.array([1,1,1]))
# geo.rayBragg(E_Lalpha_eV,np.pi)


if __name__ == '__main__':

    crysPitch = -0.3344584639102419
    CrysRoll = -0.0025759119106860723
    CamPitch = 1.18
    CamRoll = -0.00509203896479344
    rcamSpherical = np.array([0.09 ,  # r
                              2.60,  # theta
                              np.pi])

    def testPlot():
        print("crysPitch = ",crysPitch, "CrysRoll = ",CrysRoll)
        print("CamPitch = ",CamPitch,"CamRoll = ",CamRoll)
        print("rcamSpherical = ",rcamSpherical)

        geo = Geometry(crysPitch,CrysRoll,CamPitch,CamRoll,rcamSpherical)
        # # geo.visualiseLines()
        geolinesMat = geo.createLinesMatrix(imTest,np.max(imClear),phiStepSize=0.0001)

        plt.imshow(imClear+geolinesMat, cmap="hot")
        plt.title(f"crystal: Pitch = {crysPitch}, Roll = {CrysRoll}; \ncamera: Pitch = {CamPitch}, Roll = {CamRoll}; \nr_camera = {rcamSpherical}")
        plt.show()

    testPlot()

    def showTwoLines():
        linesMatLeft = Calibrate(imTest, None).matrixWithLines(Aoptimised=6.613794078473409e-05, Boptimised=862,
                                                               Coptimised=1278, plotLines=False)
        linesMatRight = Calibrate(imTest, None).matrixWithLines(Aoptimised=7.063102423636962e-05, Boptimised=862,
                                                                Coptimised=1418, plotLines=False)

        plt.imshow(imVeryClear + np.where(linesMatLeft>0,np.max(imVeryClear),0) + np.where(linesMatRight>0,np.max(imVeryClear),0)), plt.show()

    # showTwoLines()

    pass
