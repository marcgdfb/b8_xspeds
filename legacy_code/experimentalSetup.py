
from scipy.optimize import minimize,curve_fit
from imagePreProcessing import *
from tools import *
from itertools import product
import time
import pandas as pd



class GeometryCalibration:
    def __init__(self,
                 xpixels=2048, ypixels=2048, pixelWidth=pixel_width,
                 n_crystal=np.array([0, 0, 1]),
                 n_camera=np.array([0.2, 0, 1]),
                 r_camera_spherical=np.array([0.06, np.pi / 2 + 0.3, np.pi]),
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

    def computeGeometry_loss(self, imageMatrix, printImage=False,
                             E_Lalpha=E_Lalpha_eV, E_Lbeta=E_Lbeta_eV,
                             rectWidth=3,rectHeight=5,shape=None,penalise=False):
        """
        The purpose of this function is to iterate possible values of
        normal vectors and coordinate positions of the crystal to find
        using the two known emission lines how to find the orientations
        :return:

        """

        # Structure of Function
        """
        Find possible directions of ray given n_crystal. This will involve finding the rotation matrix to map
        n_crystal onto the z axis. Deploying braggs law using this to find the possible directions allowed then rotating these back 
        
        The value of n_camera and r_camera then come into play
        """

        print("ncam", self.nCam)
        print("rcam_spherical", self.r_cam_spherical)
        print("rcam", self.r_cam_cart)
        print("nCrystal", self.nCrystal)
        maxValImage = np.max(imageMatrix)

        rotMatrix_cry = inverseRotation_matrix(self.nCrystal)

        # The angles in the frame where the crystal is orientated with normal parallel to the z axis
        theta_alpha = bragg_E_to_theta(E_Lalpha)
        theta_beta = bragg_E_to_theta(E_Lbeta)

        # The allowed directional vectors in spherical coordinate are then (1,theta, [0,2 pi])

        df_alpha_xyToSum = []
        df_beta_xyToSum = []
        for phi in np.arange(np.pi/2, 3*np.pi/2, 0.0001):

            v_ray_alpha_prime = spherical_to_cartesian(np.array([1, np.pi / 2 + theta_alpha, phi]))
            v_ray_beta_prime = spherical_to_cartesian(np.array([1, np.pi / 2 + theta_beta, phi]))

            # The source is considered to be at the origin
            v_ray_alpha = np.dot(rotMatrix_cry, v_ray_alpha_prime)
            v_ray_beta = np.dot(rotMatrix_cry, v_ray_beta_prime)

            x_planeA, y_planeA = ray_in_planeCamera(v_ray_cart=v_ray_alpha, n_camera_cart=self.nCam, r_camera_cart=self.r_cam_cart)
            x_planeB, y_planeB = ray_in_planeCamera(v_ray_cart=v_ray_beta, n_camera_cart=self.nCam, r_camera_cart=self.r_cam_cart)

            if ((abs(x_planeA) < (self.xWidth - self.pixelWidth) / 2).all() and
                    (abs(y_planeA) < (self.yWidth - self.pixelWidth) / 2).all()):
                df_alpha_xyToSum.append([x_planeA, y_planeA, ])

            if ((abs(x_planeB) < (self.xWidth - self.pixelWidth) / 2).all() and
                    (abs(y_planeB) < (self.yWidth - self.pixelWidth) / 2).all()):
                df_beta_xyToSum.append([x_planeB, y_planeB])

        matrix_test = np.zeros((2048, 2048))



        def sumLossWider():
            loss = 0
            x_0 = - self.xWidth / 2
            y_0 = + self.yWidth / 2

            for rowAlpha in df_alpha_xyToSum:
                x_pixel = round((rowAlpha[0] - x_0) / self.pixelWidth)
                y_pixel = round((y_0 - rowAlpha[1]) / self.pixelWidth)

                if shape == "rectangle_old":
                    loss += self.shapeRectangle(imageMatrix,maxValImage,matrix_test,x_pixel, y_pixel)
                else:
                    loss += self.shapeRectangleGeneral(imageMatrix,maxValImage,matrix_test,x_pixel, y_pixel,rectWidth=rectWidth,rectHeight=rectHeight)

            for rowBeta in df_beta_xyToSum:
                x_pixel = round((rowBeta[0] - x_0) / self.pixelWidth)
                y_pixel = round((y_0 - rowBeta[1]) / self.pixelWidth)

                if shape == "rectangle_old":
                    loss += self.shapeRectangle(imageMatrix,maxValImage,matrix_test,x_pixel, y_pixel)
                else:
                    loss += self.shapeRectangleGeneral(imageMatrix,maxValImage,matrix_test,x_pixel, y_pixel,rectWidth=rectWidth,rectHeight=rectHeight)

            return loss

        def limitPenalty(eMax_eV=E_max_eV, eMin_eV=E_min_eV):

            penalty = 0

            theta_Emax = bragg_E_to_theta(eMax_eV)
            theta_Emin = bragg_E_to_theta(eMin_eV)

            df_maxxyToSum = []
            df_min_xyToSum = []

            for phi in np.arange(np.pi / 2, 3 * np.pi / 2, 0.0001):

                v_ray_max_prime = spherical_to_cartesian(np.array([1, np.pi / 2 + theta_Emax, phi]))
                v_ray_min_prime = spherical_to_cartesian(np.array([1, np.pi / 2 + theta_Emin, phi]))

                # The source is considered to be at the origin
                v_ray_max = np.dot(rotMatrix_cry, v_ray_max_prime)
                v_ray_min = np.dot(rotMatrix_cry, v_ray_min_prime)

                x_planeMax, y_planeMax = ray_in_planeCamera(v_ray_cart=v_ray_max, n_camera_cart=self.nCam, r_camera_cart=self.r_cam_cart)
                x_planeMin, y_planeMin = ray_in_planeCamera(v_ray_cart=v_ray_min, n_camera_cart=self.nCam, r_camera_cart=self.r_cam_cart)

                if ((abs(x_planeMax) < (self.xWidth - self.pixelWidth) / 2).all() and
                        (abs(y_planeMax) < (self.yWidth - self.pixelWidth) / 2).all()):
                    df_maxxyToSum.append([x_planeMax, y_planeMax, ])
                    matrix_test[x_planeMax, y_planeMax] = maxValImage

                if ((abs(x_planeMin) < (self.xWidth - self.pixelWidth) / 2).all() and
                        (abs(y_planeMin) < (self.yWidth - self.pixelWidth) / 2).all()):
                    df_min_xyToSum.append([x_planeMin, y_planeMin])
                    matrix_test[x_planeMin, y_planeMin] = maxValImage

                if not df_maxxyToSum:
                    print("df_maxxyToSum empty")
                    penalty += 1000
                if not df_min_xyToSum:
                    print("df_min_xyToSum empty")
                    penalty += 1000

                return penalty


        if penalise:
            print("Penalty is True")
            lossNegPrePenalty = - np.float64(sumLossWider())
            penalty = limitPenalty()
            print("penalty", penalty)
            lossNeg = lossNegPrePenalty + penalty
            print("loss", lossNeg)
        else:
            lossNeg = - np.float64(sumLossWider())
            print("loss", lossNeg)


        if printImage:
            matTest = np.where(matrix_test > 0, maxValImage, 0)
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1), plt.imshow(imageMatrix, cmap='hot'), plt.title('Original Image')
            plt.subplot(1, 2, 2), plt.imshow(matTest + imageMatrix, cmap='hot'), plt.title('With predicted Line')
            plt.show()

        return lossNeg



    def sumLoss(self, df_alpha_xyToSum,df_beta_xyToSum,x_0,y_0,imageMatrix, maxValImage, matrix_test):
        loss = 0
        for rowAlpha in df_alpha_xyToSum:
            x_pixelA = round((rowAlpha[0] - x_0) / self.pixelWidth)
            y_pixelA = round((y_0 - rowAlpha[1]) / self.pixelWidth)

            matrix_test[y_pixelA, x_pixelA] = maxValImage

            loss += imageMatrix[y_pixelA, x_pixelA]

        for rowBeta in df_beta_xyToSum:
            x_pixelB = round((rowBeta[0] - x_0) / self.pixelWidth)
            y_pixelB = round((y_0 - rowBeta[1]) / self.pixelWidth)

            matrix_test[y_pixelB, x_pixelB] = maxValImage

            loss += imageMatrix[y_pixelB, x_pixelB]

        return loss

    def shapeRectangle(self,imageMatrix, maxValImage, matrix_test, xPixel, yPixel):
        counter = 0
        val_tosum = 0
        # Create line for printing
        matrix_test[yPixel, xPixel] = maxValImage
        if xPixel + 1 < self.xpixels:
            val_tosum += imageMatrix[yPixel, xPixel + 1]
            counter += 1
            if yPixel + 1 < self.ypixels:
                val_tosum += imageMatrix[yPixel + 1, xPixel + 1]
                counter += 1
            if yPixel - 1 < self.ypixels:
                val_tosum += imageMatrix[yPixel - 1, xPixel + 1]
                counter += 1
            if yPixel + 2 < self.ypixels:
                val_tosum += imageMatrix[yPixel + 2, xPixel + 1]
                counter += 1
            if yPixel - 2 < self.ypixels:
                val_tosum += imageMatrix[yPixel - 2, xPixel + 1]
                counter += 1

        if xPixel - 1 >= 0:
            val_tosum += imageMatrix[yPixel, xPixel - 1]
            counter += 1
            if yPixel + 1 < self.ypixels:
                val_tosum += imageMatrix[yPixel + 1, xPixel - 1]
                counter += 1
            if yPixel - 1 < self.ypixels:
                val_tosum += imageMatrix[yPixel - 1, xPixel - 1]
                counter += 1
            if yPixel + 2 < self.ypixels:
                val_tosum += imageMatrix[yPixel + 2, xPixel - 1]
                counter += 1
            if yPixel - 2 < self.ypixels:
                val_tosum += imageMatrix[yPixel - 2, xPixel - 1]
                counter += 1

        val_tosum += imageMatrix[yPixel, xPixel]
        counter += 1
        if yPixel + 1 < self.ypixels:
            val_tosum += imageMatrix[yPixel + 1, xPixel]
            counter += 1
        if yPixel - 1 < self.ypixels:
            val_tosum += imageMatrix[yPixel - 1, xPixel]
            counter += 1
        if yPixel + 2 < self.ypixels:
            val_tosum += imageMatrix[yPixel + 2, xPixel]
            counter += 1
        if yPixel - 2 < self.ypixels:
            val_tosum += imageMatrix[yPixel - 2, xPixel]
            counter += 1

        return val_tosum / counter

    def shapeSquare(self, imageMatrix, maxValImage, matrix_test, xPixel, yPixel):
        counter = 0
        val_tosum = 0
        # Create line for printing
        matrix_test[yPixel, xPixel] = maxValImage
        if xPixel + 1 < self.xpixels:
            val_tosum += imageMatrix[yPixel, xPixel + 1]
            counter += 1
            if yPixel + 1 < self.ypixels:
                val_tosum += imageMatrix[yPixel + 1, xPixel + 1]
                counter += 1
            if yPixel - 1 < self.ypixels:
                val_tosum += imageMatrix[yPixel - 1, xPixel + 1]
                counter += 1

        if xPixel - 1 >= 0:
            val_tosum += imageMatrix[yPixel, xPixel - 1]
            counter += 1
            if yPixel + 1 < self.ypixels:
                val_tosum += imageMatrix[yPixel + 1, xPixel - 1]
                counter += 1
            if yPixel - 1 < self.ypixels:
                val_tosum += imageMatrix[yPixel - 1, xPixel - 1]
                counter += 1

        val_tosum += imageMatrix[yPixel, xPixel]
        counter += 1
        if yPixel + 1 < self.ypixels:
            val_tosum += imageMatrix[yPixel + 1, xPixel]
            counter += 1
        if yPixel - 1 < self.ypixels:
            val_tosum += imageMatrix[yPixel - 1, xPixel]
            counter += 1

        return val_tosum / counter

    def shapeRectangleGeneral(self, imageMatrix, maxValImage, matrix_test, xPixel, yPixel, rectWidth, rectHeight):
        """
        Compute the average value in a rectangle region of the image centered at (xPixel, yPixel).

        Parameters:
          imageMatrix: 2D array containing image pixel values.
          maxValImage:  value used for the test line matrix
          matrix_test:  the test line matrix
          xPixel:      x-coordinate of the center pixel.
          yPixel:      y-coordinate of the center pixel.
          rectWidth:   Width of the rectangle (must be an odd number).
          rectHeight:  Height of the rectangle (must be an odd number).

        Returns:
          The average of all valid pixel values in the defined rectangle.
        """
        # Calculate half-dimensions (using integer division; works because rectWidth and rectHeight are odd)
        half_width = int((rectWidth-1) / 2)
        half_height = int((rectHeight-1) / 2)

        # Values that will be iteratively summed for the total value and the number of points
        # in the rectangle we are summing over
        total_value = 0.0
        count = 0

        # Mark the center pixel in matrix_test
        matrix_test[yPixel, xPixel] = maxValImage

        # Iterate over each offset in the rectangle centered at (xPixel, yPixel)
        for dx in range(-half_width, half_width + 1):
            for dy in range(-half_height, half_height + 1):
                new_x = xPixel + dx
                new_y = yPixel + dy

                # Check that (new_x, new_y) is within image bounds.
                if 0 <= new_x < self.xpixels and 0 <= new_y < self.ypixels:
                    total_value += imageMatrix[new_y, new_x]
                    count += 1

        # Prevent division by zero; however, count should always be > 0 if (xPixel, yPixel) is valid.
        return total_value / count if count > 0 else 0


def optimiseGeometry(imageMatrix, iterations=30, printImage=False):

    # TODO add penatly if 1100 and 1600 eV are not included

    def lossFunction(params):
        p = params
        geoClass = GeometryCalibration(n_crystal=np.array([p[0], p[1], 1]),
                                       n_camera=np.array([p[2], p[3], 1]),
                                       r_camera_spherical=np.array([p[4], 1.88, np.pi]),)

        loss = geoClass.computeGeometry_loss(imageMatrix=imageMatrix, rectWidth=7,rectHeight=21,penalise=False)

        return loss

    initialGuess = np.array([0, 0,
                             1, 0,
                             0.2])

    ncrysxBounds = (None, None)
    ncrysyBounds = (None, None)

    ncamxBounds = (0, None)
    ncamyBounds = (0, 0)

    rcamBounds = (0.1, 0.4)
    # rcamBounds = (None, None)

    bounds = [ncrysxBounds, ncrysyBounds,
              ncamxBounds, ncamyBounds,
              rcamBounds
              ]
    # Perform optimization
    result = minimize(lossFunction, initialGuess, bounds=bounds, method='Nelder-Mead', options={'maxiter': iterations},
                      callback=callbackminimise, tol=None, )

    # Optimized parameters
    optimized_params = result.x
    print("Optimized Parameters:")
    print(f"n crystal: {np.array([optimized_params[0], optimized_params[1], 1])}")
    print(f"n camera: {np.array([optimized_params[2], optimized_params[3], 1])}")
    print(f"r: {optimized_params[4]}")

    geo = GeometryCalibration(n_crystal=np.array([optimized_params[0], optimized_params[1], 1]),
                              n_camera=np.array([optimized_params[2], optimized_params[3], 1]),
                              r_camera_spherical=np.array([optimized_params[4], 1.88, np.pi]))
    geo.computeGeometry_loss(imageMatrix=imageMatrix, printImage=printImage,rectWidth=5,rectHeight=21,penalise=True )

    return optimized_params

parabolateBounds = [(None,None), (None,None),()
              ]

def optimiseParabolae(imageMatrix,printImage=False, params_bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, 2047]),threshold=0):
    """
    :param imageMatrix: The XSPEDS image matrix
    :param printImage: True or False, print predicted line over image matrix?
    :param params_bounds: The boundings of the coefficients of the quadratic function.
    Particularly important is the limitation of the c value which can be done to high accuracy by eye.
    :return:
    """
    def quadratic(y,a,b,c):
        return a*y**2 + b*y + c

    y_coords, x_coords = np.where(imageMatrix > threshold)  # Extract pixel location

    # Flipping the y coords so it matches
    y_coords = y_coords.max() - y_coords

    # Fit quadratic curve
    params, covariance = curve_fit(quadratic, y_coords, x_coords,bounds=params_bounds)

    # Extract fitted parameters
    a_fit, b_fit, c_fit = params
    print(f"Fitted parameters: a={a_fit:.6f}, b={b_fit:.6f}, c={c_fit:.6f}")

    # Generate smooth curve for plotting
    y_fit = np.linspace(y_coords.min(), y_coords.max(), 500)
    x_fit = quadratic(y_fit, *params)



    if printImage:
        # Plot results
        plt.figure(figsize=(6, 8))
        plt.imshow(imageMatrix, cmap="inferno", origin="upper")
        plt.scatter(x_coords, y_coords, s=1, color="cyan", label="Extracted Points")
        plt.plot(x_fit, y_fit, color="red", linewidth=2, label="Fitted Quadratic Curve")
        plt.gca().invert_yaxis()  # Align with image coordinates
        plt.legend()
        plt.title("Quadratic Fit to Extracted Curve")
        plt.show()




optimiseGeometry(imTest,iterations=30,printImage=True)


# n_crytest = np.array([0,0,1])
# n_cam_test = np.array([1, 0, 1])
# r_cam_spher_test = np.array([0.23, 1.88, np.pi])
#
# geo = GeometryCalibration(n_crystal=n_crytest,
#                           n_camera=n_cam_test,
#                           r_camera_spherical=r_cam_spher_test)
# geo.computeGeometry_loss(imClear,printImage=True,rectWidth=5,rectHeight=21,penalise=False )



def gridSearchGeometry_toCSV(imageMatrix,
                             n_crystal_bounds=[(-0.01, 0.01), (-0.001, 0.001)],
                             n_camera_bounds=[(0, 0.3), (-0.01, 0.01)],
                             r_bounds=(0.05, 0.1),
                             grid_steps=(21, 11, 11, 3, 51),
                             output_csv='paramSpace_search_results.csv',
                             flush_interval=1000):
    """
    Perform a brute-force grid search over the 5-dimensional parameter space,
    storing the results in a DataFrame and saving to a CSV file.

    Parameters:
        imageMatrix: Data used to evaluate the loss.
        n_crystal_bounds: List of tuples defining the (min, max) for n_crystal x and y.
        n_camera_bounds: List of tuples defining the (min, max) for n_camera x and y.
        r_bounds: Tuple defining the (min, max) for r.
        grid_steps: Tuple defining how many points to sample in each dimension.
        output_csv: Filename for saving the results.
        flush_interval: Number of iterations after which to write intermediate results to disk.

    Returns:
        df: A Pandas DataFrame containing the parameter values and the computed loss.
    """

    # Create the grid values for each parameter
    n_crys_x = np.linspace(n_crystal_bounds[0][0], n_crystal_bounds[0][1], grid_steps[0])
    n_crys_y = np.linspace(n_crystal_bounds[1][0], n_crystal_bounds[1][1], grid_steps[1])
    n_cam_x = np.linspace(n_camera_bounds[0][0], n_camera_bounds[0][1], grid_steps[2])
    n_cam_y = np.linspace(n_camera_bounds[1][0], n_camera_bounds[1][1], grid_steps[3])
    r_vals = np.linspace(r_bounds[0], r_bounds[1], grid_steps[4])

    # Prepare a list to store dictionaries of results.
    results_list = []

    # Calculate total points for progress tracking
    total_points = np.prod(grid_steps)
    count = 0

    # Optionally, if the CSV already exists (e.g., if you are appending to a long-running job)
    try:
        df = pd.read_csv(output_csv)
        results_list = df.to_dict('records')
        print(f"Loaded existing results with {len(results_list)} records.")
    except FileNotFoundError:
        print("No existing results file found; starting a new grid search.")

    # Loop over every combination using itertools.product
    start_time = time.time()
    for params in product(n_crys_x, n_crys_y, n_cam_x, n_cam_y, r_vals):
        count += 1
        # Unpack parameters for clarity
        ncrys_x_val, ncrys_y_val, ncam_x_val, ncam_y_val, r_val = params

        # Instantiate your GeometryCalibration class with the current parameters.
        # (The third element for n_crystal and n_camera is fixed as 1 per your original code.)
        geo = GeometryCalibration(
            n_crystal=np.array([ncrys_x_val, ncrys_y_val, 1]),
            n_camera=np.array([ncam_x_val, ncam_y_val, 1]),
            r_camera_spherical=np.array([r_val, 1.88, np.pi])
        )

        # Compute the loss value using your method.
        loss = geo.computeGeometry_loss(imageMatrix=imageMatrix)


        if loss != -0.0:
            # Append the result as a dictionary.
            results_list.append({
                'n_crys_x': ncrys_x_val,
                'n_crys_y': ncrys_y_val,
                'n_cam_x': ncam_x_val,
                'n_cam_y': ncam_y_val,
                'r': r_val,
                'loss': loss
            })

        # Print progress every 1000 points
        if count % 1000 == 0:
            elapsed = time.time() - start_time
            print(
                f"Processed {count}/{total_points} grid points ({(count / total_points) * 100:.2f}%) in {elapsed:.2f} seconds.")

        # Periodically flush intermediate results to CSV to avoid losing data if interrupted.
        if count % flush_interval == 0:
            df = pd.DataFrame(results_list)
            df.to_csv(output_csv, index=False)
            print(f"Flushed results to {output_csv} after {count} evaluations.")

    # Convert the full results list to a DataFrame
    df = pd.DataFrame(results_list)
    # Write the final DataFrame to CSV
    df.to_csv(output_csv, index=False)
    print(f"Grid search complete. Total points processed: {count}. Results saved to {output_csv}")

    return df

# df_results = gridSearchGeometry_toCSV(imTest)
