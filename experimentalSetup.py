#%%

import numpy as np
from scipy.optimize import minimize

from getImageData import *
from tools import *


n_crytest = np.array([3.33236883e-05, -9.42804784e-05,1])
n_cam_test = np.array([1.27941744e-04 ,6.97820216e-05,1])
r_cam_spher_test = np.array([0.05984086, 1.87545166, np.pi])

class GeometryCalibration:
    def __init__(self,
                 xpixels = 2048, ypixels = 2048, pixelWidth=pixel_width,
                 n_crystal=np.array([0, 0, 1]),
                 n_camera=np.array([0.2, 0, 1]),
                 r_camera_spherical=np.array([0.06, np.pi/2+0.3, np.pi]),
                 ):
        self.xpixels = xpixels
        self.ypixels = ypixels,
        self.pixelWidth = pixelWidth
        self.xWidth = xpixels * pixelWidth
        self.yWidth = ypixels * pixelWidth
        self.nCrystal = n_crystal
        self.nCam = n_camera
        self.r_cam_spherical = r_camera_spherical
        self.r_cam = spherical_to_cartesian(r_camera_spherical)

    def computeGeometry_loss(self, imageMatrix, printImage=False,
                          E_Lalpha=E_Lalpha_eV, E_Lbeta=E_Lbeta_eV):
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

        print("ncam",self.nCam)
        print("rcam_spherical",self.r_cam_spherical)
        print("rcam", self.r_cam)
        print("nCrystal", self.nCrystal)
        # maxValImage = max(imageMatrix.flatten())
        maxValImage = 1

        rotMatrix_cry = inverseRotation_matrix(self.nCrystal)

        # The source is considered to be at the origin
        theta_alpha = bragg_E_to_theta(E_Lalpha)
        theta_beta = bragg_E_to_theta(E_Lbeta)


        # The allowed directional vectors in spherical coordinate are then (1,theta, [0,2 pi])

        df_alpha_xyToSum = []
        df_beta_xyToSum = []
        count = 0
        for phi in np.arange(0, 2*np.pi, 0.0001):
            count += 1

            v_ray_alpha_prime = spherical_to_cartesian(np.array([1,np.pi/2 + theta_alpha,phi]))
            v_ray_beta_prime = spherical_to_cartesian(np.array([1,np.pi/2 + theta_beta,phi]))

            v_ray_alpha = np.dot(rotMatrix_cry,v_ray_alpha_prime)
            v_ray_beta = np.dot(rotMatrix_cry, v_ray_beta_prime)

            # if count % 50 == 0:
            #     print(count)
            #     print(rotMatrix_cry)
            #     print(v_ray_alpha)

            x_planeA, y_planeA = self.ray_in_planeCamera(v_ray=v_ray_alpha)
            x_planeB, y_planeB = self.ray_in_planeCamera(v_ray=v_ray_beta)

            if ((abs(x_planeA) < (self.xWidth - self.pixelWidth) / 2).all() and
                    (abs(y_planeA) < (self.yWidth - self.pixelWidth) / 2).all()):
                df_alpha_xyToSum.append([x_planeA, y_planeA,x_planeB, y_planeB])

            if ((abs(x_planeB) < (self.xWidth - self.pixelWidth) / 2).all() and
                    (abs(y_planeB) < (self.yWidth - self.pixelWidth) / 2).all()):
                df_beta_xyToSum.append([x_planeA, y_planeA, x_planeB, y_planeB])

        matrix_test = np.zeros((2048, 2048))
        loss = 0
        # print(df_xyToSum)
        for rowAlpha,rowBeta in zip(df_alpha_xyToSum,df_beta_xyToSum):

            x_0 = - self.xWidth/2
            y_0 = + self.yWidth/2

            x_pixelA = round((rowAlpha[0]-x_0)/self.pixelWidth)
            y_pixelA = round((y_0 - rowAlpha[1])/self.pixelWidth)
            x_pixelB = round((rowBeta[2] - x_0) / self.pixelWidth)
            y_pixelB = round((y_0 - rowBeta[3]) / self.pixelWidth)

            # print(y_pixelA, x_pixelA)
            matrix_test[y_pixelA, x_pixelA] = maxValImage
            # print(y_pixelB, x_pixelB)
            matrix_test[y_pixelB, x_pixelB] = maxValImage

            loss += (
                imageMatrix[y_pixelA, x_pixelA] + imageMatrix[y_pixelB, x_pixelB]
            )

        print("num points",sum(matrix_test.flatten()))
        # if int(sum(matrix_test.flatten())) == 0:
        #     raise ValueError

        # matdif = (imageMatrix - matrix_test)**2
        # loss = sum(matdif.flatten())


        lossNeg = - loss

        if printImage:
            # Just curve
            # plt.imshow(matrix_test, cmap="hot", )
            # plt.show()
            # Raw with curve
            matTest = np.where(matrix_test > 0, 50, 0)

            plt.imshow(matTest + imageMatrix, cmap="hot", )
            plt.show()


        return lossNeg


    def ray_in_planeCamera(self,v_ray):
        """
        The plane is defined by r dot n = a_onPlane dot n

        To find where the ray hits the plane we wish to solve this equation for the ray and
        then convert this point into x,y pixel for the camera.

        For some scaling of the unit ray vector D * v_ray = r there is a solution
        """
        D = np.dot(self.r_cam,self.nCam)/np.dot(v_ray,self.nCam)

        # r_camera is being optimised as the center of the plane, finding the vector away from this

        r_inPlane = D*v_ray - self.r_cam

        # Considering this in the plane of the camera
        # Finding the rotation matrix

        rotMatrix_cam = Rotation_matrix(self.nCam)

        r_planePrime = np.dot(rotMatrix_cam, r_inPlane)
        x_plane = r_planePrime[0]
        y_plane = r_planePrime[1]

        # print("z", r_planePrime[2], "should be 0")
        # tj

        return x_plane,y_plane


def optimiseGeometry(imageMatrix,iterations=30):

    def lossFunction(params):

        p = params
        geoClass = GeometryCalibration(n_crystal=np.array([p[0], p[1], 1]),
                 n_camera=np.array([p[2], p[3], 1]),
                 r_camera_spherical=np.array([p[4], p[5], np.pi]))

        loss = geoClass.computeGeometry_loss(imageMatrix=imageMatrix)

        print("loss",loss)
        return loss

    initialGuess = np.array([0,0,
                             0,0,
                             0.06,np.pi / 2 + 0.3])

    ncrysxBounds = (None, None)
    ncrysyBounds = (None, None)

    ncamxBounds = (0, None)
    ncamyBounds = (None, None)

    rcamBounds= (0.05, 0.08)
    thetacamBounds = (None, None)

    bounds = [ncrysxBounds,ncrysyBounds,
          ncamxBounds,ncamyBounds,
          rcamBounds,thetacamBounds
          ]
    # Perform optimization
    result = minimize(lossFunction, initialGuess,bounds=bounds, method='Nelder-Mead',options={'maxiter': iterations})

    # Optimized parameters
    optimized_params = result.x
    print("Optimized Parameters:")
    print(f"n crystal: {np.array([optimized_params[0], optimized_params[1], 1])}")
    print(f"n camera: {np.array([optimized_params[2], optimized_params[3], 1])}")
    print(f"r: {optimized_params[4]}")
    print(f"theta: {optimized_params[5]}")

    geo = GeometryCalibration(n_crystal=np.array([optimized_params[0], optimized_params[1], 1]),
                 n_camera=np.array([optimized_params[2], optimized_params[3], 1]),
                 r_camera_spherical=np.array([optimized_params[4], optimized_params[5], np.pi]))
    geo.computeGeometry_loss(imageMatrix=imageMatrix,printImage=True,)


# optimiseGeometry(reducedCovTest)

geo = GeometryCalibration(n_crystal = n_crytest,
                 n_camera = n_cam_test,
                 r_camera_spherical= r_cam_spher_test)
geo.computeGeometry_loss(high_intensity_points,True)



# %%
