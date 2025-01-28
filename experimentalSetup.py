import numpy as np
import pandas as pd

from getImageData import *
from tools import *


class Geometry:
    def __init__(self, imageMatrix,
                 xpixels = 2048, ypixels = 2048, pixelWidth=pixel_width,
                 n_crystal=np.array([0, 0, 1]),
                 n_camera=np.array([0.1, 0, 1]),
                 r_camera_spherical=np.array([0.06, np.pi/2+0.3, np.pi]),
                 ):
        self.xpixels = xpixels
        self.ypixels = ypixels,
        self.pixelWidth = pixelWidth
        self.xWidth = xpixels * pixelWidth
        self.yWidth = ypixels * pixelWidth
        self.nCrystal = n_crystal
        self.nCam = n_camera
        self.r_cam = spherical_to_cartesian(r_camera_spherical)
        self.calibrateGeometry(imageMatrix)


    def calibrateGeometry(self, imageMatrix,
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
        print("rcam", self.r_cam)
        print("nCrystal", self.nCrystal)

        rotMatrix_cry = inverseRotation_matrix(self.nCrystal)

        # The source is considered to be at the origin
        theta_alpha = bragg_E_to_theta(E_Lalpha)
        theta_beta = bragg_E_to_theta(E_Lbeta)


        # The allowed directional vectors in spherical coordinate are then (1,theta, [0,2 pi])

        df_xyToSum = []
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

            if ((abs(x_planeA) < (self.xWidth-self.pixelWidth)/2).all() and
                    (abs(x_planeB) < (self.xWidth-self.pixelWidth)/2).all() and
                    (abs(y_planeA) < (self.yWidth-self.pixelWidth)/2).all() and
                    (abs(y_planeB) < (self.yWidth-self.pixelWidth)/2).all()):
                # print(x_planeA / self.pixelWidth, y_planeA / self.pixelWidth)
                df_xyToSum.append([x_planeA, y_planeA,x_planeB, y_planeB])

        matrix_test = np.zeros((2048, 2048))
        print(df_xyToSum)
        for row in df_xyToSum:

            x_0 = - self.xWidth/2
            y_0 = + self.yWidth/2

            x_pixelA = round((row[0]-x_0)/self.pixelWidth)
            y_pixelA = round((y_0 - row[1])/self.pixelWidth)
            x_pixelB = round((row[2] - x_0) / self.pixelWidth)
            y_pixelB = round((y_0 - row[3]) / self.pixelWidth)

            print(y_pixelA, x_pixelA)
            matrix_test[y_pixelA, x_pixelA] = 300
            print(y_pixelB, x_pixelB)
            matrix_test[y_pixelB, x_pixelB] = 300


        plt.imshow(matrix_test+imageMatrix,cmap="hot",)
        plt.show()



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


geo = Geometry(array8Test)



