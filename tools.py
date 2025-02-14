import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from constants import *

def bragg_E_to_theta(E_eV,d=d_beryl):
    """
    Takes input Energy (in eV) and returns the associated angle (in rad)
    given a beryl 10-10 crystal.
    The angle is with respect to the plane of the crystal

    """

    sint = (h_planck*c)/(2*d*E_eV*q_e)

    return np.arcsin(sint)


def bragg_theta_to_E(theta_rad, d=d_beryl):
    """
    Takes input angle (in rad) and outputs an energy in eV.
    The angle is with respect to the surface of a crystal which is
    supposed to be that of a Beryl 10-10 crystal
    """

    return (h_planck*c)/(2*d*np.sin(theta_rad)*q_e)


def bragg_vectorised(E_eV,n_crystal,d=d_beryl):

    # We demand that n dot v (normalised) give sin t where t is given by the bragg condition

    sint = (h_planck*c)/(2*d*E_eV*q_e)

    n_crys_normalised = n_crystal / np.linalg.norm(n_crystal)



# Geometry

def cartesian_to_spherical(r_cart):
    """
    Inputs: numpy array r_cart with x,y,z coordinates in metres
    returns r (m),theta and phi (both rad) that defines this point in spherical coordinates
    """
    x = r_cart[0]
    y = r_cart[1]
    z = r_cart[2]

    xsqr_Plus_ysqr = x**2 + y**2

    # Note for future self: DO NOT just use arctan it does not properly account for signs
    r = np.sqrt(xsqr_Plus_ysqr + z**2)
    theta = np.arctan2(np.sqrt(xsqr_Plus_ysqr),z)
    phi = np.arctan2(y,x)

    return np.array([r,theta,phi])

def spherical_to_cartesian(r_spherical):
    """
    Inputs: numpy array r_spherical with r,tgeta,phi coordinates in metres and radians respectively
    returns x,y,z (m) that defines this point in cartesian coordinates
    """
    r = r_spherical[0]
    theta = r_spherical[1]
    phi = r_spherical[2]

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return np.array([x,y,z])


def Rotation_matrix(n_vector, n_final=np.array([0,0,1])):
    """
    Takes a vector n_vector and finds the matrix to rotate it to n_final
    which is taken normally to be the z axis
    """

    n_v = n_vector / np.linalg.norm(n_vector)
    n_f = n_final / np.linalg.norm(n_final)

    if np.array_equal(n_v,n_f):
        # print("The two vectors are the same")
        return np.eye(3)
    elif np.array_equal(n_v,-n_f):
        # print("The two vectors are the 180 degree rotations of one another")
        return -np.eye(3)
    else:
        axisRot = np.cross(n_v,n_f)
        axis_R = axisRot / np.linalg.norm(axisRot)

        thetaRot = np.arccos(np.dot(n_v,n_f))

        # It can be shown using linear algebra that a rotation matrix is given by
        # Identity Matrix + (1- cos(theta))(axis_rot dot J matrix)^2 + sin(theta) (axis_rot dot J matrix)
        # J is a vector with the different directions of rotation matrices encoded within it

        udotJ = np.array([
            [0, -axis_R[2], axis_R[1]],
            [axis_R[2],0,-axis_R[0]],
            [-axis_R[1],axis_R[0],0]
        ])


        RotMat = np.eye(3) + (1-np.cos(thetaRot)) * np.dot(udotJ,udotJ) + np.sin(thetaRot) * udotJ

        return RotMat

def inverseRotation_matrix(n_vector, n_final=np.array([0,0,1])):
    """
    Takes a vector n_vector and finds the matrix to rotate it to n_final
    which is taken normally to be the z axis
    """

    n_v = n_vector / np.linalg.norm(n_vector)
    n_f = n_final / np.linalg.norm(n_final)

    if np.array_equal(n_v,n_f):
        # print("The two vectors are the same")
        return np.eye(3)
    elif np.array_equal(n_v,-n_f):
        # print("The two vectors are the 180 degree rotations of one another")
        return -np.eye(3)

    axisRot = np.cross(n_v,n_f)
    axis_R = axisRot / np.linalg.norm(axisRot)

    thetaRot = np.arccos(np.dot(n_v,n_f))

    # It can be shown using linear algebra that a rotation matrix is given by
    # Identity Matrix + (1- cos(theta))(axis_rot dot J matrix)^2 + sin(theta) (axis_rot dot J matrix)
    # J is a vector with the different directions of rotation matrices encoded within it

    udotJ = np.array([
        [0, -axis_R[2], axis_R[1]],
        [axis_R[2],0,-axis_R[0]],
        [-axis_R[1],axis_R[0],0]
    ])

    # As we're looking for the rotation matrix in inverse take -thetaRot:

    inverseRot = np.eye(3) + (1-np.cos(thetaRot)) * np.dot(udotJ,udotJ) + np.sin(-thetaRot) * udotJ

    return inverseRot

def ray_in_planeCamera(v_ray_cart, n_camera_cart, r_camera_cart):
    """
    The plane is defined by r dot n = a_onPlane dot n

    To find where the ray hits the plane we wish to solve this equation for the ray and
    then convert this point into x,y pixel for the camera.

    For some scaling of the unit ray vector D * v_ray = r there is a solution
    """
    D = np.dot(r_camera_cart, n_camera_cart) / np.dot(v_ray_cart, n_camera_cart)

    # r_camera is being optimised as the center of the plane, finding the vector away from this

    r_inPlane = D * v_ray_cart - r_camera_cart

    # Considering this in the plane of the camera
    # Finding the rotation matrix
    rotMatrix_cam = Rotation_matrix(n_camera_cart)

    r_planePrime = np.dot(rotMatrix_cam, r_inPlane)
    x_plane = r_planePrime[0]
    y_plane = r_planePrime[1]

    return x_plane,y_plane


def solidAngleNormalisation(energy_eV, n_crystal, n_camera, r_camera):
    """
    For a given value of energy we have an associated radius and angle and hence
    can compute the fractional normalisation. The true count is the count incident
    on the detector / this fractional normalisation

    This factor is given by (2 * phi * sin(theta)) / (4 pi)
    """


    # Given a value of radius r, due to the size of the CCD each radius has a different range of
    # phi that it is allowed 2 phi sin theta / 4 pi


    # rotation matrix from bragg crystal to our coordinate system
    rotMatrix_cry = inverseRotation_matrix(n_crystal)

    # The source is considered to be at the origin
    theta = bragg_E_to_theta(energy_eV)

    # The allowed directional vectors in spherical coordinate are then (1,theta, [0,2 pi])

    for phi in np.arange(np.pi / 2, 3 * np.pi / 2, 0.0001):

        v_ray_prime = spherical_to_cartesian(np.array([1, np.pi / 2 + theta, phi]))

        v_ray = np.dot(rotMatrix_cry, v_ray_prime)

        v_ray_spherical = cartesian_to_spherical(v_ray)

        # TODO: consider whether this geometry is entirely correct

        print(v_ray_spherical)

    phiRange = 1
    valTheta = 1

    fractional_normalisation = (2*phiRange * np.sin(valTheta)) / (4 * np.pi)

    return fractional_normalisation

# solidAngleNormalisation(1200,np.array([0.0001,0,1]),None,None)

def energy_to_pixel(energy_eV,n_crystal,n_camera, r_camera_spherical,
                    xpixels=2048, ypixels=2048, pixelWidth=pixel_width,
                    ):

    # First working within the frame where the crystal is orientated with the z axis parallel to its norm:
    theta_alpha = bragg_E_to_theta(energy_eV)
    # The rotation matrix from this frame to our general coordinate system is
    rotMatrix_crystaltoNormal = inverseRotation_matrix(n_crystal)

    df_xy_inCamPlane =[]
    for phi in np.arange(0, 2 * np.pi, 0.0001):

        # The vector of the ray in the crystal orientated 0,0,1
        v_ray_prime = spherical_to_cartesian(np.array([1, np.pi / 2 + theta_alpha, phi]))

        v_ray = np.dot(rotMatrix_crystaltoNormal, v_ray_prime)

        x_plane, y_plane = ray_in_planeCamera(v_ray_cart=v_ray, n_camera_cart=n_camera, r_camera_cart=cartesian_to_spherical(r_camera_spherical))

        if ((abs(x_plane) < (xpixels-1)*pixelWidth / 2).all() and
                (abs(y_plane) < (ypixels-1)*pixelWidth / 2).all()):
            df_xy_inCamPlane.append([x_plane, y_plane])

def pixel_to_energy(x_pixel,y_pixel,
                    n_crystal,n_camera,r_camera_spherical,
                    xpixels=2048, ypixels=2048, pixelWidth=pixel_width,
                    ):
    pass

def nVectorFromEuler(pitch_rad,roll_rad):

    nx = np.sin(pitch_rad)
    ny = - np.cos(pitch_rad) * np.sin(roll_rad)
    nz = np.cos(pitch_rad) * np.cos(roll_rad)

    return np.array([nx,ny,nz])

def EulerfromNVector(n_vector):
    nx = n_vector[0]
    ny = n_vector[1]
    nz = n_vector[2]

    pitch_rad = np.arctan(ny/nx)
    roll_rad = np.arctan(nz/np.sqrt(nx**2+ny**2))

    return pitch_rad,roll_rad

def rotMatrixUsingEuler(pitch_rad,roll_rad):

    # Here using the euler angles to rotate from 0 0 1 to the normal vector
    # The way that I have chose by x,y,z implies a rotation around the y axis first then around the x axis

    rotMatAroundY = np.array([
        [np.cos(pitch_rad),  0, np.sin(pitch_rad)],
        [0,              1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])

    rotMatAroundX = np.array([
        [1, 0,          0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])

    return np.dot(rotMatAroundX,rotMatAroundY)

def InverseRotMatrixUsingEuler(pitch_rad,roll_rad):

    # Change Sign and order
    rotMatBackAroundX = np.array([
        [1, 0,          0],
        [0, np.cos(roll_rad), np.sin(roll_rad)],
        [0, -np.sin(roll_rad), np.cos(roll_rad)]
    ])

    rotMatBackAroundy = np.array([
        [np.cos(pitch_rad),   0, -np.sin(pitch_rad)],
        [0,               1, 0],
        [np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])

    # First rotated around Y then X so to undo we want reverse X then reverse y

    return np.dot(rotMatBackAroundy,rotMatBackAroundX)

# Checked that these rotate 001 to the same as nVector from Euler, and it is the inverse for that process for pi/4 both
# print(nVectorFromEuler(-0.19,0))
# print(np.dot(rotMatrixUsingEuler(-0.19,0),np.array([0,0,1])))


minimise_count = 0
def callbackminimise(params):
    global minimise_count
    minimise_count += 1
    print("-"*40)
    print(f"Iteration {minimise_count}")
    print(params)


class Visualise:

    @staticmethod
    def spectrum(df,energyCol="Energy",countIntensityCol="Count"):

        sns.lineplot(data=df,x=energyCol, y=countIntensityCol)
        plt.show()

def append_to_file(file_path, text):
    with open(file_path, "a", encoding="utf-8") as file:
        file.write("\n"+ text)