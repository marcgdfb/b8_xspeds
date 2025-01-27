import matplotlib.pyplot as plt
import seaborn as sns

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

# Assuming that the crystal and detector lie parallel. This is the most logical
# geometric setup

# checked with sketchup
def height_source_detector(E_min_detector=E_min_eV,E_max_detector=E_max_eV,l_sep=length_detector):

    t1 = np.tan(bragg_E_to_theta(E_min_detector))
    t2 = np.tan(bragg_E_to_theta(E_max_detector))

    R_1 = l_sep/((t1/t2) - 1)
    h = R_1*t1

    return h

def radius_of_energy(energy_eV):

    theta = bragg_E_to_theta(energy_eV)
    return height_source_detector()/np.tan(theta)

def energy_of_radius(radius_metres):

    t_theta = height_source_detector()/radius_metres

    return bragg_theta_to_E(np.arctan(t_theta))



# TODO: The way that this calculates Emax,min is not correct. They need to be in the direction orthogonal to the curve...

def xypixel_observationPlane_to_energy(x_pixel,y_pixel,
                                      r_edge=radius_of_energy(E_min_eV),
                                      num_pixels_x = 2048,
                                      num_pixels_y = 2048):
    """
    The input of the pixel position i.e. the ij element of the
    image matrix is given to find the radius and hence the angle and energy associated with it.

    Currently the code assumes num_pixels x,y is an even number

    r_edge = 0.0544

    returns E, E upper bound, E lower bound
    """
    # First the 0,0 element
    x_00pixel = -r_edge - (num_pixels_x-0.5)*pixel_width
    y_00pixel = ((num_pixels_y/2)-0.5)*pixel_width

    y_point = y_00pixel - y_pixel*pixel_width
    x_point = x_00pixel + x_pixel*pixel_width

    r_point = np.sqrt(x_point**2 + y_point**2)

    E_center = energy_of_radius(r_point)

    # What is the uncertainty on this? Want to find the largest possible value in that given pixel
    # To avoid differentiation can just use geometry here:

    if y_pixel <= (num_pixels_y/2)-1:
        print("Above center of observation plane")
        # Max is in the top left corner
        y_point_Emax = y_point + pixel_width/2
        x_point_Emax = x_point - pixel_width/2
        r_point_Emax = np.sqrt(x_point_Emax**2 + y_point_Emax**2)

        E_upperLim = energy_of_radius(r_point_Emax)

        # Min is in the bottom right corner
        y_point_Emin = y_point - pixel_width/2
        x_point_Emin = x_point + pixel_width/2
        r_point_Emin = np.sqrt(x_point_Emin ** 2 + y_point_Emin ** 2)

        E_lowerLim = energy_of_radius(r_point_Emin)

    if y_pixel >= (num_pixels_y / 2):
        print("Below center of observation plane")
        # Max is now in the bottom left corner
        y_point_Emax = y_point - pixel_width / 2
        x_point_Emax = x_point - pixel_width / 2
        r_point_Emax = np.sqrt(x_point_Emax ** 2 + y_point_Emax ** 2)

        E_upperLim = energy_of_radius(r_point_Emax)

        # Min is in the top right corner
        y_point_Emin = y_point + pixel_width / 2
        x_point_Emin = x_point + pixel_width / 2
        r_point_Emin = np.sqrt(x_point_Emin ** 2 + y_point_Emin ** 2)

        E_lowerLim = energy_of_radius(r_point_Emin)
    else:
        print()


    return E_center, E_upperLim, E_lowerLim


print(xypixel_observationPlane_to_energy(1023,1023))



# To go the other way we want to have for a given E we obtain a theta,
# the phi will assume an isotropic distribution that will mean sin(theta)
# * change in phi / 2 pi is the relative proportion which is then multiplied by
# the intensity of the spectrum for the simulation

def theta_phi_to_xy_observation(theta,phi,
                                r_edge=radius_of_energy(E_min_eV),
                                num_pixels_x=2048,
                                num_pixels_y=2048
                                ):
    """
    Inputs theta and phi (in rad) and returns the associated pixel
    number in a numpy array with the y number then x number.
    """
    R = height_source_detector()/np.tan(theta)

    x_00pixel = -r_edge - (num_pixels_x - 0.5) * pixel_width
    y_00pixel = ((num_pixels_y / 2) - 0.5) * pixel_width

    x_point = - R*np.cos(phi)
    y_point = R*np.sin(phi)

    difference_y_pixels = round((y_00pixel - y_point)/pixel_width)
    difference_x_pixels = round((x_point - x_00pixel)/pixel_width)

    return np.array([difference_y_pixels,difference_x_pixels])

# proportion of phi think d theta sin theta d phi / 4pi


def cartesian_to_spherical(r_cart):
    """
    Inputs: list r with x,y,z coordinates in metres
    returns r,theta and phi that define this point in spherical coordinates
    """
    xsqr_plus_ysqr = r_cart[0]**2 + r_cart[1]**2
    r = np.sqrt( xsqr_plus_ysqr + r_cart[2]**2)
    theta = np.arctan( np.sqrt(xsqr_plus_ysqr)/r_cart[2])
    phi = np.arctan(r_cart[1]/r_cart[0])

    return np.array([r,theta,phi])

def spherical_to_cartesian(r_spherical):
    """"""


class Visualise:

    @staticmethod
    def spectrum(df,energyCol="Energy",countIntensityCol="Count"):

        sns.lineplot(data=df,x=energyCol, y=countIntensityCol)
        plt.show()