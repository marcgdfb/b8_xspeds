from tools import *
from getImageData import *

# The following assumed that the crystal and camera were parallel

def height_source_detector(E_min_detector=E_min_eV,E_max_detector=E_max_eV,l_sep=length_detector):

    t1 = np.tan(bragg_E_to_theta(E_min_detector))
    t2 = np.tan(bragg_E_to_theta(E_max_detector))

    R_1 = l_sep/((t1/t2) - 1)
    h = R_1*t1

    return h

# print(height_source_detector())

def radius_of_energy(energy_eV,E_min_detector=E_min_eV,E_max_detector=E_max_eV,l_sep=length_detector):

    theta = bragg_E_to_theta(energy_eV)
    return height_source_detector(E_min_detector,E_max_detector,l_sep)/np.tan(theta)

def energy_of_radius(radius_metres):

    t_theta = height_source_detector()/radius_metres

    return bragg_theta_to_E(np.arctan(t_theta))


# The way that this calculates Emax,min is not correct. They need to be in the direction orthogonal to the curve...

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

# To go the other way we want to have for a given E we obtain a theta,
# the phi will assume an isotropic distribution that will mean sin(theta)
# * change in phi / 2 pi is the relative proportion which is then multiplied by
# the intensity of the spectrum for the simulation

def phiHalf(energy_eV, widthDetector=length_detector):
    """
    For a given energy, by virtue of the geometry, there is a range of phi values
    that are possible. I will take this to be measured from the optical axis such
    that there are values between - phi_half(E) and + phi_half(E) to make a total
    of phi(E)

    """
    radius = radius_of_energy(energy_eV=energy_eV)

    # Finding allowed phi range
    phi_half = np.arcsin((widthDetector / 2) / radius)
    return phi_half



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

imTensor31Avg51 = convolvedImage(im31,kernelSizeTuple=(3,3),paddingTuple=(1,1))
imTensor31Avg33 = convolvedImage(im31,kernelSizeTuple=(5,3),paddingTuple=(2,1))
imTensor31Avg53 = convolvedImage(im31,kernelSizeTuple=(7,3),paddingTuple=(3,1))

plt.figure(figsize=(10,5))
plt.subplot(2,2,1), plt.imshow(imTensor31Avg31, cmap='hot'), plt.title('1')
plt.subplot(2,2,2), plt.imshow(imTensor31Avg51, cmap='hot'), plt.title('2')
plt.subplot(2,2,3), plt.imshow(imTensor31Avg33, cmap='hot'), plt.title('3')
plt.subplot(2,2,4), plt.imshow(imTensor31Avg53, cmap='hot'), plt.title('4')

plt.show()