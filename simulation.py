from legacy_code.calibrateGeometry_v1 import *

"""
Spectrum produced by plasma
-> deflection probability on crystal?
-> solid angle contributions
-> Fano factor + absorbtion probability in silicon / CCD
"""

def generate_spectrum(energy_start_eV=1090,energy_end_eV=1610,step=0.5,
                      energy_peaks_eV=None, peak_width=5, peak_height=100,
                      plotIm=False):
    """
    The Purpose of this function is to create a spectrum of
    X-Ray energies between energy_start_eV and energy_endeV with peaks at
    specific energies (given in eV)

    energy_peaks_eV: list of energies at which there are known peaks
    peak_width,peak_height: parameters of the Gaussian peaks which are then added on top of the noise

    """
    if energy_peaks_eV is None:
        energy_peaks_eV = [E_Lalpha_eV, E_Lbeta_eV]
    # Generate energy range
    range_energy = np.arange(energy_start_eV, energy_end_eV, step)

    # There is an approximate peak around eV 1400:
    count_energy = peak_height/4 * np.exp(-((range_energy - 1400) ** 2) / (2 * 100 ** 2)) + peak_height/4

    # Add Gaussian peaks
    for peak in energy_peaks_eV:
        gaussian_peak = peak_height * np.exp(-((range_energy - peak) ** 2) / (2 * peak_width ** 2))
        count_energy += gaussian_peak


    # Create DataFrame for plotting
    df_spectrum = pd.DataFrame({'Energy': range_energy, 'Count': count_energy})

    if plotIm:
        # Plot
        sns.lineplot(data=df_spectrum, x="Energy", y="Count")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Intensity")
        plt.title("Smooth X-ray Spectrum")
        plt.show()

    return df_spectrum


# generate_spectrum(plotIm=True)


def simulate_solid_angle(count_perBin,bin_width,
                         geometry_engine,
                         lowboundE,highboundE,
                         phiMin=np.pi-0.38,  phiMax=np.pi+0.38,
                         plot=False):

    # Give all energies an equal count e.g. 100,000 and the assuming isotropic, randomly pick a direction and then check
    # if theta gives a bragg reflection then see if it hits the pixel and maybe create the spectrum with that

    energy_array = np.arange(start=lowboundE,stop=highboundE,step=bin_width)

    rotMatrix_crys = rotMatrixUsingEuler(geometry_engine.crystal_pitch, geometry_engine.crystal_roll)

    def approximate_thetaBounds(E_bound):
        theta_bound_E = bragg_E_to_theta(E_bound) + np.pi/2  # to make it polar and the np.pi/2
        v_ray_prime_sphr = np.array([1,theta_bound_E,np.pi])
        v_ray_prime_cart = spherical_to_cartesian(v_ray_prime_sphr)

        v_ray_cart = np.dot(rotMatrix_crys, v_ray_prime_cart)
        v_ray_cart_normalised = v_ray_cart / np.linalg.norm(v_ray_cart)
        v_ray_spherical = cartesian_to_spherical(v_ray_cart_normalised)

        return v_ray_spherical[1]

    theta_hb = approximate_thetaBounds(lowboundE)
    theta_lb = approximate_thetaBounds(highboundE)

    print("theta_hb",theta_hb)
    print("theta_lb",theta_lb)

    countOnCCD_list = []

    for energy in energy_array:
        print("-"*30)
        theta_bragg_ub = bragg_E_to_theta(energy-bin_width/2)
        theta_bragg = bragg_E_to_theta(energy)
        theta_bragg_lb = bragg_E_to_theta(energy+bin_width/2)
        print(f"Energy = {energy} eV, Bragg Angle = {theta_bragg} radians")
        count_onCCD = 0

        # consider the range of energy in the bin to expand angle acceptance range

        for count in range(count_perBin):
            rdm_theta, rdm_phi = random_direction(theta_max=theta_hb,theta_min=theta_lb,
                                                  phi_min=phiMin,phi_max=phiMax)

            n_rdm_sphr = np.array([1,rdm_theta,rdm_phi])
            n_rdm_cart = spherical_to_cartesian(n_rdm_sphr)

            # Finding whether rdm_theta obeys the bragg condition
            n_crys = nVectorFromEuler(geometry_engine.crystal_pitch,geometry_engine.crystal_roll)

            cos_theta_crys = abs(np.dot(n_rdm_cart, n_crys))
            angle_between = np.arccos(cos_theta_crys)

            # The angle with respect to the plane of the crystal is pi/2 - this

            angle_beamToPlane = np.pi/2 - angle_between

            # print("angle_between",angle_beamToPlane)

            if theta_bragg_lb < angle_beamToPlane < theta_bragg_ub:
                # print("Random ray obeys bragg condition")
                x_plane, y_plane = ray_in_planeCamera(v_ray_cart=n_rdm_cart, n_camera_cart=geometry_engine.nCam,
                                                      r_camera_cart=geometry_engine.r_cam_cart)
                if ((abs(x_plane) < (geometry_engine.xWidth - geometry_engine.pixelWidth) / 2).all() and
                        (abs(y_plane) < (geometry_engine.yWidth - geometry_engine.pixelWidth) / 2).all()):
                    count_onCCD += 1

        print(count_onCCD)
        countOnCCD_list.append(count_onCCD)


    def saveToExcel(filePath = r"C:\Users\marcg\OneDrive\Documents\Oxford Physics\Year 3\B8\b8_xspeds\stored_variables\solidAngle.xlsx"):

        df = pd.DataFrame()
        df["Energy"] = energy_array
        df["Count"] = countOnCCD_list
        df["Normalised_Count"] = df["Count"] / df["Count"].max()

        df.to_excel(filePath, index=False)

    saveToExcel()


    if plot:
        xarray = energy_array
        yarray = np.array(countOnCCD_list)

        plt.plot(xarray,yarray,label="Count")
        plt.xlabel("Energy (eV)")
        plt.ylabel("Count")
        plt.title("Proportion of Istropically emitted counts on CCD")
        plt.show()



class Simulation:
    def __init__(self, df_spectrum, relativeCount, experimental_geometry):
        self.df = df_spectrum
        self.relativeCount = relativeCount
        self.experimental_geometry = experimental_geometry

    def simulateHits(self):
        
        energyVals = self.df["Energy"]
        countVals = self.df["Count"]
        maxCount = np.max(countVals)
        # normalising for max count and rounding to the nearest integer
        countVals = (countVals / maxCount * self.relativeCount).astype(int)

        """Now to iterate for each energy, given the geometrical setup find the polar
        theta coordinate of the rays and find the resulting range of phi values that hit the detector
        and account for solid angle normalisation."""

        n_crystal, n_camera, r_camera = self.experimental_geometry[0], self.experimental_geometry[1], self.experimental_geometry[2]

        rotMatrix_cry = inverseRotation_matrix(n_crystal)

        for energyVal,countVal in zip(energyVals, countVals):

            theta = bragg_E_to_theta(energyVal)




    class Bragg:

        @staticmethod
        def randomPixel(energy_eV,n_crystal, n_camera, r_camera):
            """
            For a given energy there is a range of x y pixels that are allowed
            """
            rotMatrix_cry = inverseRotation_matrix(n_crystal)
            theta = bragg_E_to_theta(energy_eV)


# Simulation(generate_spectrum(plotIm=True), 100, "Bragg").simulateHits()


if __name__ == '__main__':

    def check_solid_angle():

        geo_engine = geo_engine_withSavedParams()

        # simulate_solid_angle(count_perBin=5000000,
        #                      bin_width=1,
        #                      geometry_engine=geo_engine,
        #                      lowboundE=1080,
        #                      highboundE=1700,
        #                      plot=True)

        simulate_solid_angle(count_perBin=5000000,
                             bin_width=1,
                             geometry_engine=geo_engine,
                             lowboundE=1080,
                             highboundE=1082,
                             plot=True)

    # check_solid_angle()
