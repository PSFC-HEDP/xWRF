# analyze CPS
import argparse
import os
from math import inf, nan, pi
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy import integrate, interpolate, optimize

from cmap import CMAP

Point = np.dtype((float, (2,)))

SCAN_DIRECTORY = "scans"

Ξ_MIN = 0.20
Ξ_MAX = 0.95
Υ_MIN = 0.06
Υ_MAX = 0.94


def main() -> None:
	parser = argparse.ArgumentParser(
		prog="analyze_xwrf",
		description = "analyze the xWRF scan files in a given directory.")
	parser.add_argument("filename", type=str,
	                    help="Any files in the scans directory with this substring in their filename will be analyzed")
	parser.add_argument("wedge_id", type=str,
	                    help="the ID number of the wedge range filter")
	parser.add_argument("--filter", type=float,
	                    help="the thickness of flat aluminum filtering in front of the image plate in μm")
	parser.add_argument("--cr39", action="store_true",
	                    help="whether there was a piece of CR-39 in front of the image plate")
	parser.add_argument("--nose", action="store_true",
	                    help="whether a 300 μm Al nosetip was in front of the WRF")
	args = parser.parse_args()

	analyze_scanfile(args.filename, args.wedge_id, args.filter, args.cr39, args.nose)


def analyze_scanfile(filename: str, wedge_id: str, filter: Optional[float], nose: bool, cr39: bool) -> tuple[float, float]:
	""" infer electron temperature from a single xWRF scanfile
	    :param filename: a substring that specifies which file you want.  it doesn’t have to be the
	                     complete filename; just the shot number will usually do.
	    :param wedge_id: the ID of the WRF that was used (for example "G069")
	    :param filter: if an aluminum filter was placed between the wedge and the image plate, this
	                   is the thickness of that filter in μm.  otherwise, it should be 0 or None.
	    :param nose: whether a 300 μm aluminum nose was present in front of the wedge (in
	                 addition to any flat filter specified by the `filter` argument)
	    :param cr39: whether a 1500 μm piece of CR-39 was present between the wedge and the image plate
	    :return: the inferred temperature and the error bar on that temperature
	"""
	# find the scanfile
	filepath = find_file(SCAN_DIRECTORY, filename, ".h5")
	print(f"Analyzing {filepath}...")

	# load the imaging data
	with h5py.File(filepath, 'r') as f:
		image = f["PSL_per_px"][:, :].T
		x_bin = f["PSL_per_px"].attrs["pixelSizeX"]  # (μm)
		y_bin = f["PSL_per_px"].attrs["pixelSizeY"]  # (μm)
		fade_time = f["PSL_per_px"].attrs["scanDelaySeconds"]
	if x_bin != y_bin:
		raise ValueError("why would these ever not be equal")
	pixel_width = x_bin*1e-4

	# rotate, average in the nondispersive direction, and map to thickness
	thicknesses, psl = collapse_image(pixel_width, image, wedge_id)

	# calculate sensitivity curves
	reference_energies = np.geomspace(1, 1e3, 61)
	Al_attenuation = load_attenuation_curve(reference_energies, "Al")
	log_sensitivity = log_xray_sensitivity(reference_energies, fade_time)
	if filter is not None:
		log_sensitivity -= filter*load_attenuation_curve(reference_energies, "Al")
	if nose:
		log_sensitivity -= 300*load_attenuation_curve(reference_energies, "Al")
	if cr39:
		log_sensitivity -= 1500*load_attenuation_curve(reference_energies, "cr39")

	# perform the temperature fit
	Te, dTe, εL, dεL = fit_temperature(thicknesses, np.zeros(thicknesses.shape),
	                                   psl, np.ones(psl.shape),
	                                   reference_energies, Al_attenuation, log_sensitivity)

	print(f"Inferred T_e = {Te:.2f} ± {dTe:.2f}")
	plt.show()

	return Te, dTe


def collapse_image(pixel_width: float, image: NDArray[float], filter_id: str) -> tuple[NDArray[float], NDArray[float]]:
	""" find the fiducials in an xWRF scan, rotate the image, and map PSL to filter thickness """
	x_centers = np.arange(0.5, image.shape[0])*pixel_width  # (cm)
	y_centers = np.arange(0.5, image.shape[1])*pixel_width  # (cm)
	image_interpolator = interpolate.RegularGridInterpolator((x_centers, y_centers), image)

	# declare the wedge coordinates
	ξ_data = np.linspace(0.0, 1.0, round(2./pixel_width))
	υ_data = np.linspace(0.0, 1.0, round(2./pixel_width))

	# load the WRF thickness calibration
	calibration_table = pd.read_csv("tables/wrf_calibrations.csv", sep=",", index_col="id")
	calibration = calibration_table.loc[filter_id.lower()]
	thickness = wedge_thickness_function(ξ_data, calibration.xp, calibration.dt0, calibration.dtdx)

	# locate the fiducials and rebase to them
	ξ_lever, origin, υ_lever = find_fiducials(pixel_width, image)[:, np.newaxis, np.newaxis, :]
	xy_data = origin + \
	          ξ_data[:, np.newaxis, np.newaxis]*(ξ_lever - origin) + \
	          υ_data[np.newaxis, :, np.newaxis]*(υ_lever - origin)
	rebased_image = image_interpolator(tuple(np.transpose(xy_data, (2, 0, 1))))

	# guess if you have to rotate it
	υ_in_υ_bounds = (υ_data >= Υ_MIN) & (υ_data <= Υ_MAX)
	clean_image = rebased_image[(ξ_data >= Υ_MIN) & (ξ_data <= Υ_MAX)][:, υ_in_υ_bounds]
	if np.ptp(np.mean(clean_image, axis=0)) > np.ptp(np.mean(clean_image, axis=1)):
		rebased_image = rebased_image.T
	# guess if you have to flip it
	if np.mean(rebased_image[0, υ_in_υ_bounds]) < np.mean(rebased_image[-1, υ_in_υ_bounds]):
		rebased_image = rebased_image[::-1, :]

	# crop out the fiducial region
	cropped_image = rebased_image[:, υ_in_υ_bounds]

	# show the rebased image
	plt.figure()
	plt.imshow(rebased_image.T, extent=(0, 1, 0, 1), origin="lower",
	           cmap=CMAP["psl"], vmax=min(np.max(rebased_image), 1.1*np.max(cropped_image)))
	plt.fill([Ξ_MIN, Ξ_MIN, Ξ_MAX, Ξ_MAX], [Υ_MIN, Υ_MAX, Υ_MAX, Υ_MIN],
	         color="none", edgecolor="w", linewidth=1)
	plt.title("Data region (wedge is thinnest on the left)")
	plt.xlabel("Dispersive direction")
	plt.ylabel("Nondispersive direction")
	plt.locator_params(steps=[1, 2, 5, 10])

	# integrate the image along the nondispersive direction
	measurement = np.mean(cropped_image, axis=1)

	ξ_in_ξ_bounds = (ξ_data >= Ξ_MIN) & (ξ_data <= Ξ_MAX)
	return thickness[ξ_in_ξ_bounds], measurement[ξ_in_ξ_bounds]


def find_fiducials(pixel_width: float, image: NDArray[float]) -> NDArray[Point]:
	""" find the centers of the three big fiducial circles """
	expected_radius = .079
	sub_image_size = int(4*expected_radius/pixel_width)
	fiducials = []
	for i in range(0, image.shape[0] - sub_image_size, sub_image_size//2):
		for j in range(0, image.shape[1] - sub_image_size, sub_image_size//2):
			sub_image = image[i:i + sub_image_size, j:j + sub_image_size]
			bright_pixels = sub_image >= np.quantile(sub_image, 1 - pi/16)
			# if it sees a bright spot in its entirety...
			if np.any(bright_pixels) and not np.all(bright_pixels):
				xs = pixel_width*(i + np.arange(sub_image_size))
				ys = pixel_width*(j + np.arange(sub_image_size))
				x0 = np.average(xs, weights=np.sum(bright_pixels, axis=1))
				if x0 < xs[0] + expected_radius or x0 > xs[-1] - expected_radius:
					continue
				y0 = np.average(ys, weights=np.sum(bright_pixels, axis=0))
				if y0 < ys[0] + expected_radius or y0 > ys[-1] - expected_radius:
					continue
				r = np.hypot(*np.meshgrid(xs - x0, ys - y0, indexing="ij"))
				# if the spot is not too much bigger than we expect
				if not np.any(bright_pixels[r >= 1.33*expected_radius]):
					# if the spot is not much smaller than we expect
					if np.all(bright_pixels[r <= .75*expected_radius]):
						# save the centroid we found
						fiducials.append((x0, y0))

	fiducials = np.array(fiducials, dtype=Point)

	# show the user where it thinks the fiducials are
	plt.imshow(image.T,
	           extent=(0, image.shape[1]*pixel_width, 0, image.shape[0]*pixel_width),
	           origin="lower", cmap=CMAP["psl"])
	plt.plot(fiducials[:, 0], fiducials[:, 1], "wx", markersize=13)
	plt.title("Raw data (fiducials are marked with white exes)")
	plt.ylabel("Axes measured in cm")
	plt.locator_params(steps=[1, 2, 5, 10])

	if len(fiducials) == 0:
		raise RuntimeError("no fiducials found")
	elif len(fiducials) < 3:
		raise RuntimeError("not enough fiducials found")
	elif len(fiducials) > 4:
		print("too many fiducials found (but it’s chill; I’ll just ignore some of them)")

	# now sort the fiducials in somewhat random order
	initial = fiducials[np.argmax(np.max(
		np.hypot(fiducials[:, np.newaxis, 0] - fiducials[np.newaxis, :, 0],
		         fiducials[:, np.newaxis, 1] - fiducials[np.newaxis, :, 1]), axis=1))]
	fiducials = fiducials[np.argsort(
		np.hypot(fiducials[:, 0] - initial[0],
		         fiducials[:, 1] - initial[1]))]
	fiducials = fiducials[[0, 1, -1]]  # downselect somewhat randomly to just three
	return fiducials


def fit_temperature(thicknesses: NDArray[float], thickness_errors: NDArray[float],
                    measurements: NDArray[float], measurement_errors: NDArray[float],
                    energies: NDArray[float], attenuations: NDArray[float],
                    log_sensitivities: NDArray[float]
                    ) -> tuple[float, float, float, float]:
	""" take a set of measured x-ray intensity values from a single chord thru the implosion and
		use their average and their ratios to infer the emission-averaged electron temperature,
		and the total line-integrated photic emission along that chord.
		:param thicknesses: the set of aluminum thicknesses represented in the filter (μm)
		:param thickness_errors: the error bars on the thickness values (μm)
		:param measurements: the detected radiation corresponded to each thickness (PSL/μm^2)
		:param measurement_errors: the error bars on the psl for each pixel (PSL/μm^2)
		:param energies: the photon energies at which the sensitivities have been calculated (keV)
		:param attenuations: the attenuation coefficient of aluminium at each reference energy (μm^-1)
		:param log_sensitivities: the log of the dimensionless sensitivity of the detector at each reference energy
		:return: the electron temperature (keV) and the total emission (PSL/μm^2/sr)
	"""
	def compute_values(βe):
		integrand = np.exp(
			-energies*βe - thicknesses[:, np.newaxis]*attenuations + log_sensitivities)
		unscaled_values = integrate.trapezoid(x=energies, y=integrand, axis=1)
		numerator = np.sum(unscaled_values*measurements/measurement_errors**2)
		denominator = np.sum(unscaled_values**2/measurement_errors**2)
		return integrand, numerator, denominator, unscaled_values

	def compute_residuals(βe):
		_, numerator, denominator, unscaled_values = compute_values(βe)
		values = numerator/denominator*unscaled_values
		return (values - measurements)/measurement_errors

	def compute_derivatives(βe):
		integrand, numerator, denominator, unscaled_values = compute_values(βe)
		unscaled_derivatives = integrate.trapezoid(x=energies, y=-energies*integrand, axis=1)
		numerator_derivative = np.sum(unscaled_derivatives*measurements/measurement_errors**2)
		denominator_derivative = 2*np.sum(unscaled_derivatives*unscaled_values/measurement_errors**2)
		return (numerator/denominator*unscaled_derivatives +
		        numerator_derivative/denominator*unscaled_values -
		        numerator*denominator_derivative/denominator**2*unscaled_values
		        )/measurement_errors

	if np.all(measurements == 0):
		return 0, inf, 0, 0
	else:
		# start with a scan
		best_Te, best_χ2 = None, inf
		for Te in np.geomspace(5e-2, 5e-0, 11):
			χ2 = np.sum(compute_residuals(1/Te)**2)
			if χ2 < best_χ2:
				best_Te = Te
				best_χ2 = χ2
		# then do a newton’s method
		result = optimize.least_squares(fun=lambda x: compute_residuals(x[0]),
		                                jac=lambda x: np.expand_dims(compute_derivatives(x[0]), 1),
		                                x0=[1/best_Te],
		                                bounds=(0, inf))
		if result.success:
			best_βe = result.x[0]
			error_βe = np.linalg.inv(1/2*result.jac.T@result.jac)[0, 0]
			best_Te = 1/best_βe
			error_Te = error_βe/best_βe**2

			_, numerator, denominator, unscaled_psl = compute_values(best_βe)
			best_εL = numerator/denominator*best_Te
			error_εL = 0

			fig, axs = plt.subplots(2, 1, sharex="all")
			axs[0].plot(thicknesses, measurements, "C0-", label="Data")
			axs[0].plot(thicknesses, numerator/denominator*unscaled_psl, "C1--", label="Fit")
			axs[0].grid("on")
			axs[0].set_title(f"Best fit (Te = {best_Te:.3f} ± {error_Te:.3f} keV)")
			axs[0].set_ylabel("PSL")
			axs[0].locator_params(steps=[1, 2, 5, 10], nbins=10)
			axs[1].errorbar(x=thicknesses, y=measurements - numerator/denominator*unscaled_psl,
			                yerr=measurement_errors, fmt="C2.", linewidth=.8)
			axs[1].grid("on")
			axs[1].set_ylabel("Residual")
			axs[1].set_xlabel("Aluminum thickness (μm)")
			axs[1].locator_params(steps=[1, 2, 5, 10], nbins=10)

			return best_Te, error_Te, best_εL, error_εL
		else:
			return nan, nan, nan, nan


def log_xray_sensitivity(energy: NDArray[float], fade_time: float,
                         thickness=112., psl_attenuation=1/45., material="phosphor",
                         A1=.436, A2=.403, τ1=1.134e3, τ2=9.85e4) -> NDArray[float]:
	""" calculate the log of the fraction of x-ray energy at some frequency that is measured by an
	    image plate of the given characteristics, given some filtering in front of it
	    :param energy: the photon energies (keV)
	    :param fade_time: the delay between the experiment and the image plate scan (s)
	    :param thickness: the thickness of the image plate (μm)
	    :param psl_attenuation: the attenuation constant of the image plate's characteristic photostimulated luminescence
	    :param material: the name of the image plate material (probably just the elemental symbol)
	    :param A1: an initial condition for the fade term
	    :param A2: an initial condition for the fade term
	    :param τ1: a decay time for the fade term (s)
	    :param τ2: a decay time for the fade term (s)
	    :return: the fraction of photic energy that reaches the scanner
	"""
	attenuation = load_attenuation_curve(energy, material)
	self_transparency = 1/(1 + psl_attenuation/attenuation)
	log_sensitivity = np.log(
		self_transparency * (1 - np.exp(-attenuation*thickness/self_transparency)) *
		psl_fade(fade_time, A1, A2, τ1, τ2))
	return log_sensitivity


def load_attenuation_curve(energy: NDArray[float], material: str) -> NDArray[float]:
	""" load the attenuation curve for x-rays in a material
	    :param energy: the photon energies (keV)
	    :param material: the name of the material (probably just the elemental symbol)
	    :return: the attenuation constant at the specified energy (μm^-1)
	"""
	table = np.loadtxt(f"tables/attenuation_{material}.csv", delimiter=",")
	return np.interp(energy, table[:, 0], table[:, 1])


def psl_fade(time: float, A1: float, A2: float, τ1: float, τ2: float):
	""" the portion of PSL that remains after some seconds have passed
	    :param time: the time between exposure and scan (s)
	    :param A1: the portion of energy initially in the fast-decaying eigenmode (nominally .436±.016)
	    :param A2: the portion of energy initially in the slow-decaying eigenmode (nominally .403±.013)
	    :param τ1: the decay time of the faster eigenmode (nominally 1134±90 s)
	    :param τ2: the decay time of the slower eigenmode (nominally 98.5±9.1 ks)
	"""
	return A1*np.exp(-time/τ1) + A2*np.exp(-time/τ2) + (1 - A1 - A2)


def wedge_thickness_function(x: NDArray[float], x0: float, delta_t0: float, delta_dtdx: float) -> NDArray[float]:
	""" calculate the thickness of a WRF at a set of specified x values, given calibration info
	    :param x: the desired normalized x values, from 0 (at the left fiducial) to 1 (at the right fiducial)
	    :param x0: the x, measured from the midpoint, at which the calibration measurements were made
	    :param delta_t0: the opposite of the difference between the actual thickness at xp and the nominal one
	    :param delta_dtdx: the difference between the actual slope and the nominal slope (why didn’t Fredrick just record the total slope?)
	    :return: the average thickness at that x value (μm)
	"""
	half_width = 0.474*2.54
	X = (2*x - 1)*half_width  # convert to the coordinates Fredrick uses to define his calibration
	t_left = 0.0155*2.54e4
	t_rite = 0.0710*2.54e4
	t0 = (t_left + t_rite)/2
	dtdx = (t_rite - t_left)/(2*half_width)
	return t0 - delta_t0 + delta_dtdx*x0 + X*(dtdx - delta_dtdx)


def find_file(directory: str, substring: str, extension: str) -> str:
	""" search a directory for a filename containing the given substring and ending in the given
	    extension, and return that filepath
	"""
	for filename in os.listdir(SCAN_DIRECTORY):
		if substring in filename and os.path.splitext(filename)[-1] == extension:
			return os.path.join(directory, filename)
	raise FileNotFoundError(f"did not find any file in `{directory}` matching `*{substring}*{extension}`")


if __name__ == "__main__":
	main()
