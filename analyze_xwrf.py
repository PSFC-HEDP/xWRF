# analyze CPS
import argparse
import os
from math import inf, nan, pi
import sys
from typing import Optional

import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy import integrate, interpolate, optimize

from cmap import CMAP
from image_plate import load_attenuation_curve, fade, log_xray_sensitivity

plt.rcParams.update({'font.family': 'sans', 'font.size': 14})

Point = np.dtype((float, (2,)))

SCAN_DIRECTORY = "scans"

Ξ_MIN = 0.20
Ξ_MAX = 0.95
Υ_MIN = 0.01
Υ_MAX = 0.94
NUM_SAMPLES = 2
THICKNESS_ERROR = 3.e-2  # (dimensionless)
PSL_ATTENUATION = 1/45.  # μm^-1
PSL_ATTENUATION_ERROR = 5.e-3  # μm^-1
WORK_FUNCTION = 2200  # keV
WORK_FUNCTION_ERROR = 100  # keV


def main() -> None:
	parser = argparse.ArgumentParser(
		prog="analyze_xwrf",
		description = "analyze the xWRF scan files in a given directory.")
	parser.add_argument("filename", type=str,
	                    help="Any files in the scans directory with this substring in their filename will be analyzed")
	parser.add_argument("wedge_id", type=str,
	                    help="the ID number of the wedge range filter")
	parser.add_argument("--distance", type=float, default=1,
	                    help="the distance from the implosion to the detector, if you care about absolute fluence (cm)")
	parser.add_argument("--filter", type=float,
	                    help="the thickness of flat aluminum filtering in front of the image plate in μm")
	parser.add_argument("--cr39", action="store_true",
	                    help="whether there was a piece of CR-39 in front of the image plate")
	parser.add_argument("--nose", action="store_true",
	                    help="whether a 300 μm Al nosetip was in front of the WRF")
	args = parser.parse_args()

	try:
		analyze_scanfile(args.filename, args.wedge_id, args.distance, args.filter, args.cr39, args.nose)
	except Exception as e:
		print("Error!", e)
		plt.show()
		sys.exit(1)


def analyze_scanfile(filename: str, wedge_id: str, distance: float, filter: Optional[float], nose: bool, cr39: bool) -> tuple[float, float]:
	""" infer electron temperature from a single xWRF scanfile
	    :param filename: a substring that specifies which file you want.  it doesn’t have to be the
	                     complete filename; just the shot number will usually do.
	    :param wedge_id: the ID of the WRF that was used (for example "G069")
	    :param distance: the distance between the implosion and the image plate (cm)
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
		x_bin = f["PSL_per_px"].attrs["pixelSizeX"]  # (μm)
		y_bin = f["PSL_per_px"].attrs["pixelSizeY"]  # (μm)
		fade_time = f["PSL_per_px"].attrs["scanDelaySeconds"]  # (s)
		image = f["PSL_per_px"][:, :].T/x_bin/y_bin*(distance*1e4)**2/fade(fade_time)  # (?/sr)
	if x_bin != y_bin:
		raise ValueError("why would these ever not be equal")
	pixel_width = x_bin*1e-4

	# rotate, average in the nondispersive direction, and map to thickness
	thicknesses, thickness_error, psl = collapse_image(pixel_width, image, wedge_id)

	# put together whatever flat filter is out front
	reference_energies = np.geomspace(1, 1e3, 61)
	Al_attenuation = load_attenuation_curve(reference_energies, "Al")
	log_transmission = np.zeros(reference_energies.size)
	if filter is not None:
		log_transmission -= filter*load_attenuation_curve(reference_energies, "Al")
	if nose:
		log_transmission -= 300*load_attenuation_curve(reference_energies, "Al")
	if cr39:
		log_transmission -= 1500*load_attenuation_curve(reference_energies, "cr39")

	# perform the temperature fit
	Te, dTe, εL, dεL = fit_temperature_with_error_bars(
		thicknesses, thickness_error, psl, reference_energies, log_transmission, Al_attenuation)

	print(f"Inferred electron temperature = {Te:.2f} ± {dTe:.2f} keV\n"
	      f"               total emission = {εL:.3g} ± {dεL:.3g} (units unclear)")
	plt.show()

	return Te, dTe


def collapse_image(pixel_width: float, image: NDArray[float], wedge_id: str
                   ) -> tuple[NDArray[float], float, NDArray[float]]:
	""" find the fiducials in an xWRF scan, rotate the image, and map PSL to filter thickness
	    :return the array of thicknesses (μm), the error bar on those thicknesses (μm), and the array of PSL values (PSL/sr)
	"""
	x_centers = np.arange(0.5, image.shape[0])*pixel_width  # (cm)
	y_centers = np.arange(0.5, image.shape[1])*pixel_width  # (cm)
	image_interpolator = interpolate.RegularGridInterpolator((x_centers, y_centers), image)

	# declare the wedge coordinates
	ξ_data = np.linspace(0.0, 1.0, round(2./pixel_width))
	υ_data = np.linspace(0.0, 1.0, round(2./pixel_width))

	# load the WRF thickness calibration
	calibration_table = pd.read_csv("tables/wrf_calibrations.csv", sep=",", index_col="id")
	calibration = calibration_table.loc[wedge_id.lower()]
	thickness = wedge_thickness_function(ξ_data, calibration.xp, calibration.dt0, calibration.dtdx)
	thickness_error = calibration["systematic error"]/.02  # fudge this by assuming dE/dx=20keV/μm (MeV/(MeV/μm) = μm)

	# locate the fiducials and rebase to them
	υ_lever, origin, ξ_lever = find_fiducials(pixel_width, image)[:, np.newaxis, np.newaxis, :]
	xy_data = origin + \
	          ξ_data[:, np.newaxis, np.newaxis]*(ξ_lever - origin) + \
	          υ_data[np.newaxis, :, np.newaxis]*(υ_lever - origin)
	rebased_image = image_interpolator(tuple(np.transpose(xy_data, (2, 0, 1))))

	# guess if you have to rotate it
	clean_image = rebased_image[(ξ_data >= 1 - Υ_MAX) & (ξ_data <= Υ_MAX)][:, (υ_data >= 1 - Υ_MAX) & (υ_data <= Υ_MAX)]
	if np.ptp(np.mean(clean_image, axis=0)) > np.ptp(np.mean(clean_image, axis=1)):
		rebased_image = rebased_image.T[:, ::-1]
		clean_image = clean_image.T[:, ::-1]
	# guess if you have to flip it
	if np.mean(clean_image[0, :]) < np.mean(clean_image[-1, :]):
		rebased_image = rebased_image[::-1, ::-1]

	# crop out the fiducial region and the unusable left area
	full_image = rebased_image[:, (υ_data >= 1 - Υ_MAX) & (υ_data <= Υ_MAX)]
	cropped_image = rebased_image[(ξ_data >= Ξ_MIN) & (ξ_data <= Ξ_MAX)][:, (υ_data >= Υ_MIN) & (υ_data <= Υ_MAX)]
	thickness = thickness[(ξ_data >= Ξ_MIN) & (ξ_data <= Ξ_MAX)]

	# show the rebased image
	plt.figure(figsize=(5, 5))
	plt.imshow(rebased_image.T, extent=(0, 1, 0, 1), origin="lower",
	           cmap=CMAP["psl"], vmax=min(np.max(rebased_image), 1.1*np.max(full_image)))
	plt.fill([Ξ_MIN, Ξ_MIN, Ξ_MAX, Ξ_MAX], [Υ_MIN, Υ_MAX, Υ_MAX, Υ_MIN],
	         color="none", edgecolor="w", linewidth=1)
	plt.title("Data region (wedge is thinnest on the left)")
	plt.xlabel("Dispersive direction")
	plt.ylabel("Nondispersive direction")
	plt.locator_params(steps=[1, 2, 5, 10])

	# integrate the image along the nondispersive direction
	measurement = np.mean(cropped_image, axis=1)

	return thickness, thickness_error, measurement


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
	plt.figure(figsize=(5, 5))
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
	if not triangle_is_counterclockwise(fiducials[0, :], fiducials[1, :], fiducials[2, :]):
		fiducials = fiducials[::-1, :]  # order them so they make a widdershins triangle
	return fiducials


def fit_temperature_with_error_bars(thicknesses: NDArray[float], thickness_error: float,
                                    measurements: NDArray[float],
                                    energies: NDArray[float],
                                    log_transmissions: NDArray[float], attenuations: NDArray[float],
                                    ) -> tuple[float, float, float, float]:
	""" take a set of measured x-ray intensity values from a single chord thru the implosion and use their average and
	    their ratios to infer the emission-averaged electron temperature, and the total line-integrated photic emission
	    along that chord. assume you know the PSL values and thicknesses exactly.
	    :param thicknesses: the set of aluminum thicknesses represented in the wedge (μm)
	    :param thickness_error: the maximum plausible difference between the nominal and actual wedge thickness at
	           any given point (μm)
	    :param measurements: the detected radiation corresponded to each thickness (PSL/sr)
	    :param energies: the photon energies at which the sensitivities should be calculated (keV)
	    :param log_transmissions: the log of the transmission probability at each reference energy through whatever
	                              flat filtering exists in front of the IP
		:param attenuations: the attenuation coefficient of aluminium at each reference energy (μm^-1)
	    :return: the electron temperature (keV), the error bar on electron temperature (keV),
	             the total emission (PSL/sr), and the error bar on that (PSL/sr)
	"""
	# do, not a MC per se, but a sweep of some unknowns to get error bars
	sample_reconstructions = []
	temperatures, emissions = [], []
	for x in np.linspace(-1, 1, NUM_SAMPLES, endpoint=False):
		for x_left, x_right in [(-1, x), (x, 1), (1, -x), (-x, -1)]:
			perturbations = thickness_error*np.linspace(x_left, x_right, thicknesses.size)
			for transmission_factor in [1 - THICKNESS_ERROR, 1 + THICKNESS_ERROR]:
				for psl_attenuation in [PSL_ATTENUATION - PSL_ATTENUATION_ERROR, PSL_ATTENUATION + PSL_ATTENUATION_ERROR]:
					for work_function in [WORK_FUNCTION - WORK_FUNCTION_ERROR, WORK_FUNCTION + WORK_FUNCTION_ERROR]:
						log_sensitivities = log_transmissions*transmission_factor + log_xray_sensitivity(
							energies, work_function=work_function, psl_attenuation=psl_attenuation)

						temperature, emission, reconstruction = fit_temperature_exactly(
							thicknesses + perturbations, measurements,
							energies, attenuations, log_sensitivities)
						sample_reconstructions.append(reconstruction)
						temperatures.append(temperature)
						emissions.append(emission)

	# define error bars as absolute ranges
	temperature = (np.min(temperatures) + np.max(temperatures))/2
	dtemperature = (np.max(temperatures) - np.min(temperatures))/2
	emission = (np.min(emissions) + np.max(emissions))/2
	demission = (np.max(emissions) - np.min(emissions))/2
	sample_residuals = [measurements - fit for fit in sample_reconstructions]

	# perform the nominal reconstruction to get the reconstructed curve out
	log_sensitivities = log_transmissions + log_xray_sensitivity(
		energies, work_function=WORK_FUNCTION, psl_attenuation=PSL_ATTENUATION)
	_, _, reconstruction = fit_temperature_exactly(
		thicknesses, measurements, energies, attenuations, log_sensitivities)

	# plot the fit quality
	fig, axs = plt.subplots(2, 1, sharex="all", figsize=(7, 5), gridspec_kw=dict(hspace=0))
	axs[0].plot(thicknesses, measurements, "C0-", label="Data", zorder=30)
	axs[0].plot(thicknesses, reconstruction, "C1--", label="Fit", zorder=20)
	axs[0].fill_between(thicknesses,
	                    np.min(sample_reconstructions, axis=0),
	                    np.max(sample_reconstructions, axis=0), color="C1", alpha=1/4, zorder=10)
	axs[0].grid("on")
	axs[0].set_title(f"Best fit (Tₑ = {temperature:.2f} ± {dtemperature:.2f} keV)")
	axs[0].set_ylabel("PSL")
	axs[0].locator_params(steps=[1, 2, 5, 10], nbins=10)

	axs[1].plot(thicknesses, (measurements - reconstruction)/reconstruction, "C2.", markersize=4, zorder=30)
	axs[1].fill_between(thicknesses,
	                    np.min(sample_residuals, axis=0)/reconstruction,
	                    np.max(sample_residuals, axis=0)/reconstruction,
	                    color="C2", alpha=1/4, zorder=20)
	axs[1].axhline(0, color="k", linewidth=1, zorder=10)
	axs[1].grid("on")
	axs[1].set_ylabel("Residual")
	axs[1].set_xlabel("Aluminum thickness (μm)")
	axs[1].set_xlim(thicknesses[0], thicknesses[-1])
	axs[1].locator_params(steps=[1, 2, 5, 10], nbins=10)
	fig.tight_layout()

	# plt.figure(figsize=(5, 5))
	# plt.scatter(temperatures, emissions, c="C2", s=5, zorder=10)
	# plt.xlabel("Inferred temperature (keV)")
	# plt.ylabel("Inferred emission ()")
	# plt.title("Error bar determination by sampling thicknesses")
	# plt.grid("on")
	# plt.tight_layout()

	return temperature, dtemperature, emission, demission


def fit_temperature_exactly(thicknesses: NDArray[float], measurements: NDArray[float],
                            energies: NDArray[float], attenuations: NDArray[float],
                            log_sensitivities: NDArray[float]
                            ) -> tuple[float, float, NDArray[float]]:
	""" take a set of measured x-ray intensity values from a single chord thru the implosion and use their average and
	    their ratios to infer the emission-averaged electron temperature, and the total line-integrated photic emission
	    along that chord. assume you know the PSL values and thicknesses exactly.
		:param thicknesses: the set of aluminum thicknesses represented in the wedge (μm)
		:param measurements: the detected radiation corresponded to each thickness (PSL/sr)
		:param energies: the photon energies at which the sensitivities have been calculated (keV)
		:param attenuations: the attenuation coefficient of aluminium at each reference energy (μm^-1)
		:param log_sensitivities: the log of the dimensionless sensitivity of the detector at each reference energy
		:return: the electron temperature (keV), the total emission (PSL/sr), and the fit measurements (PSL/sr)
	"""
	measurement_errors = 1e-3*np.sqrt(np.max(measurements)*measurements)

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
		return nan, 0, measurements
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
		if not result.success:
			raise RuntimeError(result.message)

		best_βe = result.x[0]
		best_Te = 1/best_βe

		_, numerator, denominator, unscaled_psl = compute_values(best_βe)
		best_εL = numerator/denominator*best_Te

		return best_Te, best_εL, numerator/denominator*unscaled_psl


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


def triangle_is_counterclockwise(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> bool:
	""" calculate the handedness of a triplet of points """
	return (b[0] - a[0])*(c[1] - b[1]) - (c[0] - b[0])*(b[1] - a[1]) >= 0


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
