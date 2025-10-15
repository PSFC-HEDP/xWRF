import matplotlib.pyplot as plt
from numpy import linspace
import pandas as pd

from analyze_xwrf import wedge_thickness_function


dimensions_table = pd.read_csv("tables/wrf_types.csv", sep=",", index_col="type")
calibration_table = pd.read_csv("tables/wrf_calibrations.csv", sep=",", index_col="id")

plt.figure()
plt.locator_params(steps=[1, 2, 5, 10])
for wedge_id in calibration_table.index:
	if wedge_id.startswith("bb"):
		color = "C0"
	elif wedge_id.startswith("nb"):
		color = "C1"
	elif wedge_id.startswith("in"):
		color = "C2"
	elif wedge_id.startswith("g"):
		color = "C3"
	else:
		color = "C4"
	calibration = calibration_table.loc[wedge_id]
	dimensions = dimensions_table.loc[calibration.type]
	X = linspace(-dimensions.width/2, dimensions.width/2)
	thickness = wedge_thickness_function(
		X, dimensions.width, dimensions.t_min, dimensions.t_max,
		calibration.x0, calibration.t0_deviation, calibration.dtdx_deviation)
	plt.plot(X, thickness, color=color)
plt.grid()
plt.ylim(0, None)
plt.xlabel("X (cm)")
plt.ylabel("Thickness (μm)")
plt.tight_layout()

plt.figure()
plt.locator_params(steps=[1, 2, 5, 10])
plt.scatter(calibration_table.t0_deviation, calibration_table.dtdx_deviation, zorder=2.1)
for wedge_id in calibration_table.index:
	calibration = calibration_table.loc[wedge_id]
	plt.text(calibration.t0_deviation, calibration.dtdx_deviation, wedge_id)
plt.grid()
plt.xlabel("Thickness deviation (μm)")
plt.ylabel("Slope deviation (μm/normalized)")
plt.tight_layout()

plt.show()
