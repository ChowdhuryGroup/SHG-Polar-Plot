import matplotlib.pyplot as plt
import numpy as np

filename = open("289khoriz.txt", "r")
lines = filename.readlines()

date_start = len(lines[0].split(":")[0])
date = lines[0][date_start + 1 :]

spectra_start = 2
degrees = ""
for line in lines[spectra_start:]:
    degrees += line[3:]
    spectra_start += 1
    if line.strip()[-1] == "]":
        break
degrees = np.fromstring(degrees[:-2], sep=" ")

data = []
for line in lines[spectra_start:]:
    data.append(np.fromstring(line, sep=","))
data = np.array(data)

if data.shape[1] - 2 != degrees.shape[0]:
    raise Exception("The number of spectra and number of degrees doesn't match")

wavelengths = data[:, 0]
background = data[:, 1]
spectra = data[:, 2:]

#  480nm-530nm
def calc_counts_area(
    wavelengths: np.ndarray,
    spectrum: np.ndarray,
    start_wavelength: float,
    end_wavelength: float,
) -> float:

    filter = np.logical_and(
        wavelengths >= start_wavelength, wavelengths <= end_wavelength
    )

    integral = 0.0
    for i, counts in enumerate(spectrum):
        if filter[i]:
            integral += (wavelengths[i + 1] - wavelengths[i - 1]) * 0.5 * counts

    return integral


integrals = []
for i in range(spectra.shape[1]):
    # spectra[:, i] -= background
    integrals.append(calc_counts_area(wavelengths, spectra[:, i], 480.0, 530.0))

plt.axes(projection="polar")
plt.polar(degrees * (np.pi / 180.0), integrals)
plt.show()

np.savetxt("289K Horizontal.tsv", np.stack((degrees, integrals), axis=1))
