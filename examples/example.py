from pathlib import Path

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from phringe.core.entities.photon_sources.exozodi import Exozodi
from phringe.core.entities.photon_sources.planet import Planet
from phringe.core.entities.photon_sources.star import Star
from phringe.core.entities.scene import Scene
from phringe.phringe import PHRINGE

phringe = PHRINGE()
#
# settings = Settings(
#     grid_size=20,
#     has_planet_orbital_motion=False,
#     has_stellar_leakage=False,
#     has_local_zodi_leakage=False,
#     has_exozodi_leakage=False,
#     has_amplitude_perturbations=False,
#     has_phase_perturbations=False,
#     has_polarization_perturbations=False
# )

planet = Planet(
    name='Earth',
    mass='1 Mearth',
    radius='1 Rearth',
    temperature='288 K',
    semi_major_axis='1 au',
    eccentricity='0',
    inclination='180 deg',
    raan='0 deg',
    argument_of_periapsis='0 deg',
    true_anomaly='0 deg',
    grid_position=(0, 19)
)

star = Star(
    name='Sun',
    distance='10 pc',
    mass='1 Msun',
    radius='1 Rsun',
    temperature='5700 K',
    luminosity='1 Lsun',
    right_ascension='0 h',
    declination='-75 deg'
)

exozodi = Exozodi(
    level=3,
    inclination='0 deg'
)

scene = Scene(planets=[planet], star=star, exozodi=exozodi)

phringe.run(
    Path('config.yaml'),
    Path('system.yaml'),
    # settings,
    # scene=scene,
    spectrum_files=None,  # (('Earth', Path('psg_rad_long.txt')),),
    gpus=(5,),
    output_dir=Path('.'),
    write_fits=False,
    create_copy=False
)

data = phringe.get_data()
wavelengths = phringe.get_wavelength_bin_centers().numpy()
wavelengths = [round(wavelength * 1e6, 1) for wavelength in wavelengths]
time_steps = [int(round(time, 0)) for time in phringe.get_time_steps().numpy()]
# print(time_steps)

plt.figure()
ax = plt.gca()
im = ax.imshow(data[0], cmap='Greys')
plt.title('Synthetic Photometry Data')
# plt.suptitle('Differential Photon Counts', y=0.75, fontsize=14)
# plt.title('Planet + Astrophysical Noise + Instrumental Noise', fontsize=12)
plt.ylabel('Wavelength ($\mu$m)')
plt.xlabel('Time (s)')
# ax.set_yticks([0, 10, 20, 30, 40])
# ax.set_yticklabels([wavelengths[0], wavelengths[10], wavelengths[20], wavelengths[30], wavelengths[40]])
# ax.set_xticks([0, 20, 40, 60, 80, 100, 120, 140, 160])
# ax.set_xticklabels([time_steps[0], time_steps[20], time_steps[40], time_steps[60], time_steps[80],
#                     time_steps[100], time_steps[120], time_steps[140], time_steps[160]])
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im, cax=cax, label='Differential Photon\n Counts')
plt.savefig('photon_counts420.pdf', dpi=500, bbox_inches='tight')
plt.show()

# plt.scatter(range(len(data[0, 20])), data[0, 20], color='teal')
# plt.plot(range(len(data[0, 20])), data[0, 20], color='teal')
# plt.show()

# coefficients = np.polyfit(range(len(data['Earth'][0][:, 10])), data['Sun'][0][:, 10] / np.max(data['Sun'][0][:, 10]), 3)
# fitted_function = np.poly1d(coefficients)
#
# plt.plot(data['Earth'][0][:, 10] / np.max(data['Earth'][0][:, 10]), range(len(data['Earth'][0][:, 10])), label='Earth')
# plt.plot(data['Sun'][0][:, 10] / np.max(data['Sun'][0][:, 10]), range(len(data['Earth'][0][:, 10])), label='Sun')
# plt.plot(fitted_function(range(len(data['Earth'][0][:, 10]))), range(len(data['Earth'][0][:, 10])), label='Fitted Sun')
# plt.legend()
# plt.show()
#
# plt.plot(
#     data['Earth'][0][:, 10] / np.max(data['Earth'][0][:, 10]) - fitted_function(range(len(data['Earth'][0][:, 10]))),
#     range(len(data['Earth'][0][:, 10])), label='Earth')
# plt.plot(data['Sun'][0][:, 10] / np.max(data['Sun'][0][:, 10]) - fitted_function(range(len(data['Earth'][0][:, 10]))),
#          range(len(data['Earth'][0][:, 10])),
#          label='Sun')
# plt.legend()
# plt.show()
