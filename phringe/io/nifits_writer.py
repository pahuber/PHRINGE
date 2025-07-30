from copy import copy
from importlib.metadata import version

import numpy as np
from astropy.io import fits
from nifits.io.oifits import NI_CATM
from nifits.io.oifits import NI_FOV_DEFAULT_HEADER, NI_FOV
from nifits.io.oifits import NI_MOD
from nifits.io.oifits import OI_TARGET
from nifits.io.oifits import nifits
from torch import Tensor

from phringe.core.entities.instrument import Instrument
from phringe.core.entities.observation import Observation
from phringe.core.entities.scene import Scene


class NIFITSWriter:
    """Class representation of the NIFITS writer.
    """

    def __init__(self):
        """Initialize the NIFITSWriter."""
        # TODO: Finish implementation of NIFITSWriter

    def write(self, data: Tensor, observation: Observation, instrument: Instrument, scene: Scene):
        """Write the data to a FITS file.

        :param data: The data to be written to FITS
        :type data: Tensor
        :param observation: The observation parameters
        :type observation: Observation
        :param instrument: The instrument parameters
        :type instrument: Instrument
        :param scene: The scene parameters
        :type scene: Scene
        """
        ni_catm = NI_CATM(data_array=np.array(instrument.complex_amplitude_transfer_matrix.evalf()))

        fov_header = copy(NI_FOV_DEFAULT_HEADER)
        fov_header['FOV_TELDIAM'] = instrument.aperture_diameter.item()
        fov_header['FOV-TELDIAM_UNIT'] = 'm'

        ni_fov = NI_FOV.simple_from_header(
            header=fov_header,
            lamb=instrument.wavelength_bin_centers.cpu().numpy(),
            n=int(observation.total_integration_time.item() // observation.detector_integration_time.item())
        )

        oi_target = OI_TARGET()

        ni_mod = NI_MOD()

        header = fits.Header()
        header['CREATOR'] = (f'PHRINGE v{version("phringe")}', 'Created with github.com/pahuber/PHRINGE')

        nifits_file = nifits(
            header=header,
            # ni_catm=ni_catm,
            ni_fov=ni_fov,
            oi_target=oi_target,
            ni_mod=ni_mod,
        )

        nifits_file.to_nifits('a.nifits', overwrite=True)
