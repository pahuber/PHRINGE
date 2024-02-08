from datetime import datetime
from pathlib import Path

from astropy.io import fits


class FITSWriter():
    """Class representation of the FITS writer.
    """

    def write(self, data, output_dir: Path = None):
        """Write the data to a FITS file.

        :param data: The data to be written to FITS
        :param output_dir: The output directory of the FITS file
        """
        primary = fits.PrimaryHDU()
        header = primary.header
        hdu_list = []
        hdu_list.append(primary)
        for data_per_output in data:
            hdu = fits.ImageHDU(data_per_output)
            hdu_list.append(hdu)
        hdul = fits.HDUList(hdu_list)
        output_dir = Path(output_dir) if output_dir else Path('')
        hdul.writeto(output_dir.joinpath(f'data_{datetime.now().strftime("%Y%m%d_%H%M%S.%f")}.fits'))
