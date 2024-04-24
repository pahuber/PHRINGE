from pathlib import Path

from astropy.io import fits


class FITSWriter():
    """Class representation of the FITS writer.
    """

    def write(self, data, output_dir: Path, fits_suffix: str = ''):
        """Write the data to a FITS file.

        :param data: The data to be written to FITS
        :param output_dir: The output directory of the FITS file
        :param fits_suffix: The suffix of the FITS file
        """
        primary = fits.PrimaryHDU()
        header = primary.header
        hdu_list = []
        hdu_list.append(primary)
        for data_per_output in data:
            hdu = fits.ImageHDU(data_per_output)
            hdu_list.append(hdu)
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(output_dir.joinpath(f'data{fits_suffix}.fits'))
