from pathlib import Path

from astropy.io import fits


class FITSWriter():
    """Class representation of the FITS writer.
    """

    def write(self, data, output_dir: Path, source_name: str = None):
        """Write the data to a FITS file.

        :param data: The data to be written to FITS
        :param output_dir: The output directory of the FITS file
        :param source_name: The name of the source
        """
        primary = fits.PrimaryHDU()
        header = primary.header
        hdu_list = []
        hdu_list.append(primary)
        for data_per_output in data:
            hdu = fits.ImageHDU(data_per_output)
            hdu_list.append(hdu)
        hdul = fits.HDUList(hdu_list)

        if source_name:
            hdul.writeto(output_dir.joinpath(f'data_{source_name}.fits'))
        else:
            hdul.writeto(output_dir.joinpath('data.fits'))
