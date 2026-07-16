from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Union

import nifits.io.niio as io
import numpy as np
import sympy as sp
import torch
from astropy import units as u
from astropy.table import Table, Column
from astropy.time import Time

if TYPE_CHECKING:
    from phringe.main import PHRINGE


class NIFITSWriter:
    """Class representation of the NIFITS writer.
    """

    @staticmethod
    def _get_header(start_time, array, observer, target_object) -> io.fits.Header:
        header = copy(
            io.NI_NIFITS_DEFAULT_HEADER
            + io.NI_NIFITS_OPTIONAL_KEYWORDS
        )

        header["DATE"] = Time.now().isot
        header["DATE-OBS"] = start_time.isot
        header["MJD-OBS"] = start_time.mjd
        header["ORIGIN"] = "PHRINGE Simulation"
        header["TELESCOP"] = array
        header["INSTRUME"] = array
        header["OBSERVER"] = observer
        header["OBJECT"] = target_object
        header["INSMODE"] = "Nulling Interferometry"

        return header

    @staticmethod
    def _get_ni_catm(phringe, wl_bin_centers) -> io.NI_CATM:
        one_catm_numpy = np.array(
            phringe._instrument.complex_amplitude_transfer_matrix.evalf(),
            dtype=np.complex_
        )
        my_ni_catm = np.array([one_catm_numpy for _ in wl_bin_centers])

        return io.NI_CATM(data_array=my_ni_catm)

    @staticmethod
    def _get_ni_fov(phringe, wl_bin_centers, n_t_frames) -> io.NI_FOV:
        fov_header = copy(io.NI_FOV_DEFAULT_HEADER)
        fov_header["FOV_TELDIAM"] = (phringe._instrument.aperture_diameter.item())
        fov_header["FOV_TELDIAM_UNIT"] = "m"

        ni_fov = io.NI_FOV.simple_from_header(
            fov_header,
            lamb=wl_bin_centers,
            n=n_t_frames
        )

        return ni_fov

    @staticmethod
    def _get_ni_mod(n_t_frames, n_wl_ch, n_tel, phringe, time_frames) -> io.NI_MOD:
        def _get_times_relative():
            return Column(data=time_frames - time_frames[0], name="TIME", unit="", dtype=float)

        def _get_int_times():
            overhead_for_nifits = 1  # Placeholder to include overhead, currently set to 1.

            return Column(
                data=np.gradient(time_frames - time_frames[0]) * overhead_for_nifits,
                name="INT_TIME",
                unit="s",
                dtype=float
            )

        def _get_mod_phase():
            mod_phas = []
            for atime in time_frames:
                # Placeholder for phase shift due to wavefront effects in space
                g_wavefront = np.zeros((n_wl_ch, n_tel), dtype=complex)
                # Placeholder for factor representing amplitude loss inside the instrument
                g_internal = np.ones((n_wl_ch, n_tel), dtype=complex)
                throughput = np.ones(n_wl_ch) * phringe._instrument.throughput
                my_total_phasor = throughput[:, None] * np.exp(1j * g_wavefront) * g_internal
                # original line: total_phasor = throughput[:, None]
                #                              * np.exp(1j*g_wavefront) * g_internal
                mod_phas.append(my_total_phasor)
            mod_phas = np.array(mod_phas).swapaxes(0, 1).swapaxes(1, 2)

            return Column(data=mod_phas, name="MOD_PHAS", unit=None, dtype=complex)

        def _get_mjd():
            return Column(data=Time.now().mjd + np.asarray(times_relative) / 86400.0, name="MJD", unit="d")

        def _get_ap_xy():
            t, tm, b = sp.symbols('t tm b')
            acm_w_b_tm = (phringe._instrument.array_configuration_matrix.subs(
                [(b, phringe._observation._nulling_baseline),
                 (tm, phringe._observation.modulation_period)]))
            acm_w_b_tm_num = sp.lambdify([t], acm_w_b_tm, modules="numpy")
            # arraylocpertimestep = np.array([acm_w_b_tm_num(t) for t in all_time_steps])
            # reshaped_array = arraylocpertimestep.reshape(n_t_frames, dit_per_frame, *arraylocpertimestep.shape[1:])
            # appxy_frames_for_nifits = np.mean(reshaped_array, axis=1).transpose(0, 2, 1)
            ap_xy = phringe.get_collector_positions().swapaxes(0, 1)

            return Column(data=ap_xy, name="AP_XY", unit="m", dtype=float)

        def _get_arr_col():
            arrcol = np.ones((n_t_frames, n_tel)) * (np.pi * np.array(phringe._instrument.aperture_diameter) ** 2 / 4)
            arrcol = arrcol.swapaxes(0, 1
                                     )
            return Column(data=arrcol, name="COL_AR", unit="m^2", dtype=float)

        def _get_fov_index():
            fov_index = np.ones(len(time_frames))

            return Column(data=fov_index, name="FOV_INDEX", unit=None, dtype=int)

        times_relative = _get_times_relative()
        int_times = _get_int_times()
        mod_phas = _get_mod_phase()
        mjd = _get_mjd()
        ap_xy = _get_ap_xy()
        arr_col = _get_arr_col()
        fov_index = _get_fov_index()

        mod_phas, ap_xy, arr_col = [
            np.moveaxis(array, -1, 0)
            for array in (mod_phas, ap_xy, arr_col)
        ]

        mymod_table = Table()
        mymod_table.add_columns((times_relative, int_times, mod_phas, mjd, ap_xy, arr_col, fov_index))

        return io.NI_MOD(mymod_table)

    @staticmethod
    def _get_ni_iout(phringe) -> io.NI_IOUT:
        niout_frames = phringe.get_counts().cpu().numpy().swapaxes(0, 2)  # TODO: Are units as expected by NIFITS? ph/s

        Iout_table = Table(data=(niout_frames,),
                           names=("VALUE",),
                           dtype=(float,), )
        Iout_header = io.fits.Header()
        myiout = io.NI_IOUT(data_table=Iout_table, header=Iout_header,
                            unit=u.photon / u.s)

        return myiout

    @staticmethod
    def _get_ni_kiout(mydata) -> io.NI_KIOUT:
        # kiout_frames = np.mean(mydata[0, :, :].reshape(n_wl_ch, n_t_frames, dit_per_frame), axis=-1)
        kiout_frames = np.transpose(mydata, (2, 1, 0))  # TODO: Are units as expected by NIFITS? ph/s
        KIout_table = Table(data=(kiout_frames,), names=("VALUE",), dtype=(float,))
        KIout_header = io.fits.Header()
        mykiout = io.NI_KIOUT(data_table=KIout_table, header=KIout_header, unit=u.photon / u.s)

        return mykiout

    @staticmethod
    def _get_ni_kcov(mydata, n_t_frames, dit_per_frame=20) -> io.NI_KCOV:  # TODO: Check dit per frame
        mykcov = np.array([np.cov(mydata[0, :, it * dit_per_frame: it * dit_per_frame + dit_per_frame])
                           for it in range(n_t_frames)])
        kcov_header = io.fits.Header()
        kcov_header["SHAPE"] = ("frame (wavelength output)",
                                "The shape of the covariance array.")
        mykcov = io.NI_KCOV(data_array=mykcov, header=kcov_header, unit=u.photon ** 2 / u.s ** 2)
        mykcov.name = "NI_KCOV"

        return mykcov

    @staticmethod
    def _get_ni_kmat(phringe) -> io.NI_KMAT:
        kmat = phringe._instrument.kernels.tolist()
        mykmat = io.NI_KMAT(data_array=np.array(kmat, dtype=float))

        return mykmat

    @staticmethod
    def _get_oi_wavelength(wl_bin_centers, wl_bin_widths) -> io.OI_WAVELENGTH:
        oi_wl_col = Column(data=wl_bin_centers, name="EFF_WAVE", dtype=float)
        oi_wlw_col = Column(data=wl_bin_widths, name="EFF_BAND", dtype=float)

        return io.OI_WAVELENGTH(data_table=Table([oi_wl_col, oi_wlw_col]), )

    def write(
            self,
            phringe: PHRINGE,
            counts: Union[torch.Tensor, np.ndarray],
            start_time: Time,
            array: str,
            observer: str,
            target_object: str,
            output_dir: Path = Path('.'),
            fits_suffix: str = ''
    ):
        """Write data to NIFITS format.
        """
        time_frames = phringe.get_time_steps().cpu().numpy()
        n_t_frames = len(time_frames)
        wl_bin_centers = phringe.get_wavelength_bin_centers().cpu().numpy()
        wl_bin_widths = phringe.get_wavelength_bin_widths().cpu().numpy()
        n_wl_ch = len(wl_bin_centers)
        n_tel = phringe.get_collector_positions().shape[1]

        if isinstance(counts, torch.Tensor):
            counts = counts.cpu().numpy()

        header = self._get_header(start_time, array, observer, target_object)
        ni_catm = self._get_ni_catm(phringe, wl_bin_centers)
        ni_fov = self._get_ni_fov(phringe, wl_bin_centers, n_t_frames)
        oi_wavelength = self._get_oi_wavelength(wl_bin_centers, wl_bin_widths)
        ni_mod = self._get_ni_mod(n_t_frames, n_wl_ch, n_tel, phringe, time_frames)
        ni_iout = self._get_ni_iout(phringe)
        ni_kiout = self._get_ni_kiout(counts)
        ni_kcov = self._get_ni_kcov(counts, n_t_frames)
        ni_kmat = self._get_ni_kmat(phringe)

        mynifit = io.nifits(
            header=header,
            ni_catm=ni_catm,
            ni_fov=ni_fov,
            oi_wavelength=oi_wavelength,
            ni_mod=ni_mod,
            ni_iout=ni_iout,
            ni_kiout=ni_kiout,
            ni_kcov=ni_kcov,
            ni_kmat=ni_kmat
        )
        mynifit.refresh_primary_header(all=True)

        filename = output_dir.joinpath(f'data{fits_suffix}.nifits')
        hdu = mynifit.to_nifits(filename, overwrite=True)
        print('')
        print('Finished writing to nifits! Filename = ', filename)
