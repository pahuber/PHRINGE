from pathlib import Path


class NIFITSWriter:
    """Class representation of the NIFITS writer.
    """

    # TODO: Basic implementation of NIFITS

    def write(self, phringe, output_dir: Path, fits_suffix: str = '', dit_per_frame=20):
        """Write data to NIFITS format.
        To be determined:
        0. time parameters
        1. wavelength
        2. catm
        3. fov
        4. mod_phas
        5. iout
        6. kiout
        7. kcov
        8. appxy
        9. kmat
        10. arr_col
        """

        # Import packages
        import numpy as np
        import nifits.io.oifits as io
        from astropy.table import Table, Column
        from astropy import units as u
        from copy import copy
        import torch
        import sympy as sp

        # 0. Time parameters
        all_time_steps = phringe.get_time_steps().cpu().numpy()
        n_t_frames = len(phringe.get_time_steps()) \
                     // dit_per_frame
        time_frames = np.mean(all_time_steps.reshape(n_t_frames, dit_per_frame), axis=1)
        seconds = time_frames - time_frames[0]
        overhead_for_nifits = 1  # Placeholder to include overhead, currently set to 1.
        exptimes_for_nifits = np.gradient(seconds) * overhead_for_nifits

        # 1. Wavelengths
        wl_bin_centers = phringe.get_wavelength_bin_centers().cpu().numpy()
        wl_bin_widths = phringe.get_wavelength_bin_widths().cpu().numpy()
        n_wl_ch = len(wl_bin_centers)
        #
        oi_wl_col = Column(data=wl_bin_centers, name="EFF_WAVE", dtype=float)
        oi_wlw_col = Column(data=wl_bin_widths, name="EFF_BAND", dtype=float)
        my_wl_table = Table([oi_wl_col, oi_wlw_col])
        myiowl = io.OI_WAVELENGTH(data_table=Table([oi_wl_col, oi_wlw_col]), )

        # 2. CATM
        one_catm_numpy = np.array(phringe._instrument.complex_amplitude_transfer_matrix.evalf(),
                                  dtype=np.complex_)
        my_ni_catm = np.array([one_catm_numpy for awl in wl_bin_centers])
        mycatm = io.NI_CATM(data_array=my_ni_catm)
        n_tel = my_ni_catm.shape[-1]

        # 3. FOV
        # inspired from scifysim.director.prepare_injector and scify_space_test
        copy(io.NI_FOV_DEFAULT_HEADER)
        my_FOV_header = copy(io.NI_FOV_DEFAULT_HEADER)
        my_FOV_header["FOV_TELDIAM"] = (phringe._instrument.aperture_diameter.item())
        my_FOV_header["FOV_TELDIAM_UNIT"] = "m"
        my_ni_fov = io.NI_FOV.simple_from_header(header=my_FOV_header,
                                                 lamb=wl_bin_centers,
                                                 n=n_t_frames)

        # 4. MOD
        # inspired from scify_space_test
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
        mod_phas = np.array(mod_phas)

        # 5. IOUT
        niout_frames = np.zeros((n_t_frames, n_wl_ch, n_tel))
        # get_counts returns dimensions [n app, n wls, n t steps]
        iout_counts = phringe.get_counts()
        # nifits expects [n tsteps, n wls, n app]
        reshaped_iout = iout_counts.unfold(dimension=2, size=dit_per_frame, step=dit_per_frame)
        mean_iout = torch.mean(reshaped_iout, dim=-1)
        niout_frames = (mean_iout.permute(2, 1, 0)).cpu().numpy()
        Iout_table = Table(data=(niout_frames,),
                           names=("value",),
                           dtype=(float,), )
        Iout_header = io.fits.Header()
        myiout = io.NI_IOUT(data_table=Iout_table, header=Iout_header,
                            unit=u.photon / u.s)

        # 6. KIOUT
        # phringe.get_data has shape (n differential outputs, n wavelength bins, n timesteps)
        mydata = phringe.get_counts(kernels=True).cpu().numpy()
        kiout_frames = np.mean(mydata[0, :, :].reshape(n_wl_ch, n_t_frames, dit_per_frame), axis=-1)
        KIout_table = Table(data=(kiout_frames,), names=("value",), dtype=(float,))
        KIout_header = io.fits.Header()
        mykiout = io.NI_KIOUT(data_table=KIout_table, header=KIout_header, unit=u.photon / u.s)

        # 7. KCOV
        mykcov = np.array([np.cov(mydata[0, :, it * dit_per_frame: it * dit_per_frame + dit_per_frame])
                           for it in range(n_t_frames)])
        kcov_header = io.fits.Header()
        kcov_header["SHAPE"] = ("frame (wavelength output)",
                                "The shape of the covariance array.")
        mykcov = io.NI_KCOV(data_array=mykcov, header=kcov_header, unit=u.photon ** 2 / u.s ** 2)
        mykcov.name = "NI_KCOV"

        # 8. APPXY
        t, tm, b = sp.symbols('t tm b')
        acm_w_b_tm = (phringe._instrument.array_configuration_matrix.subs(
            [(b, phringe._observation._nulling_baseline),
             (tm, phringe._observation.modulation_period)]))
        acm_w_b_tm_num = sp.lambdify([t], acm_w_b_tm, modules="numpy")
        arraylocpertimestep = np.array([acm_w_b_tm_num(t) for t in all_time_steps])
        reshaped_array = arraylocpertimestep.reshape(n_t_frames, dit_per_frame, *arraylocpertimestep.shape[1:])
        appxy_frames_for_nifits = np.mean(reshaped_array, axis=1).transpose(0, 2, 1)

        # 9. KMAT
        # Currently hard-coded for one differential output. To be revised if there are multiple pairs.
        # diffouts = phringe._instrument.differential_outputs
        # diffouts = [(2, 3), ]
        # kmat = [0] * my_ni_catm.shape[-2]
        # for i, val in enumerate([1, -1]):
        #     kmat[diffouts[0][i]] = val
        kmat = phringe._instrument.kernels.tolist()
        mykmat = io.NI_KMAT(data_array=np.array([kmat], dtype=float))

        # 10. ARRCOL
        arrcol = np.ones((n_t_frames, n_tel)) * (
                np.pi * np.array(phringe._instrument.aperture_diameter) ** 2
                / 4)

        # 11. Take everything and save to nifits
        times_relative = Column(data=time_frames - time_frames[0], name="TIME",
                                unit="", dtype=float)
        int_times = Column(data=np.gradient(time_frames - time_frames[0])
                                * overhead_for_nifits,
                           name="INT_TIME", unit="s", dtype=float)
        mod_phas = Column(data=mod_phas, name="MOD_PHAS",
                          unit=None, dtype=complex)
        appxy = Column(data=appxy_frames_for_nifits, name="APPXY",
                       unit="m", dtype=float)
        arrcol = Column(data=arrcol, name="ARRCOL",
                        unit="m^2", dtype=float)
        fov_index = np.ones(len(time_frames))
        fov_index = Column(data=fov_index, name="FOV_INDEX",
                           unit=None, dtype=int)

        mymod_table = Table()
        mymod_table.add_columns((times_relative, int_times, mod_phas,
                                 appxy, arrcol, fov_index))
        mynimod = io.NI_MOD(mymod_table)
        myheader = io.fits.Header()

        mynifit = io.nifits(header=myheader,
                            ni_catm=mycatm,
                            ni_fov=my_ni_fov,
                            oi_wavelength=myiowl,
                            ni_mod=mynimod,
                            ni_iout=myiout,
                            ni_kiout=mykiout,
                            ni_kcov=mykcov,
                            ni_kmat=mykmat)

        filename = output_dir.joinpath(f'data{fits_suffix}.nifits')
        hdu = mynifit.to_nifits(filename, overwrite=True)
        print('')
        print('Finished writing to nifits! Filename = ', filename)
