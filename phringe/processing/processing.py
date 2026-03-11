from typing import Union, Tuple

import astropy.units as u
import numpy as np
import torch
from scipy.optimize import leastsq
from scipy.stats import ncx2
from torch import Tensor
from torch.distributions import Normal

from phringe.core.scene import Scene
from phringe.util.baseline import OptimalNullingBaseline
from phringe.util.spectrum import get_blackbody_spectrum_standard_units


def get_sensitivity_limits(
        get_counts: classmethod,
        get_model_counts: classmethod,
        wavelength_bin_centers: Tensor,
        scene: Scene,
        device: torch.device,
        temperature: float,
        pfa: float,
        pdet: float,
        ang_seps_mas: Union[list, np.ndarray, torch.tensor] = np.linspace(10, 150, 2),
        num_reps: int = 1,
        as_radius: bool = True,
        diag_only: bool = False,
) -> Tensor:
    """Return the sensitivity limits of the instrument based on the energy detector and Neyman-Pearson test.
    Returns inf if the planet is outside the fov.


    Returns
    -------
    torch.Tensor
        Sensitivity limits.
    """
    # Prepare arrays
    ang_seps_rad = ang_seps_mas * (1e-3 / 3600) * (np.pi / 180)
    num_ang_seps = len(ang_seps_rad)
    sensitivities = torch.zeros((2, num_reps, num_ang_seps), device=device)
    # i = 1

    # Loop through repetitions i.e. random samples
    for rep in range(num_reps):

        # Get whitening matrix
        noise_ref = get_counts(kernels=True)

        nk = noise_ref.shape[0]
        nl = noise_ref.shape[1]
        nt = noise_ref.shape[2]

        wk = torch.zeros((nk, nl, nl), device=noise_ref.device, dtype=noise_ref.dtype)

        for k in range(nk):
            noise_ref_k = noise_ref[k]
            cov_k = torch.cov(noise_ref_k)
            eigvals, eigvecs = torch.linalg.eigh(cov_k)
            wk[k] = eigvecs @ torch.diag(eigvals.clamp(min=1e-12).rsqrt()) @ eigvecs.T

            # noise_ref = noise_ref.permute(1, 0, 2).reshape(noise_ref.shape[1], -1)

            # plt.imshow(w.cpu().numpy())
            # plt.colorbar()
            # plt.show()

            if diag_only:
                wk[k] = torch.diag(torch.diag(wk[k]))

        # Build block diag w matrix
        w = torch.zeros((nk * nl, nk * nl), device=noise_ref.device, dtype=noise_ref.dtype)
        for k in range(nk):
            w[k * nl:(k + 1) * nl, k * nl:(k + 1) * nl] = wk[k]

        # plt.imshow(w.cpu().numpy())
        # plt.colorbar()
        # plt.show()

        # Loop through angular separations
        for i, ang_sep in enumerate((ang_seps_rad)):
            # Get model
            solid_angle_ref = 1e-20

            x0 = get_model_counts(
                spectral_energy_distribution=get_blackbody_spectrum_standard_units(
                    temperature,
                    wavelength_bin_centers,
                ).cpu().numpy() * solid_angle_ref,
                x_position=ang_sep,
                y_position=0,
                kernels=True
            )
            # plt.imshow(x0[0])
            # plt.colorbar()
            # plt.show()
            # x_shape = x0.shape
            # x0 = x0.transpose(1, 0, 2).reshape(x0.shape[1], -1)
            x0 = x0.reshape(nk * nl, nt)
            x0 = torch.from_numpy(x0).float().to(noise_ref.device)

            # print(i, x0)

            # Whiten model
            xw = w @ x0
            xw = xw.reshape(nk, nl, nt)

            # reshape and permute xw back to original shape
            # xw = xw.reshape(x_shape[1], x_shape[0], x_shape[2]).permute(1, 0, 2)

            #
            xw = xw.flatten()
            s = torch.linalg.norm(xw)
            xtx = torch.dot(xw, xw)

            # Calculate sensitivity limit NP
            std_normal = Normal(0.0, 1.0)
            zfa = std_normal.icdf(torch.tensor(1.0 - pfa, device=s.device))
            zdet = std_normal.icdf(torch.tensor(1 - pdet, device=s.device))
            omega_min = solid_angle_ref * (zfa - zdet) / s  # minimal solid angle (sr) to hit (pfa,pdet)
            sensitivities[1, rep, i] = omega_min

            # Calculate sensitivity limit ED
            df = xw.numel()

            # H0 threshold (central chi-square)
            threshold = torch.tensor(
                ncx2.ppf(1.0 - pfa, df=df, nc=0),
                device=xw.device,
                dtype=xw.dtype
            )

            # Find the non-centrality λ giving Pdet
            # λ is the non-centrality parameter of χ²(df, λ)
            # We need P(T > threshold | λ) = pdet
            def residual(lmbda):
                return (
                        1.0 - ncx2.cdf(threshold.cpu().numpy(), df, float(lmbda))
                        - pdet
                )

            lmbda0 = df * 1e-1
            lmbda_sol = leastsq(residual, lmbda0)[0][0]
            lmbda = torch.tensor(lmbda_sol, device=xw.device, dtype=xw.dtype)
            omega_min = solid_angle_ref * torch.sqrt(lmbda / xtx)
            sensitivities[0, rep, i] = omega_min

    if as_radius:
        try:
            sensitivities = scene.star.distance * torch.sqrt(sensitivities / torch.pi) / (1 * u.Rearth).to(
                u.m).value
        except:
            sensitivities = scene.exozodi.host_star_distance * torch.sqrt(sensitivities / torch.pi) / (
                    1 * u.Rearth).to(
                u.m).value

    return sensitivities


def get_detection_probabilities1(
        get_counts: classmethod,
        get_model_counts: classmethod,
        wavelength_bin_centers: Tensor,
        scene,
        device: torch.device,
        temperature: float,
        ang_sep_mas: float,
        radius_earth: Union[float, np.ndarray, Tensor],
        pfa: float = 2.9e-7,
        num_reps: int = 1,
        diag_only: bool = False,
) -> dict:
    """
    Given a planet radius (in Earth radii), return Pdet for:
      - Neyman-Pearson matched filter (NP)
      - Energy detector (ED)
    at a fixed Pfa, for each angular separation.

    Returns a dict with:
      - "pdet_np": Tensor [num_reps, num_ang_seps]
      - "pdet_ed": Tensor [num_reps, num_ang_seps]
    """

    # ---- helpers: distance and omega <-> radius ----
    def get_star_distance_m(scene_obj) -> float:
        # Prefer scene.star.distance; fallback to exozodi.host_star_distance
        try:
            d = scene_obj.star.distance
        except Exception:
            d = scene_obj.exozodi.host_star_distance
        # d should already be in meters in your codebase; if it's an astropy quantity, convert
        if hasattr(d, "to"):
            d = d.to(u.m).value
        return float(d)

    d_m = get_star_distance_m(scene)
    Rearth_m = (1 * u.Rearth).to(u.m).value

    # Convert input radius (Earth radii) -> omega (sr):  omega = pi * (R/d)^2
    radius_earth_t = torch.as_tensor(radius_earth, device=device, dtype=torch.float32)
    radius_m_t = radius_earth_t * Rearth_m
    omega_in = torch.pi * (radius_m_t / d_m) ** 2  # sr

    # Prepare angular separations
    ang_sep_rad = torch.as_tensor(ang_sep_mas, device=device, dtype=torch.float32) * (
            (1e-3 / 3600) * (np.pi / 180)
    )
    num_ang_seps = 1

    # Output arrays
    pdet_np = torch.zeros((num_reps, num_ang_seps), device=device, dtype=torch.float32)
    pdet_ed = torch.zeros((num_reps, num_ang_seps), device=device, dtype=torch.float32)

    # Broadcast omega to [num_reps, num_ang_seps] if needed
    # Accept scalar, [num_ang_seps], or [num_reps, num_ang_seps]
    if omega_in.ndim == 0:
        omega_grid = omega_in.expand(num_reps, num_ang_seps)
    elif omega_in.ndim == 1:
        if omega_in.numel() != num_ang_seps:
            raise ValueError("If radius_earth is 1D, it must have length == num_ang_seps.")
        omega_grid = omega_in.unsqueeze(0).expand(num_reps, num_ang_seps)
    elif omega_in.ndim == 2:
        if omega_in.shape != (num_reps, num_ang_seps):
            raise ValueError("If radius_earth is 2D, it must have shape (num_reps, num_ang_seps).")
        omega_grid = omega_in
    else:
        raise ValueError("radius_earth must be scalar, 1D [num_ang_seps], or 2D [num_reps,num_ang_seps].")

    std_normal = Normal(0.0, 1.0)

    # Loop repetitions (noise draws)
    for rep in range(num_reps):

        # --- whitening matrix from noise reference ---
        noise_ref = get_counts(kernels=True)
        noise_ref = noise_ref.permute(1, 0, 2).reshape(noise_ref.shape[1], -1)
        cov = torch.cov(noise_ref)

        eigvals, eigvecs = torch.linalg.eigh(cov)
        w = eigvecs @ torch.diag(eigvals.clamp(min=1e-12).rsqrt()) @ eigvecs.T
        if diag_only:
            w = torch.diag(torch.diag(w))

        # --- fixed H0 threshold for ED for this rep (depends on df and pfa only) ---
        # (df may vary if your flattening size varies; here it’s constant across ang_seps)
        # We'll compute df from first model once per rep, but easiest is inside loop after xw exists.

        # for j, ang_sep in enumerate(ang_seps_rad):

        # Model at unit reference solid angle
        solid_angle_ref = 1e-20

        x0 = get_model_counts(
            spectral_energy_distribution=get_blackbody_spectrum_standard_units(
                temperature,
                wavelength_bin_centers,
            ).cpu().numpy(),
            x_position=float(ang_sep_rad),
            y_position=0,
            kernels=True,
        )
        x0 = x0.transpose(1, 0, 2).reshape(x0.shape[1], -1) * solid_angle_ref
        x0 = torch.from_numpy(x0).float().to(noise_ref.device)

        # Whiten model
        xw = (w @ x0).flatten()
        s = torch.linalg.norm(xw)  # ||xw||
        xtx = torch.dot(xw, xw)  # xw^T xw  (== s^2)

        # Requested omega for this (rep, ang_sep)
        omega = omega_grid[rep, 0]

        # ---------------- NP: invert closed form ----------------
        # Original: omega_min = solid_angle_ref * (zfa - zdet) / s
        # where zfa = Phi^{-1}(1-pfa) and zdet = Phi^{-1}(1-pdet)
        zfa = std_normal.icdf(torch.tensor(1.0 - pfa, device=device, dtype=torch.float32))
        zdet = zfa - (omega / solid_angle_ref) * s
        # 1 - pdet = Phi(zdet)  => pdet = 1 - Phi(zdet)
        pdet_np[rep, 0] = 1.0 - std_normal.cdf(zdet)

        # ---------------- ED: use noncentral chi-square CDF ----------------
        # Original: omega_min = solid_angle_ref * sqrt(lambda / xtx)
        # => lambda = (omega/solid_angle_ref)^2 * xtx
        df = int(xw.numel())
        threshold = ncx2.ppf(1.0 - pfa, df=df, nc=0.0)  # scalar float

        lmbda = ((omega / solid_angle_ref) ** 2) * xtx
        lmbda_cpu = float(lmbda.detach().cpu().item())
        # Pdet = P(T > threshold | df, lambda) = 1 - CDF(threshold)
        pdet_ed_val = 1.0 - ncx2.cdf(threshold, df=df, nc=lmbda_cpu)
        pdet_ed[rep, 0] = torch.tensor(pdet_ed_val, device=device, dtype=torch.float32)

    return pdet_ed[0, 0].item(), pdet_np[0, 0].item()


def get_detection_probabilities(
        get_counts: classmethod,
        get_model_counts: classmethod,
        wavelength_bin_centers: Tensor,
        scene,
        device: torch.device,
        temperature: float,
        ang_seps_mas: Union[float, np.ndarray, Tensor],
        radius_earth: Union[float, np.ndarray, Tensor],
        pfa: float = 2.9e-7,
        num_reps: int = 1,
        diag_only: bool = False,
        solid_angle_ref: float = 1e-20,
) -> Tuple[Tensor, Tensor]:
    """
    Returns Pdet for:
      - Energy detector (ED)
      - Neyman–Pearson matched filter (NP)
    at fixed Pfa.

    Parameters
    ----------
    ang_seps_mas : scalar or array-like
        Angular separations in mas. Output has this as the last dimension.
    radius_earth : scalar or array-like
        Planet radius in Earth radii. Can be scalar, [A], [R], or [R, A].
        Will be broadcast against ang_seps_mas.

    Returns
    -------
    pdet_ed : Tensor
        Shape [num_reps, ...broadcasted..., num_ang_seps]
    pdet_np : Tensor
        Shape [num_reps, ...broadcasted..., num_ang_seps]
    """

    # ---- distance ----
    try:
        d = scene.star.distance
    except Exception:
        d = scene.exozodi.host_star_distance
    d_m = float(d.to(u.m).value) if hasattr(d, "to") else float(d)
    Rearth_m = (1 * u.Rearth).to(u.m).value

    # ---- angular separations (mas -> rad) ----
    ang_seps_mas_t = torch.as_tensor(ang_seps_mas, device=device, dtype=torch.float32)
    ang_seps_rad = ang_seps_mas_t * ((1e-3 / 3600) * (np.pi / 180))
    ang_seps_rad = ang_seps_rad.flatten()
    A = ang_seps_rad.numel()

    # ---- radius -> omega via small-angle disk solid angle ----
    # omega = pi*(R/d)^2
    r_earth_t = torch.as_tensor(radius_earth, device=device, dtype=torch.float32)
    r_m_t = r_earth_t * Rearth_m
    omega = torch.pi * (r_m_t / d_m) ** 2  # sr

    # We want omega to broadcast with [A] separations
    # Convert to shape [..., A]
    if omega.ndim == 0:
        omega_grid = omega.view(1).expand(A)  # [A]
    elif omega.ndim == 1:
        if omega.numel() == A:
            omega_grid = omega  # [A]
        else:
            # interpret as [R] and broadcast to [R, A]
            omega_grid = omega[:, None].expand(omega.numel(), A)
    elif omega.ndim == 2:
        if omega.shape[-1] != A:
            raise ValueError("If radius_earth is 2D, its last dimension must match len(ang_seps_mas).")
        omega_grid = omega  # [..., A]
    else:
        raise ValueError("radius_earth must be scalar, 1D, or 2D.")

    # Output: [num_reps, ...omega_prefix..., A]
    omega_prefix_shape = omega_grid.shape[:-1]
    pdet_ed = torch.zeros((num_reps, *omega_prefix_shape, A), device=device, dtype=torch.float32)
    pdet_np = torch.zeros_like(pdet_ed)

    std_normal = Normal(0.0, 1.0)
    zfa = std_normal.icdf(torch.tensor(1.0 - pfa, device=device, dtype=torch.float32))

    # Loop reps (noise realisations)
    for rep in range(num_reps):
        # --- whitening matrix from noise reference ---
        noise_ref = get_counts(kernels=True)
        noise_ref = noise_ref.permute(1, 0, 2).reshape(noise_ref.shape[1], -1)
        cov = torch.cov(noise_ref)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        w = eigvecs @ torch.diag(eigvals.clamp(min=1e-12).rsqrt()) @ eigvecs.T
        if diag_only:
            w = torch.diag(torch.diag(w))

        # Loop separations (likely cannot vectorise because get_model_counts is scalar in position)
        for j, ang_sep in enumerate(ang_seps_rad):
            # --- model at reference solid angle (match your sensitivity code convention) ---
            # IMPORTANT: keep the same convention as in get_sensitivity_limits.
            x0 = get_model_counts(
                spectral_energy_distribution=(
                        get_blackbody_spectrum_standard_units(temperature, wavelength_bin_centers)
                        .cpu().numpy() * solid_angle_ref
                ),
                x_position=float(ang_sep.item()),
                y_position=0,
                kernels=True,
            )
            x0 = x0.transpose(1, 0, 2).reshape(x0.shape[1], -1)
            x0 = torch.from_numpy(x0).float().to(device)

            # Whiten
            xw = (w @ x0).flatten()
            s = torch.linalg.norm(xw)  # ||W x_ref||
            xtx = torch.dot(xw, xw)  # == s^2

            # Grab omega for this separation with broadcasting
            # omega_grid[..., j] has shape omega_prefix_shape
            omega_j = omega_grid[..., j]

            # ---------- NP ----------
            # From omega_min = omega_ref * (zfa - zdet)/s
            # => zdet = zfa - (omega/omega_ref)*s
            zdet = zfa - (omega_j / solid_angle_ref) * s
            pdet_np[rep, ..., j] = 1.0 - std_normal.cdf(zdet)

            # ---------- ED ----------
            # threshold gamma from central chi-square (nc=0)
            df = int(xw.numel())
            gamma = ncx2.ppf(1.0 - pfa, df=df, nc=0.0)

            # lambda = (omega/omega_ref)^2 * (xw^T xw)
            lam = ((omega_j / solid_angle_ref) ** 2) * xtx

            # SciPy cdf wants python floats/ndarrays on CPU
            lam_cpu = lam.detach().cpu().numpy()
            pdet_ed_val = 1.0 - ncx2.cdf(gamma, df=df, nc=lam_cpu)
            pdet_ed[rep, ..., j] = torch.as_tensor(pdet_ed_val, device=device, dtype=torch.float32)

    return pdet_ed, pdet_np


def get_sep_at_max_mod_eff(
        current_nulling_baseline: float,
        optimal_nulling_baseline: OptimalNullingBaseline,
        get_instrument_response: classmethod,
        wavelength_bin_centers: Tensor,
) -> Union[float, tuple]:
    """Return the separation at maximum modulation efficiency in units of (optimized wavelength / nulling baseline).

    Parameters
    ----------
    optimal_nulling_baseline : OptimalNullingBaseline
        Optimal nulling baseline object to extract the optimized wavelength that was used in the setup.

    Returns
    -------
    torch.Tensor
        Separation at maximum modulation efficiency.
    """
    optimized_wavelength = optimal_nulling_baseline.wavelength

    ir = get_instrument_response(kernels=True, perturbations=False,
                                 fov=2 * optimized_wavelength / current_nulling_baseline)

    mod_eff = torch.sqrt(torch.sum(ir ** 2, dim=2))

    closest_wavelength_idx = (torch.abs(wavelength_bin_centers - optimized_wavelength)).argmin()

    seps = []

    for i in range(mod_eff.shape[0]):
        mod_eff_curve = torch.diag(mod_eff[i, closest_wavelength_idx]).cpu().numpy()

        l = len(mod_eff_curve)
        mod_eff_curve = mod_eff_curve[l // 2:]
        # curve += mod_eff_curve

        max_idx = np.argmax(mod_eff_curve)
        max_sep = np.linspace(0, np.sqrt(2), l // 2)[max_idx]

        seps.append(max_sep)

    return tuple(seps) if len(seps) > 1 else seps[0]

    # def get_total_modulation_efficiency(self) -> Tensor:
    #     """Return the total modulation efficiency.
    #
    #     Returns
    #     -------
    #     torch.Tensor
    #         Total modulation efficiency.
    #     """
    #     angles = torch.linspace(0, 300, 100)
    #     bb = torch.ones(self._instrument.number_of_inputs)
    #     perturbations = False
    #
    #     times = self.simulation_time_steps[None, :, None, None]
    #     wavelengths = self._instrument.wavelength_bin_centers[:, None, None, None]
    #     x_coordinates, y_coordinates = get_meshgrid(
    #         2 * 300 * 1e-3 / 3600 * np.pi / 180 / 2,
    #         self._grid_size,
    #         self._device
    #     )
    #     x_coordinates = x_coordinates[None, None, :, :]
    #     y_coordinates = y_coordinates[None, None, :, :]
    #
    #     amplitude_pert_time_series = self._instrument.amplitude_perturbation.time_series if (
    #             self._instrument.amplitude_perturbation is not None and perturbations) else torch.zeros(
    #         (self._instrument.number_of_inputs, len(self.simulation_time_steps)),
    #         dtype=torch.float32,
    #         device=self._device
    #     )
    #     phase_pert_time_series = self._instrument.phase_perturbation.time_series if (
    #             self._instrument.phase_perturbation is not None and perturbations) else torch.zeros(
    #         (self._instrument.number_of_inputs, len(self._instrument.wavelength_bin_centers),
    #          len(self.simulation_time_steps)),
    #         dtype=torch.float32,
    #         device=self._device
    #     )
    #     polarization_pert_time_series = self._instrument.polarization_perturbation.time_series if (
    #             self._instrument.polarization_perturbation is not None and perturbations) else torch.zeros(
    #         (self._instrument.number_of_inputs, len(self.simulation_time_steps)),
    #         dtype=torch.float32,
    #         device=self._device
    #     )
    #
    #     tme = self._instrument._tme_torch(
    #         times,
    #         wavelengths,
    #         x_coordinates,
    #         y_coordinates,
    #         self._observation.modulation_period,
    #         self.get_nulling_baseline(),
    #         *[self._instrument._get_amplitude(self._device) for _ in range(self._instrument.number_of_inputs)],
    #         *[amplitude_pert_time_series[k][None, :, None, None] for k in
    #           range(self._instrument.number_of_inputs)],
    #         *[phase_pert_time_series[k][:, :, None, None] for k in
    #           range(self._instrument.number_of_inputs)],
    #         *[torch.tensor(0) for _ in range(self._instrument.number_of_inputs)],
    #         *[polarization_pert_time_series[k][None, :, None, None] for k in
    #           range(self._instrument.number_of_inputs)],
    #         angles,
    #     )
    #     return tme
    #
    #
    # def get_spectral_covariance(self, amplitude_var, phase_var, polarization_var):
    #     times = self.simulation_time_steps[None, :, None, None]
    #     wavelength_bin_centers = self._instrument.wavelength_bin_centers[:, None, None, None]
    #     wavelength_bin_widths = self._instrument.wavelength_bin_widths[:, None, None, None]
    #     alpha = self._scene.star._sky_coordinates[0][None, None, :, :]
    #     beta = self._scene.star._sky_coordinates[1][None, None, :, :]
    #     star_sky_brightness_distribution = self._scene.star._sky_brightness_distribution[:, None, :, :]
    #     modulation_period = self._observation.modulation_period
    #     nulling_baseline = self.get_nulling_baseline()
    #     detector_integration_time = self._observation.detector_integration_time
    #     amplitudes = [self._instrument._get_amplitude(self._device) for _ in range(self._instrument.number_of_inputs)]
    #     perturbation_covariance = torch.diag(
    #         torch.ones(3 * self._instrument.number_of_inputs, dtype=torch.float32, device=self._device)
    #     )
    #     for i in range(3 * self._instrument.number_of_inputs):
    #         if i < self._instrument.number_of_inputs:
    #             perturbation_covariance[i, i] *= amplitude_var
    #         elif i < 2 * self._instrument.number_of_inputs:
    #             perturbation_covariance[i, i] *= phase_var
    #         else:
    #             perturbation_covariance[i, i] *= polarization_var
    #
    #     # Get lambdified functions
    #     l_func, H_func = self._instrument._get_lambdified_spectral_covariance()
    #
    #     # Get l and stack into tensor
    #     l = l_func(
    #         times,
    #         wavelength_bin_centers,
    #         alpha,
    #         beta,
    #         modulation_period,
    #         nulling_baseline,
    #         *amplitudes,
    #     )
    #
    #     ref = None
    #     for row in l:
    #         for elem in row:
    #             if isinstance(elem, torch.Tensor):
    #                 ref = elem
    #                 break
    #         if ref is not None:
    #             break
    #
    #     if ref is None:
    #         raise RuntimeError("No tensor found in output.")
    #
    #     target_shape = ref.shape
    #     device = ref.device
    #     dtype = ref.dtype
    #
    #     l_fixed = [
    #         [
    #             (elem if isinstance(elem, torch.Tensor)
    #              else torch.full(target_shape, float(elem), device=device, dtype=dtype))
    #             for elem in row
    #         ]
    #         for row in l
    #     ]
    #
    #     l = torch.stack([torch.stack(row, dim=0) for row in l_fixed], dim=0)
    #
    #     # Calc tilde l
    #     tilde_l = self._instrument.quantum_efficiency * self._observation.detector_integration_time * \
    #               self.get_wavelength_bin_widths()[None, None, :, None] * torch.sum(
    #         l * star_sky_brightness_distribution[None, None, :, :, :, :],
    #         dim=(-1, -2)
    #     )
    #
    #     # Get H and stack into tensor
    #     H = H_func(
    #         times,
    #         wavelength_bin_centers,
    #         alpha,
    #         beta,
    #         modulation_period,
    #         nulling_baseline,
    #         *amplitudes,
    #     )
    #
    #     ref = None
    #     for row in H:
    #         for row2 in row:
    #             for elem in row2:
    #                 if isinstance(elem, torch.Tensor):
    #                     ref = elem
    #                     break
    #         if ref is not None:
    #             break
    #
    #     if ref is None:
    #         raise RuntimeError("No tensor found in output.")
    #
    #     target_shape = ref.shape
    #     device = ref.device
    #     dtype = ref.dtype
    #
    #     # print(target_shape)
    #
    #     def ensure_tensor(x):
    #         return x if isinstance(x, torch.Tensor) else torch.zeros(target_shape, device=device, dtype=dtype)
    #
    #     # First convert every single entry, THEN stack three levels
    #     H_fixed = [
    #         [
    #             [ensure_tensor(elem) for elem in row]  # m2
    #             for row in block  # m1
    #         ]
    #         for block in H  # j
    #     ]
    #
    #     for j1, i1 in enumerate(H):
    #         for j2, i2 in enumerate(i1):
    #             for j3, i3 in enumerate(i2):
    #                 if not isinstance(i3, torch.Tensor):
    #                     H[j1][j2][j3] = torch.zeros(target_shape, device=device, dtype=dtype)
    #
    #     def bla(x):
    #         if x.shape != target_shape:
    #             x = torch.zeros(target_shape, device=device, dtype=dtype)
    #         return x
    #
    #     H_tensor = torch.stack(
    #         [
    #             torch.stack(
    #                 [
    #                     torch.stack([
    #                         bla(elem) for elem in row
    #                     ], dim=0)
    #                     for row in block  # stack m1
    #                 ],
    #                 dim=0
    #             )
    #             for block in H_fixed  # stack j
    #         ],
    #         dim=0
    #     )
    #
    #     tilde_H = self._instrument.quantum_efficiency * self._observation.detector_integration_time * \
    #               self.get_wavelength_bin_widths()[None, None, None, :, None] * torch.sum(
    #         H_tensor * star_sky_brightness_distribution[None, None, None, :, :, :, :],
    #         dim=(-1, -2)
    #     )
    #
    #     # Get kernels
    #     kernels_torch = torch.tensor(self._instrument.kernels.tolist(), dtype=torch.float32, device=self._device)
    #
    #     # diff_ir = torch.einsum('ij, jklmn -> iklmn', kernels_torch, ir)
    #     # print(tilde_l.shape)
    #     # print(tilde_H.shape)
    #     # print(kernels_torch.shape)
    #
    #     tilde_l_kernel = tilde_l
    #     # tilde_l_kernel = torch.einsum('ij, jklm -> iklm', kernels_torch, tilde_l)
    #     tilde_H_kernel = tilde_H
    #     # tilde_H_kernel = torch.einsum('ij, jklmn -> iklmn', kernels_torch, tilde_H)
    #
    #     tilde_l_reduced = tilde_l_kernel  # .mean(dim=(-1))
    #     Lambda_kernel = tilde_l_reduced.permute(0, 2, 1, 3)
    #
    #     # print(tilde_l_reduced.shape)
    #     # print(Lambda_kernel.shape)
    #
    #     tilde_H_reduced = tilde_H_kernel  # .mean(dim=(-1))
    #     shape = tilde_H_reduced.shape
    #     # print(shape)
    #     Gamma_kernel = tilde_H_reduced.reshape((shape[0], shape[1] ** 2, shape[3], shape[4])).permute(0, 2, 1, 3)
    #
    #     # print(Gamma_kernel.shape)
    #
    #     kron = torch.kron(perturbation_covariance, perturbation_covariance)
    #
    #     # print(kron.shape)
    #     #
    #     # cov1 = (Lambda_kernel @ perturbation_covariance @ Lambda_kernel.transpose(1, 2) + \
    #     #         2 * Gamma_kernel @ kron @ Gamma_kernel.transpose(1, 2)).mean(dim=-1)
    #
    #     # print(cov1)
    #     linear = torch.einsum(
    #         'o l m t, m n, o k n t -> o l k t',
    #         Lambda_kernel, perturbation_covariance, Lambda_kernel
    #     ).mean(dim=-1)  # -> (o, l, k)
    #
    #     quad = torch.einsum(
    #         'o l a t, a b, o k b t -> o l k t',
    #         Gamma_kernel, kron, Gamma_kernel
    #     ).mean(dim=-1)
    #     quad = 2 * quad
    #
    #     cov = linear + quad
    #
    #     # print(cov1.shape)
    #
    #     return cov
