from typing import Union

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
        pfa: float = 2.9e-7,
        pdet: float = 0.9,
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
    i = 10

    # Loop through repetitions i.e. random samples
    for rep in range(num_reps):

        # Get whitening matrix
        noise_ref = get_counts(kernels=True)
        noise_ref = noise_ref.permute(1, 0, 2).reshape(noise_ref.shape[1], -1)
        cov = torch.cov(noise_ref)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        w = eigvecs @ torch.diag(eigvals.clamp(min=1e-12).rsqrt()) @ eigvecs.T

        if diag_only:
            w = torch.diag(torch.diag(w))

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
            x0 = x0.transpose(1, 0, 2).reshape(x0.shape[1], -1)
            x0 = torch.from_numpy(x0).float().to(noise_ref.device)

            # Whiten model
            xw = w @ x0
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


def get_detection_probabilities(
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
