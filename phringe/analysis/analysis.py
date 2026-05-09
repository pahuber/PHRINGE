from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Union, Tuple

import astropy.units as u
import numpy as np
import torch
from scipy.optimize import leastsq
from scipy.stats import ncx2
from torch import Tensor
from torch.distributions import Normal

from phringe.util.baseline import OptimalNullingBaseline
from phringe.util.spectrum import get_blackbody_spectrum_si_units

if TYPE_CHECKING:
    from phringe.main import PHRINGE


class Analysis:
    def __init__(self, phringe: PHRINGE):
        self.phringe = phringe

    def get_sensitivity_limits(
            self,
            temperature: float,
            pfa: float = 2.87e-7,
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
        sensitivities = torch.zeros((2, num_reps, num_ang_seps), device=self.phringe._device)
        # i = 1

        # Loop through repetitions i.e. random samples
        for rep in range(num_reps):

            # Get whitening matrix from estimated noise
            noise_ref = self.phringe.get_counts(kernels=True)

            nk = noise_ref.shape[0]
            nl = noise_ref.shape[1]
            nt = noise_ref.shape[2]

            noise_ref = noise_ref.reshape(nk * nl, nt)

            cov = torch.cov(noise_ref)

            U, Svals, _ = torch.linalg.svd(cov)
            w = U @ torch.diag(1 / torch.sqrt(Svals)) @ U.T

            if diag_only:
                w = torch.diag(torch.diag(w))

            # Loop through angular separations
            for i, ang_sep in enumerate((ang_seps_rad)):
                # Get model
                solid_angle_ref = 1e-20

                x0 = self.phringe.get_model_counts(
                    spectral_energy_distribution=get_blackbody_spectrum_si_units(
                        temperature,
                        self.phringe.get_wavelength_bin_centers(),
                    ).cpu().numpy() * solid_angle_ref,
                    x_position=ang_sep,
                    y_position=0,
                    kernels=True
                )

                x0 = x0.reshape(nk * nl, nt)
                x0 = torch.from_numpy(x0).float().to(noise_ref.device)

                # np.save(
                #     f'/home/huberph/phringe/_wd/projects/2026/2_dbw_kernel_comparison/_huber+2026/data_cov_plot/out/cov_{nk}.npy',
                #     cov.cpu().numpy())
                # np.save(
                #     f'/home/huberph/phringe/_wd/projects/2026/2_dbw_kernel_comparison/_huber+2026/data_cov_plot/out/noise_ref_{nk}.npy',
                #     noise_ref.cpu().numpy())
                # np.save(
                #     f'/home/huberph/phringe/_wd/projects/2026/2_dbw_kernel_comparison/_huber+2026/data_cov_plot/out/x0_{nk}.npy',
                #     x0.cpu().numpy())

                # Whiten model
                xw = w @ x0
                xw = xw.reshape(nk, nl, nt)
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
                sensitivities = self.phringe._scene.star.distance * torch.sqrt(sensitivities / torch.pi) / (
                        1 * u.Rearth).to(
                    u.m).value
            except:
                sensitivities = self.phringe._scene.exozodi.host_star_distance * torch.sqrt(
                    sensitivities / torch.pi) / (
                                        1 * u.Rearth).to(
                    u.m).value

        return sensitivities

    def get_detection_probabilities(
            self,
            temperature: float,
            radius_planet: Union[float, np.ndarray, Tensor],
            pfa: float = 2.87e-7,
            ang_seps_mas: Union[float, np.ndarray, Tensor] = np.linspace(10, 150, 2),
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
        radius_planet : scalar or array-like
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
            d = self.phringe._scene.star.distance
        except Exception:
            d = self.phringe._scene.exozodi.host_star_distance
        d_m = float(d.to(u.m).value) if hasattr(d, "to") else float(d)
        Rearth_m = (1 * u.Rearth).to(u.m).value

        # ---- angular separations (mas -> rad) ----
        ang_seps_mas_t = torch.as_tensor(ang_seps_mas, device=self.phringe._device, dtype=torch.float32)
        ang_seps_rad = ang_seps_mas_t * ((1e-3 / 3600) * (np.pi / 180))
        ang_seps_rad = ang_seps_rad.flatten()
        A = ang_seps_rad.numel()

        # ---- radius -> omega via small-angle disk solid angle ----
        # omega = pi*(R/d)^2
        r_earth_t = torch.as_tensor(radius_planet, device=self.phringe._self.phringe._device, dtype=torch.float32)
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
        pdet_ed = torch.zeros((num_reps, *omega_prefix_shape, A), device=self.phringe._device, dtype=torch.float32)
        pdet_np = torch.zeros_like(pdet_ed)

        std_normal = Normal(0.0, 1.0)
        zfa = std_normal.icdf(torch.tensor(1.0 - pfa, device=self.phringe._device, dtype=torch.float32))

        # Loop reps (noise realisations)
        for rep in range(num_reps):
            # --- whitening matrix from noise reference ---
            noise_ref = self.phringe.get_counts(kernels=True)
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
                x0 = self.phringe.get_model_counts(
                    spectral_energy_distribution=(
                            get_blackbody_spectrum_si_units(temperature, self.phringe.get_wavelength_bin_centers())
                            .cpu().numpy() * solid_angle_ref
                    ),
                    x_position=float(ang_sep.item()),
                    y_position=0,
                    kernels=True,
                )
                x0 = x0.transpose(1, 0, 2).reshape(x0.shape[1], -1)
                x0 = torch.from_numpy(x0).float().to(self.phringe._device)

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
                pdet_ed[rep, ..., j] = torch.as_tensor(pdet_ed_val, device=self.phringe._device, dtype=torch.float32)

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
