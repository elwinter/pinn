"""Shared code for Tsyganenko neutral current sheet problems

This file is based on:

N. A. Tsyganenko, V. A. Andreeva, and E. I. Gordeev, "Internally and
externally induced deformations of the magnetospheric equatorial current as
inferred from spacecraft data", Ann. Geophys., 33, 1–11, 2015

www.ann-geophys.net/33/1/2015/
doi:10.5194/angeo-33-1-2015

Author
------
Eric Winter (eric.winter62@gmail.com)
"""


# Import standard modules.

# Import supplemental modules.
import numpy as np

# Import project modules.


# Empirical constants for model, and RMS mean absolute deviation
# These values are from Table 1.
_RH0, _RH0_rms = 11.02, 0.05
_RH1, _RH1_rms = 6.05, 0.88
_RH2, _RH2_rms = 0.84, 0.09
_RH3, _RH3_rms = -2.28, 0.08
_RH4, _RH4_rms = -0.25, 0.37
_RH5, _RH5_rms = -0.96, 0.16
_T0, _T0_rms = 0.29, 0.02
_T1, _T1_rms = 0.18, 0.08
_a00, _a00_rms = 2.91, 0.02
_a01, _a01_rms = -0.16, 0.07
_a02, _a02_rms = 0.56, 0.03
_a10, _a10_rms = 1.89, 0.03
_a11, _a11_rms = 0.06, 0.04
_a12, _a12_rms = 0.49, 0.04
_alpha0, _alpha0_rms = 7.13, 0.06
_alpha1, _alpha1_rms = 4.87, 0.07
_alpha2, _alpha2_rms = -0.22, 0.12
_alpha3, _alpha3_rms = -0.14, 0.04
_chi, _chi_rms = -0.29, 0.03
_beta0, _beta0_rms = 2.18, 0.09
_beta1, _beta1_rms = 0.40, 0.11

# Pressure scale (nPa)
Pmean = 2.0  # P6C1L3

# Magnetic field scale (nT)
By0 = 5.0  # P5C2L6
Bz0 = 5.0  # P6C1L3

# Radius scale (Earth radii)
rho0 = 10.0  # P5C2L7


def RH_empirical(fP, fBz, phi, RH0=_RH0, RH1=_RH1, RH2=_RH2, RH3=_RH3,
                 RH4=_RH4, RH5=_RH5):
    """Equation 4"""
    return RH0 + RH1*fP + RH2*fBz + (RH3 + RH4*fP + RH5*fBz)*np.cos(phi)


def T_empirical(fP, T0=_T0, T1=_T1):
    """Equation 5"""
    return T0 + T1*fP


def a0_empirical(fP, fBz, a00=_a00, a01=_a01, a02=_a02):
    """Equation 6"""
    return a00 + a01*fP + a02*fBz


def a1_empirical(fP, fBz, a10=_a10, a11=_a11, a12=_a12):
    """Equation 7"""
    return a10 + a11*fP + a12*fBz


def alpha_empirical(fP, fBz, phi, alpha0=_alpha0, alpha1=_alpha1,
                    alpha2=_alpha2, alpha3=_alpha3):
    """Equation 8"""
    return alpha0 + alpha1*np.cos(phi) + alpha2*fP + alpha3*fBz


def beta_empirical(fBz, beta0=_beta0, beta1=_beta1):
    """Equation 9"""
    return beta0 + beta1*fBz


def fP_empirical(P, chi=_chi):
    """Equation 10"""
    return (P/Pmean)**chi - 1


def fBz_empirical(Bz):
    """Equation 10"""
    return Bz/Bz0


def Zs_empirical(
    rho, phi, P, By, Bz, psi,
    RH0=_RH0, RH1=_RH1, RH2=_RH2, RH3=_RH3, RH4=_RH4, RH5=_RH5,
    T0=_T0, T1=_T1,
    a00=_a00, a01=_a01, a02=_a02,
    a10=_a10, a11=_a11, a12=_a12,
    alpha0=_alpha0, alpha1=_alpha1, alpha2=_alpha2, alpha3=_alpha3,
    chi=_chi,
    beta0=_beta0, beta1=_beta1
):
    """Equation 3"""
    fP = fP_empirical(P, chi)
    fBz = fBz_empirical(Bz)
    RH = RH_empirical(fP, fBz, phi, RH0, RH1, RH2, RH3, RH4, RH5)
    a0 = a0_empirical(fP, fBz, a00, a01, a02)
    a1 = a1_empirical(fP, fBz, a10, a11, a12)
    T = T_empirical(fP, T0, T1)
    alpha = alpha_empirical(fP, fBz, phi, alpha0, alpha1, alpha2, alpha3)
    beta = beta_empirical(fBz, beta0, beta1)
    Zs = (
        RH*np.tan(psi) *
        (1 - (1 + (rho/RH)**alpha)**(1/alpha)) *
        (a0 + a1*np.cos(phi)) +
        T*By/By0*(rho/rho0)**beta*np.sin(phi)
    )
    return Zs


if __name__ == "__main__":
    pass