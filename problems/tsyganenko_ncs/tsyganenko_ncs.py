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
RH0, RH0_rms = 11.02, 0.05
RH1, RH1_rms = 6.05, 0.88
RH2, RH2_rms = 0.84, 0.09
RH3, RH3_rms = -2.28, 0.08
RH4, RH4_rms = -0.25, 0.37
RH5, RH5_rms = -0.96, 0.16
T0, T0_rms = 0.29, 0.02
T1, T1_rms = 0.18, 0.08
a00, a00_rms = 2.91, 0.02
a01, a01_rms = -0.16, 0.07
a02, a02_rms = 0.56, 0.03
a10, a10_rms = 1.89, 0.03
a11, a11_rms = 0.06, 0.04
a12, a12_rms = 0.49, 0.04
alpha0, alpha0_rms = 7.13, 0.06
alpha1, alpha1_rms = 4.87, 0.07
alpha2, alpha2_rms = -0.22, 0.12
alpha3, alpha3_rms = -0.14, 0.04
chi, chi_rms = -0.29, 0.03  # 
beta0, beta0_rms = 2.18, 0.09
beta1, beta1_rms = 0.40, 0.11

# Pressure scale (nPa)
Pmean = 2.0  # P6C1L3

# Magnetic field scale (nT)
By0 = 5.0  # P5C2L6
Bz0 = 5.0  # P6C1L3

# Radius scale (Earth radii)
rho0 = 10.0  # P5C2L7


def RH_empirical(fP, fBz, phi):
    """Equation 4"""
    return RH0 + RH1*fP + RH2*fBz + (RH3 + RH4*fP + RH5*fBz)*np.cos(phi)


def T_empirical(fP):
    """Equation 5"""
    return T0 + T1*fP


def a0_empirical(fP, fBz):
    """Equation 6"""
    return a00 + a01*fP + a02*fBz


def a1_empirical(fP, fBz, phi):
    """Equation 7"""
    return a10 + a11*fP + a12*fBz


def alpha_empirical(fP, fBz, phi):
    """Equation 8"""
    return alpha0 + alpha1*np.cos(phi) + alpha2*fP + alpha3*fBz


def beta_empirical(fBz,):
    """Equation 9"""
    return beta0 + beta1*fBz


def fP_empirical(P):
    """Equation 10"""
    return (P/Pmean)**chi - 1


def fBz_empirical(Bz):
    """Equation 10"""
    return Bz/Bz0


def Zs_empirical(P, By, Bz, psi, rho, phi):
    """Equation 3"""
    fP = fP_empirical(P)
    fBz = fBz_empirical(Bz)
    a0 = a0_empirical(fP, fBz)
    a1 = a1_empirical(fP, fBz, phi)
    RH = RH_empirical(fP, fBz, phi)
    T = T_empirical(fP)
    alpha = alpha_empirical(fP, fBz, phi)
    beta = beta_empirical(fBz)
    Zs = (
        RH*np.tan(psi) *
        (1 - (1 + (rho/RH)**alpha)**(1/alpha)) *
        (a0 + a1*np.cos(phi)) +
        T*By/By0*(rho/rho0)**beta*np.sin(phi)
    )
    return Zs


if __name__ == "__main__":
    pass