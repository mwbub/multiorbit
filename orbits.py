"""
Module containing the Orbits class, used to hold multiple orbits and perform
orbit integration using parallel processing.
"""
from types import MethodType
from galpy.orbit import Orbit
from galpy.util.multi import parallel_map


class Orbits:
    """A class used to contain a sequence of Orbit objects.

    Can be used to integrate orbits using parallel processing.
    """

    def __init__(self, vxvv=None, radec=False, uvw=False, lb=False, ro=None,
                 vo=None, zo=None, solarmotion=None):
        """Initialize an Orbits instance.
        
        Args:
            vxvv (list): List of initial conditions. Elements can be either:
                1) Orbit instance.
                2) In galactocentric cylindrical coordinates [R, vR, vT, (z, vz,
                   phi)]; can be Quantities.
                3) Astropy (>v3.0) SkyCoord that includes velocities (this turns
                   on physical output even if ro and vo are not given).
                4) [ra, dec, d, mu_ra, mu_dec, vlos] in [deg, deg, kpc, mas/yr,
                   mas/yr, km/s] (all J2000.0; mu_ra = mu_ra * cos dec); can be 
                   Quantities; ICRS frame.
                5) [ra, dec, d, U, V, W] in [deg, deg, kpc, km/s, km/s, kms];
                   can be Quantities; ICRS frame.
                6) [l, b, d, mu_l, mu_b, vlos] in [deg, deg, kpc, mas/yr,
                   mas/yr, km/s) (all J2000.0; mu_l = mu_l * cos b); can be 
                   Quantities.
                7) [l, b, d, U, V, W] in [deg,deg,kpc,km/s,km/s,kms]; can be
                   Quantities.
                8) Unspecified: assumed to be the Sun (equivalent to vxvv=[0, 0,
                   0, 0, 0, 0] and radec=True).
                All elements must be of the same form.
            radec (bool; optional): If True, input is 4) or 5) above.
            uvw (bool, optional): If True, velocities are UVW.
            lb (bool; optional): If True, input is 6) or 7) above (note that
                this turns on physical output even if ro and vo are not given).
            ro (float): Distance from the vantage point to the GC (kpc; can be
                Quantity).
            vo (float): Circular velocity at ro (km/s; can be Quantity).
            zo (float; optional): Offset toward teh NGP of the Sun wrt the plane
                (kpc; can be Quantity; default = 25 pc).
            solarmotion (optional): 'hogg', or 'dehnen', or 'schoenrich', or
                value in [-U, V, W]; can be Quantity.

        """
        if vxvv is None:
            vxvv = []

        self.orbits = []
        for initial_value in vxvv:
            if isinstance(initial_value, Orbit):
                self.orbits.append(initial_value)
            else:
                orbit = Orbit(vxvv=initial_value, radec=radec, uvw=uvw, lb=lb,
                              ro=ro, vo=vo, zo=zo, solarmotion=solarmotion)
                self.orbits.append(orbit)

    def __getattr__(self, name):
        """Get the value of an attribute.

        Args:
            name: Attribute to evaluate.

        Returns:
            Value of name for each orbit in this Orbits instance.

        """
        attribute = getattr(Orbit(), name)
        if isinstance(attribute, MethodType):
            setattr(Orbits, name, lambda inner_self, *args, **kwargs: [
                getattr(o, name)(*args, **kwargs) for o in inner_self.orbits
            ])
            return getattr(self, name)
        else:
            return [getattr(orbit, name) for orbit in self.orbits]

    def integrate(self, t, pot, method='symplec4_c', dt=None, numcores=1):
        """Integrate the Orbit instances in this Orbits instance.

        Args:
            t: List of times at which to output, including 0; can be Quantity.
            pot: Potential instance of list of instances.
            method (optional): 'odeint' for scipy's odeint;
                'leapfrog' for a simple leapfrog implementation;
                'leapfrog_c' for a simple leapfrog implementation in C;
                'symplec6_c' for a 6th order symplectic integrator in C;
                'rk4_c' for a 4th-order Runge-Kutta integrator in C;
                'rk6_c' for a 6-th order Runge-Kutta integrator in C;
                'dopr54_c' for a Dormand-Prince integrator in C.
            dt (optional): If set, force the integrator to use this basic
                step size; must be an integer divisor of output step size (only
                works for C integrators that use a fixed step size); can be
                Quantity.
            numcores (optional): Number of cores to use for multiprocessing.

        Returns:
            None

        """
        # Must return each Orbit for its values to correctly update
        def integrate(orbit):
            orbit.integrate(t, pot, method=method, dt=dt)
            return orbit

        self.orbits = list(parallel_map(integrate, self.orbits,
                                        numcores=numcores))
