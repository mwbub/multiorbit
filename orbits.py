"""
Module containing the Orbits class, used to hold multiple orbits and perform
orbit integration using parallel processing.
"""
import astropy.units as u
from astropy.io import fits
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
                8) None: assumed to be the Sun (equivalent to vxvv=[0, 0, 0, 0,
                   0, 0] and radec=True).
                All elements must be of the same form.
            radec (bool; optional): If True, input is 4) or 5) above.
            uvw (bool, optional): If True, velocities are UVW.
            lb (bool; optional): If True, input is 6) or 7) above (note that
                this turns on physical output even if ro and vo are not given).
            ro (float; optional): Distance from the vantage point to the GC
                (kpc; can be Quantity).
            vo (float; optional): Circular velocity at ro (km/s; can be
                Quantity).
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

    @classmethod
    def integrate_chunks(cls, t, pot, filename, vxvv=None, radec=False,
                         uvw=False, lb=False, ro=None, vo=None, zo=None,
                         solarmotion=None, method='symplec4_c', dt=None,
                         numcores=1, chunk_size=1000000, save_all=False):
        """Create and integrate Orbits in a series of chunks.

        This method is designed to integrate a large number of orbits in a
        series of chunks. Due to the memory requirements of each Orbit instance,
        this method creates and integrates all orbits internally, and saves
        the results in a FITS file. As such, it accepts the parameters of both
        the Orbits.integrate method and the Orbits initialization keywords.

        Args:
            t: List of times at which to output, including 0; can be Quantity.
            pot: Potential instance of list of instances.
            filename: File in which to save the results of the integration.
                The file will be saved in the FITS format. Note that the file
                name will be modified by appending a '_0', '_1', '_2', etc.
                in order to number each time step.
            vxvv: List of initial conditions. Elements can be either:
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
                8) None: assumed to be the Sun (equivalent to vxvv=[0, 0, 0, 0,
                   0, 0] and radec=True).
                All elements must be of the same form.

        Optional Args:
            radec: If True, vxvv is 4) or 5) above.
            uvw: If True, velocities are UVW.
            lb: If True, vxvv is 6) or 7) above (note that this turns on
                physical output even if ro and vo are not given).
            ro: Distance from the vantage point to the GC (kpc; can be
                Quantity).
            vo: Circular velocity at ro (km/s; can be Quantity).
            zo: Offset toward teh NGP of the Sun wrt the plane (kpc; can be
                Quantity; default = 25 pc).
            solarmotion: 'hogg', or 'dehnen', or 'schoenrich', or value in
                [-U, V, W]; can be Quantity.
            method: 'odeint' for scipy's odeint;
                'leapfrog' for a simple leapfrog implementation;
                'leapfrog_c' for a simple leapfrog implementation in C;
                'symplec6_c' for a 6th order symplectic integrator in C;
                'rk4_c' for a 4th-order Runge-Kutta integrator in C;
                'rk6_c' for a 6-th order Runge-Kutta integrator in C;
                'dopr54_c' for a Dormand-Prince integrator in C.
            dt: If set, force the integrator to use this basic step size; must
                be an integer divisor of output step size (only works for C
                integrators that use a fixed step size); can be Quantity.
            numcores: Number of cores to use for multiprocessing.
            chunk_size: Number of orbits to integrate per chunk;
                default = 1000000.
            save_all: If True, save all time steps. Otherwise, save only the
                final time step. Default = False

        Returns:
            None

        """
        if vxvv is None:
            vxvv = []

        # Remove the .fits file extension if it was provided
        if filename.lower()[-5:] == '.fits':
            filename = filename[:-5]

        # Save all time steps or just the last time step
        times = t if save_all else [t[-1]]

        # Generate empty FITS files for each time step
        for i in range(len(times)):
            file = filename + '_{}.fits'.format(i)
            hdu = fits.BinTableHDU.from_columns([
                fits.Column(name='R', format='D'),
                fits.Column(name='phi', format='D'),
                fits.Column(name='z', format='D'),
                fits.Column(name='vR', format='D'),
                fits.Column(name='vT', format='D'),
                fits.Column(name='vz', format='D'),
                fits.Column(name='t', format='D')])
            hdu.writeto(file, overwrite=True)

        # Iterate over chunks of orbits of size chunk_size
        for i in range(0, len(vxvv), chunk_size):
            # Generate and integrate the Orbits chunk
            chunk = cls(vxvv=vxvv[i:i+chunk_size], radec=radec, uvw=uvw, lb=lb,
                        ro=ro, vo=vo, zo=zo, solarmotion=solarmotion)
            chunk.integrate(t, pot, method=method, dt=dt, numcores=numcores)

            # Update the file for each time step
            for j in range(len(times)):
                # Generate new data columns for the current chunk
                time = times[j]
                nrows1 = len(chunk.orbits)
                if isinstance(time, u.Quantity):
                    time_col = [time.value]*nrows1
                else:
                    time_col = [time]*nrows1
                hdu = fits.BinTableHDU.from_columns([
                    fits.Column(name='R', format='D', array=chunk.R(time)),
                    fits.Column(name='phi', format='D', array=chunk.phi(time)),
                    fits.Column(name='z', format='D', array=chunk.z(time)),
                    fits.Column(name='vR', format='D', array=chunk.vR(time)),
                    fits.Column(name='vT', format='D', array=chunk.vT(time)),
                    fits.Column(name='vz', format='D', array=chunk.vz(time)),
                    fits.Column(name='t', format='D', array=time_col)])

                # Append the new data columns to the FITS file
                file = filename + '_{}.fits'.format(j)
                with fits.open(file, mode='update') as hdul:
                    nrows2 = hdul[1].data.shape[0]
                    nrows = nrows1 + nrows2
                    hdul[1] = fits.BinTableHDU.from_columns(
                        hdul[1].columns, nrows=nrows)
                    for colname in hdul[1].columns.names:
                        hdul[1].data[colname][nrows2:] = hdu.data[colname]
                    hdul.flush()
