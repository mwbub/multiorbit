import os
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
from galpy.orbit import Orbits

try:
    _NUMCORES = int(os.environ['OMP_NUM_THREADS'])
except KeyError:
    import multiprocessing
    _NUMCORES = multiprocessing.cpu_count()


def integrate_chunks(t, pot, filename, vxvv=None, ro=None, vo=None, zo=None,
                     solarmotion=None, method='symplec4_c', dt=None,
                     numcores=_NUMCORES, force_map=False, chunk_size=1000000,
                     save_all=False):
    """Create and integrate Orbits in a series of chunks.

    This function is designed to integrate a large number of orbits in a series
    of chunks. Due to the memory requirements of each Orbit instance, this
    function creates and integrates all orbits internally, and saves the results
    in a .fits file. As such, it accepts the parameters of both the
    Orbits.integrate method and the Orbits initialization keywords.

    Parameters
    ----------
    t
        Array of times at which to output, including 0; can be Quantity.
    pot
        Potential instance or list of instances.
    filename
        File in which to save the results of the integration. The file will be
        saved in the .fits format. Note that the file name will be modified by
        appending a '_0', '_1', '_2', etc. in order to number each time step.
    vxvv
        Array or list of initial conditions in Galactocentric cylindrical
        coordinates. The array must have shape (N, 6), with elements arranged as
        [R, vR, vT, z, vz, phi]. Can also be a list of Orbit instances with
        6 dimensional phase-space coordinates.
    ro
        Distance from vantage point to GC (kpc; can be Quantity).
    vo
        Circular velocity at ro (km/s; can be Quantity).
    zo
        Offset toward the NGP of the Sun wrt the plane (kpc; can be Quantity;
        default = 25 pc).
    solarmotion
        'hogg' or 'dehnen', or 'schoenrich', or value in [-U,V,W]; can be
        Quantity.
    method
        'odeint' for scipy's odeint;
        'leapfrog' for a simple leapfrog implementation;
        'leapfrog_c' for a simple leapfrog implementation in C;
        'symplec4_c' for a 4th order symplectic integrator in C;
        'symplec6_c' for a 6th order symplectic integrator in C;
        'rk4_c' for a 4th-order Runge-Kutta integrator in C;
        'rk6_c' for a 6-th order Runge-Kutta integrator in C;
        'dopr54_c' for a 5-4 Dormand-Prince integrator in C;
        'dopr853_c' for a 8-5-3 Dormand-Prince integrator in C.
    dt
        If set, force the integrator to use this basic step size; must be an
        integer divisor of output step size (only works for C integrators that
        use a fixed step size); can be Quantity.
    numcores
        Number of cores to use for multiprocessing with force_map.
    force_map
        Use Python multiprocessing to integrate the orbits, rather than OpenMP;
        default = False.
    chunk_size
        Number of orbits to integrate per chunk; default = 1000000.
    save_all
        If True, save all time steps. Otherwise, save only the final time step;
        default = False.

    Returns
    -------
    None
    """
    if vxvv is None:
        vxvv = [None]

    # Remove the .fits file extension if it was provided
    if filename.lower()[-5:] == '.fits':
        filename = filename[:-5]

    # Save all time steps or just the last time step
    times = t if save_all else [t[-1]]

    # Check for an existing restart file
    directory = os.path.dirname(filename)
    if os.path.exists(directory + '/restart.txt'):
        with open(directory + '/restart.txt', 'r') as f:
            start = int(f.readline())
    else:
        start = 0

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
                fits.Column(name='t', format='D')
            ])
            hdu.writeto(file)

        # Generate a restart file
        with open(directory + '/restart.txt', 'w') as f:
            f.write('0\n')

    # Iterate over chunks of orbits of size chunk_size
    for i in range(start, len(vxvv), chunk_size):

        # Generate and integrate the Orbits chunk
        chunk = Orbits(vxvv=vxvv[i:i+chunk_size], ro=ro, vo=vo, zo=zo,
                       solarmotion=solarmotion)
        chunk.integrate(t, pot, method=method, dt=dt, numcores=numcores,
                        force_map=force_map)

        # Update the file for each time step
        for j in range(len(times)):

            # Generate new data columns for the current chunk
            nrows = len(vxvv[i:i+chunk_size])
            time = times[j]
            time_val = time if not isinstance(time, u.Quantity) else time.value
            time_col = np.full(nrows, time_val)
            R, vR, vT, z, vz, phi = chunk._call_internal(time)
            hdu = fits.BinTableHDU.from_columns([
                fits.Column(name='R', format='D', array=R),
                fits.Column(name='phi', format='D', array=phi),
                fits.Column(name='z', format='D', array=z),
                fits.Column(name='vR', format='D', array=vR),
                fits.Column(name='vT', format='D', array=vT),
                fits.Column(name='vz', format='D', array=vz),
                fits.Column(name='t', format='D', array=time_col)
            ])

            # Append the new data columns to the FITS file
            file = filename + '_{}.fits'.format(j)
            with fits.open(file, mode='update') as hdul:
                hdul[1] = fits.BinTableHDU.from_columns(
                    hdul[1].columns, nrows=i+nrows
                )
                for colname in hdul[1].columns.names:
                    hdul[1].data[colname][i:] = hdu.data[colname]
                hdul.flush()

            # Delete the hdu's to save memory
            del hdu, hdul

        # Update the restart file
        with open(directory + '/restart.txt', 'w') as f:
            f.write(str(i+chunk_size) + '\n')


def integrate_batch(t, pot, initfile, method='symplec4_c', dt=None,
                    numcores=_NUMCORES, force_map=False, chunk_size=1000000,
                    save_all=False):
    """Integrate a specified batch of orbits from an initial conditions file.

    This function is designed to read orbit initial conditions from a .fits
    file, divide the initial conditions into a series of batches, and integrate
    one of the specified batches. It uses the integrate_chunks method to
    integrate and save the desired orbits.

    Three command line arguments must be passed to any script which calls this
    function. These are:

    1) An integer representing the number of the batch to integrate.
    2) An integer representing the total number of batches into which the
       initial conditions are divided.
    3) A string representing the name of the directory where the results of the
       integration will be saved.

    The results of the integration of the specified batch will then be saved
    in the indicated directory.

    Parameters
    ----------
    t
        Array of times at which to output, including 0; can be Quantity.
    pot
        Potential instance or list of instances.
    initfile
        The initial conditions file. The file should be in the .fits format,
        and should contain columns named 'R', 'vR', 'vT', 'z', 'vz', and 'phi',
        corresponding to Galactocentric cylindrical coordinates.
    method
        'odeint' for scipy's odeint;
        'leapfrog' for a simple leapfrog implementation;
        'leapfrog_c' for a simple leapfrog implementation in C;
        'symplec4_c' for a 4th order symplectic integrator in C;
        'symplec6_c' for a 6th order symplectic integrator in C;
        'rk4_c' for a 4th-order Runge-Kutta integrator in C;
        'rk6_c' for a 6-th order Runge-Kutta integrator in C;
        'dopr54_c' for a 5-4 Dormand-Prince integrator in C;
        'dopr853_c' for a 8-5-3 Dormand-Prince integrator in C.
    dt
        If set, force the integrator to use this basic step size; must be an
        integer divisor of output step size (only works for C integrators that
        use a fixed step size); can be Quantity.
    numcores
        Number of cores to use for multiprocessing with force_map.
    force_map
        Use Python multiprocessing to integrate the orbits, rather than OpenMP;
        default = False.
    chunk_size
        Number of orbits to integrate per chunk; default = 1000000.
    save_all
        If True, save all time steps. Otherwise, save only the final time step;
        default = False.

    Returns
    -------
    None
    """
