#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script takes as input a set of measurement sets at various frequencies and
(possibly) multiple fields and performs directional calibration that can be
used to correct for time- and position-dependent ionospheric effects. All
solutions are copied to a single H5parm file that can be used with LoSoTo to
fit for TEC values and to derive phase screens that may be used to make
ionospheric corrections with the AWimager.

Command-line arguments define the position and radius over which to search for
potential calibrators, the minimum apparent flux for a calibrator, the maximum
size of a calibrator, etc. From these inputs, the script can determine the
optimal set of calibrators to solve for in each band and in each field.  At a
given frequency, only the field in which the calibrator is brightest is used.

All the measurement sets are assumed to be a single directory (defined by the
'indir' option). All results are saved in an output directory (defined by the
'outdir' option).

Direction-independent selfcal solutions must have been applied to the
CORRECTED_DATA column of all MS files before running this script.

"""
import logging
import os
import commands
import subprocess
import glob
import shutil
import sys
import glob
import numpy as np
import pyrap.tables as pt
import scipy.signal
import lofar.parameterset
import lofar.parmdb
import lofar.expion.parmdbmain
import multiprocessing
import multiprocessing.pool
import lsmtool
from numpy import sum, sqrt, min, max, any
from numpy import argmax, argmin, mean, abs
from numpy import int32 as Nint
from numpy import float32 as Nfloat
import copy
try:
    from jug import Task
    has_jug = True
except ImportError:
    has_jug = False
_version = '1.0'


def init_logger(logfilename, debug=False):
    if debug:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    # Remove any existing handlers
    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[0])

    # File handler
    fh = logging.FileHandler(logfilename, 'a')
    fmt = MultiLineFormatter('%(asctime)s:: %(name)-6s:: %(levelname)-8s: '
        '%(message)s', datefmt='%a %d-%m-%Y %H:%M:%S')
    fh.setFormatter(fmt)
    logging.root.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    fmt = logging.Formatter('\033[31;1m%(levelname)s\033[0m: %(message)s')
    ch.setFormatter(fmt)
    logging.root.addHandler(ch)


class MultiLineFormatter(logging.Formatter):
    """Simple logging formatter that splits a string over multiple lines"""
    def format(self, record):
        str = logging.Formatter.format(self, record)
        header, footer = str.split(record.message)
        str = str.replace('\n', '\n' + ' '*len(header))
        return str


# The following multiprocessing classes allow us to run multiple bands
# in parallel with multiple time-correlated calibrations.
class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def create_peeling_skymodel(MS, gsm=None, radius=10, flux_cutoff_Jy=0.1, outdir='.',
        master_skymodel=None, use_patches=False):
    """Creates a sky model for peeling with BBS using input gsm or gsm.py

    MS - measurement set to create sky model for
    gsm - global sky model
    radius - radius in deg
    flux_cutoff_Jy - cutoff flux in Jy
    outdir - output directory
    master_skymodel - sky model with patch definitions
    """
    ms = MS.split("/")[-1]
    outfile = "{0}/skymodels/{1}.peeling.skymodel".format(outdir, ms)
    if os.path.exists(outfile):
        return

    if gsm is not None:
        s = lsmtool.load(gsm)
    else:
        obs = pt.table(MS + '/FIELD', ack=False)
        ra = np.degrees(float(obs.col('REFERENCE_DIR')[0][0][0]))
        if ra < 0.:
            ra = 360. + (ra)
        dec = np.degrees(float(obs.col('REFERENCE_DIR')[0][0][1]))
        obs.close()

        subprocess.call("gsm.py {0} {1} {2} {3} {4} 2>/dev/null".format(outfile,
            ra, dec, radius, flux_cutoff_Jy), shell=True)
        s = lsmtool.load(outfile)

    if use_patches:
        # Use master_skymodel to define patches in band's sky model. If a source
        # falls within 2 arcmin of a source in the master_skymodel, then the
        # patch is transferred.
        s.transfer(master_skymodel, matchBy='position', radius='2 arcmin')
        s.setPatchPositions(method='mid')

    s.write(outfile, clobber=True)


def scan_directory(indir, outdir='.'):
    """Scans the directory and returns list of Field objects

    The search assumes measurement sets all end with '.MS'
    """
    ms_list = sorted(glob.glob(indir+"/*.MS"))
    if len(ms_list) == 0:
        ms_list = sorted(glob.glob(indir+"/*.ms"))

    # Create a Band object for each MS
    band_list = []
    for ms in ms_list:
        band_list.append(Band(ms, outdir))

    # Sort the Band objects into Fields: two Bands belong to the same field if
    # their phase centers are within 10 arcmin of each other.
    field_list = []
    field_idx = 0
    for band in band_list:
        field_exists = False
        for field in field_list:
            # Check whether band pointing is within 10 arcmin of existing field
            # center
            if approx_equal(band.ra*np.cos(band.dec), field.ra*np.cos(band.dec),
                tol=0.167) and approx_equal(band.dec, field.dec, tol=0.167):
                field.bands.append(band)
                field_exists = True
        if not field_exists:
            name = 'Field{0}'.format(field_idx)
            field_list.append(Field(name, band.ra, band.dec, [band]))
            field_idx += 1

    # Sort the Band objects into bands: two Bands belong to the same band if
    # their frequencies are with 1 MHz of each other.
    band_names = []
    for band in band_list:
        band_name_exists = False
        for band_name in band_names:
            # Check whether band frequency is within 1 MHz of existing band
            # frequency
            if approx_equal(band.freq, float(band_name), tol=1e6):
                band.name = band_name
                band_name_exists = True
        if not band_name_exists:
            band_name = str(band.freq)
            band.name = band_name
            band_names.append(band_name)

    # Add field names to Band objects
    for field in field_list:
        for band in field.bands:
            band.field_name = field.name

    return field_list


def _float_approx_equal(x, y, tol=1e-18, rel=1e-7):
    if tol is rel is None:
        raise TypeError('cannot specify both absolute and relative errors are None')
    tests = []
    if tol is not None: tests.append(tol)
    if rel is not None: tests.append(rel*abs(x))
    assert tests
    return abs(x - y) <= max(tests)


def approx_equal(x, y, *args, **kwargs):
    """approx_equal(float1, float2[, tol=1e-18, rel=1e-7]) -> True|False
    approx_equal(obj1, obj2[, *args, **kwargs]) -> True|False

    Return True if x and y are approximately equal, otherwise False.

    If x and y are floats, return True if y is within either absolute error
    tol or relative error rel of x. You can disable either the absolute or
    relative check by passing None as tol or rel (but not both).

    For any other objects, x and y are checked in that order for a method
    __approx_equal__, and the result of that is returned as a bool. Any
    optional arguments are passed to the __approx_equal__ method.

    __approx_equal__ can return NotImplemented to signal that it doesn't know
    how to perform that specific comparison, in which case the other object is
    checked instead. If neither object have the method, or both defer by
    returning NotImplemented, approx_equal falls back on the same numeric
    comparison used for floats.

    >>> almost_equal(1.2345678, 1.2345677)
    True
    >>> almost_equal(1.234, 1.235)
    False

    """
    if not (type(x) is type(y) is float):
        # Skip checking for __approx_equal__ in the common case of two floats.
        methodname = '__approx_equal__'
        # Allow the objects to specify what they consider "approximately equal",
        # giving precedence to x. If either object has the appropriate method, we
        # pass on any optional arguments untouched.
        for a,b in ((x, y), (y, x)):
            try:
                method = getattr(a, methodname)
            except AttributeError:
                continue
            else:
                result = method(b, *args, **kwargs)
                if result is NotImplemented:
                    continue
                return bool(result)
    # If we get here without returning, then neither x nor y knows how to do an
    # approximate equal comparison (or are both floats). Fall back to a numeric
    # comparison.
    return _float_approx_equal(x, y, *args, **kwargs)


def find_calibrators(master_skymodel, beamMS, flux_cut_Jy=15.0,
    maj_cut_arcsec=None, plot=False, applyBeam=True, band_skymodel=None):
    """Returns list of optimal set of GSM calibrators for peeling

    Calibrators are determined as compact sources above the given apparent flux
    cutoff in Jy and are selected from the master list defined by sources in
    master_skymodel.
    """
    log.info('Checking {0}:'.format(beamMS))

    if band_skymodel is None or band_skymodel == master_skymodel:
        s = lsmtool.load(master_skymodel, beamMS=beamMS)
    else:
        s = lsmtool.load(band_skymodel, beamMS=beamMS)
        s.transfer(master_skymodel, matchBy='position', radius='2 arcmin')
        s.setPatchPositions(method='mid')

    if maj_cut_arcsec is not None:
        log.info('Filtering out sources larger than {0} arcsec:'.format(maj_cut_arcsec))
        if s.hasPatches:
            sizes = s.getPatchSizes(units='arcsec', weight=True, applyBeam=applyBeam)
        else:
            sizes = s.getColValues('MajorAxis', units='arcsec')
        s.select(sizes <= maj_cut_arcsec, force=True, aggregate=True)
        if len(s) == 0:
            return [], [], []

    # Make sure all fluxes are at 60 MHz (if possible)
    if 'ReferenceFrequency' in s.getColNames():
        reffreqs = s.getColValues('ReferenceFrequency')
        fluxes =  s.getColValues('I')
        try:
            alphas = s.getColValues('SpectralIndex')[:, 0] # just use slope
        except IndexError:
            alphas = -0.8
        fluxes_60 = fluxes*(60e6/reffreqs)**alphas
        s.setColValues('I', fluxes_60)
        s.setColValues('ReferenceFrequency', np.array([60e6]*len(reffreqs)))

    # Now select only those sources above the given apparent flux cut
    log.info('Filtering out sources with apparent fluxes at obs. midpoint below {0} Jy:'.format(flux_cut_Jy))
    s.select(['I', '>', flux_cut_Jy, 'Jy'], applyBeam=applyBeam, aggregate='sum',
        force=True)

    if len(s) > 0:
        if plot:
            print('Showing potential calibrators. Close the plot window to continue.')
            s.plot()
        cal_fluxes = s.getColValues('I', aggregate='sum', applyBeam=applyBeam).tolist()
        if s.hasPatches:
            cal_names = s.getPatchNames().tolist()
            cal_sizes = s.getPatchSizes(units='arcsec', weight=True,
                applyBeam=applyBeam).tolist()
        else:
            cal_names = s.getColValues('Name').tolist()
            cal_sizes = s.getColValues('MajorAxis', units='arcsec').tolist()
        return cal_names, cal_fluxes, cal_sizes, s.hasPatches
    else:
        return [], [], [], False


def setup_peeling(band):
    """Sets up peeling parsets and GSM sky models"""
    log = logging.getLogger("Peeler")

    if len(band.cal_names) == 0:
        log.info('No calibrators found for {0}. No peeling done.'.format(band.file))
        band.do_peeling = False
        return

    # Split calibrator list into flux bins of nsrc_per_bin or fewer sources
    nbins = int(np.ceil(len(band.cal_names)/band.nsrc_per_bin))
    sorted_cal_ind = np.argsort(band.cal_apparent_fluxes)[::-1]
    list_of_flux_bins = np.array_split(np.array(band.cal_apparent_fluxes)[sorted_cal_ind], nbins)
    bin_ends = []
    for f, flux_bin in enumerate(list_of_flux_bins):
        if f == 0:
            bin_ends.append(len(flux_bin))
        elif f < nbins-1:
            bin_ends.append(bin_ends[f-1]+len(flux_bin))
        else:
            continue
    list_of_name_bins = np.array_split(np.array(band.cal_names)[sorted_cal_ind], bin_ends)
    list_of_size_bins = np.array_split(np.array(band.cal_sizes)[sorted_cal_ind], bin_ends)

    peel_bins = []
    mean_flux1 = None
    solint_min = band.solint_min
    for names, fluxes, sizes in zip(list_of_name_bins, list_of_flux_bins, list_of_size_bins):
        mean_flux = np.mean(fluxes)
        if band.use_timecorr or not band.scale_solint:
            sol_int = solint_min
        else:
            if mean_flux1 is not None:
                # Scale by flux_ratio^2
                sol_int = min([int(np.ceil(solint_min * (mean_flux1 / mean_flux)**2)), 5*solint_min])
            else:
                mean_flux1 = mean_flux
                sol_int = solint_min
            if sol_int < 1:
                sol_int = 1
        bin_dict = {'names': names.tolist(), 'sol_int': sol_int, 'fluxes': fluxes, 'sizes': sizes}
        peel_bins.append(bin_dict)

    if band.use_timecorr:
        logtxt = 'Calibrator sets for {0}:\n      Set 1 = {1}; app. flux = {2}; size = {3} arcmin'.format(
            band.file, peel_bins[0]['names'], peel_bins[0]['fluxes'], peel_bins[0]['sizes']/60.0)
        if len(peel_bins) > 1:
            for p, peel_bin in enumerate(peel_bins[1:]):
                logtxt += '\n      Set {0} = {1}; app. flux = {2}; size = {3} arcmin'.format(p+2,
                    peel_bin['names'], peel_bin['fluxes'], peel_bin['sizes']/60.0)
    else:
        logtxt = 'Calibrator sets for {0}:\n      Set 1 = {1}; solint = {2} time slots; app. flux = {3}; size = {4} arcmin'.format(
            band.file, peel_bins[0]['names'], peel_bins[0]['sol_int'], peel_bins[0]['fluxes'], peel_bins[0]['sizes']/60.0)
        if len(peel_bins) > 1:
            for p, peel_bin in enumerate(peel_bins[1:]):
                logtxt += '\n      Set {0} = {1}; solint = {2} time slots; app. flux = {3}; size = {4} arcmin'.format(p+2,
                    peel_bin['names'], peel_bin['sol_int'], peel_bin['fluxes'], peel_bin['sizes']/60.0)
    log.info(logtxt)

    msname = band.file.split('/')[-1]
    make_peeling_parset('{0}/parsets/{1}.peeling.parset'.format(band.outdir,
        msname), peel_bins, scalar_phase=band.use_scalar_phase,
        phase_only=band.phase_only, sol_int_amp=band.solint_amp,
        beam_mode=band.beam_mode, uvmin=band.uvmin)

    create_peeling_skymodel(band.file, band.skymodel, radius=band.fwhm_deg*1.5,
        flux_cutoff_Jy=0.1, outdir=band.outdir, master_skymodel=band.master_skymodel,
        use_patches=band.use_patches)
    band.do_peeling = True
    band.peel_bins = peel_bins


def peel_band(band):
    """Performs peeling on a band using BBS"""
    log = logging.getLogger("Peeler")

    # Check if peeling is required
    if not band.do_peeling:
        return

    # Define file names
    msname = band.msname
    skymodel =  "{0}/skymodels/{1}.peeling.skymodel".format(band.outdir, msname)
    p_shiftname = "{0}/parsets/peeling_shift_{1}.parset".format(band.outdir,
        msname)
    peelparset = "{0}/parsets/{1}.peeling.parset".format(band.outdir,
        msname)

    # Make a copy of the MS for peeling and average if desired
    newmsname = "{0}/{1}.peeled".format(band.outdir, msname)
    log.info('Performing averaging and peeling for {0}...\n'
        '      Phase-only calibration: {4}\n'
        '      See the following logs for details:\n'
        '      - {3}/logs/ndppp_avg_{1}.log\n'
        '      - {3}/logs/{2}_peeling_calibrate.log'.format(band.file, msname,
        msname, band.outdir, band.phase_only))
    f = open(p_shiftname, 'w')
    f.write("msin={0}\n"
        "msin.datacolumn=CORRECTED_DATA\n"
        "msout={1}\n"
        "msin.startchan = 0\n"
        "msin.nchan = 0\n"
        "steps = [avg]\n"
        "avg.type = average\n"
        "avg.freqstep = {2}".format(band.file, newmsname, band.navg))
    f.close()
    subprocess.call("NDPPP {0} > {1}/logs/ndppp_avg_{2}.log 2>&1".format(p_shiftname,
        band.outdir, msname), shell=True)

    # Perform dir-independent calibration if desired
    if band.do_dirindep:
        dirindep_parset = '{0}/parsets/{1}.dirindep.parset'.format(
            band.outdir, msname)
        make_dirindep_parset(dirindep_parset, scalar_phase=band.use_scalar_phase,
            sol_int=band.solint_min, beam_mode=band.beam_mode, uvmin=band.uvmin)
        subprocess.call("calibrate-stand-alone -f {0} {1} {2} > {3}/logs/"
            "{4}_dirindep_calibrate.log 2>&1".format(newmsname, dirindep_parset,
            skymodel, band.outdir, msname), shell=True)

    # Perform the peeling. Do this step even if time-correlated solutions
    # are desired so that the proper parmdb is made and so that the correlation
    # time can be estimated
    subprocess.call("calibrate-stand-alone -f {0} {1} {2} > {3}/logs/"
        "{4}_peeling_calibrate.log 2>&1".format(newmsname, peelparset,
        skymodel, band.outdir, msname), shell=True)

    if band.use_timecorr:
        # Do time-correlated peeling.
        peelparset_timecorr = '{0}/parsets/{1}.timecorr_peeling.parset'.format(
            band.outdir, msname)

        # Estimate ionfactor from the non-time-correlated peeling solutions
        # TODO: allow ionfactor to vary with time -- maybe a dictionary of
        # {'ionfactor': [0.5, 0.3, 0.2], 'start_sol_num': [0, 24, 256]}?
#        band.ionfactor = get_ionfactor(msname, instrument='instrument')

        # Make the parset and do the peeling
        make_peeling_parset(peelparset_timecorr, band.peel_bins,
            scalar_phase=band.use_scalar_phase, phase_only=True,
            time_block=band.time_block, beam_mode=band.beam_mode,
            uvmin=band.uvmin)
        calibrate(newmsname, peelparset_timecorr, skymodel, msname,
            use_timecorr=True, outdir=band.outdir, instrument='instrument',
            time_block=band.time_block, ionfactor=band.ionfactor,
            solint=band.solint_min, flag_filler=band.flag_filler,
            ncores=band.ncores_per_cal)


def make_dirindep_parset(parset, scalar_phase=True, sol_int=1,
    beam_mode='DEFAULT', uvmin=250.0):
    """Makes a BBS parset for dir-independent calibration

    Reads from and writes to DATA column
    """

    # Handle beam
    if beam_mode.lower() == 'off':
        beam_enable = 'F'
        beam_mode = 'DEFAULT'
    else:
        beam_enable = 'T'

    newlines = ['Strategy.InputColumn = DATA\n',
        'Strategy.ChunkSize = 100\n',
        'Strategy.Baselines = *&\n',
        'Strategy.UseSolver = F\n',
        'Strategy.Steps = [solve, correct]\n',
        '\n',
        'Step.solve.Operation = SOLVE\n',
        'Step.solve.Baselines = [CR]S*&\n',
        'Step.solve.Model.Sources = []\n',
        'Step.solve.Model.Cache.Enable = T\n',
        'Step.solve.Model.Phasors.Enable = T\n',
        'Step.solve.Model.Gain.Enable = T\n',
        'Step.solve.Model.CommonScalarPhase.Enable= T\n',
        'Step.solve.Model.Beam.Enable = {0}\n'.format(beam_enable),
        'Step.solve.Model.Beam.Mode = {0}\n'.format(beam_mode),
        'Step.solve.Model.Beam.UseChannelFreq = T\n',
        'Step.solve.Solve.Mode = COMPLEX\n',
        'Step.solve.Solve.UVRange = [{0}]\n'.format(uvmin)]
    if scalar_phase:
        newlines += ['Step.solve.Solve.Parms = ["CommonScalarPhase:*"]\n']
    else:
        newlines += ['Step.solve.Solve.Parms = ["Gain:0:0:Phase:*", "Gain:1:1:Phase:*"]\n']
    newlines += ['Step.solve.Solve.ExclParms = []\n',
        'Step.solve.Solve.CalibrationGroups = []\n',
        'Step.solve.Solve.CellSize.Freq = 0\n',
        'Step.solve.Solve.CellSize.Time = {0}\n'.format(int(sol_int)),
        'Step.solve.Solve.CellChunkSize = 10\n',
        'Step.solve.Solve.PropagateSolutions = T\n',
        'Step.solve.Solve.Options.MaxIter = 150\n',
        'Step.solve.Solve.Options.EpsValue = 1e-9\n',
        'Step.solve.Solve.Options.EpsDerivative = 1e-9\n',
        'Step.solve.Solve.Options.ColFactor = 1e-9\n',
        'Step.solve.Solve.Options.LMFactor = 1\n',
        'Step.solve.Solve.Options.BalancedEqs = F\n',
        'Step.solve.Solve.Options.UseSVD = T\n',
        '\n',
        'Step.correct.Operation = CORRECT\n',
        'Step.correct.Model.Sources = []\n',
        'Step.correct.Model.Phasors.Enable = T\n',
        'Step.correct.Model.CommonScalarPhase.Enable= T\n',
        'Step.correct.Model.Gain.Enable  = T\n',
        'Step.correct.Model.Beam.Enable  = F\n',
        'Step.correct.Output.Column = DATA\n']
    f = open(parset, 'w')
    f.writelines(newlines)
    f.close()


def make_peeling_parset(parset, peel_bins, scalar_phase=True, phase_only=True,
    sol_int_amp=500, time_block=None, beam_mode='DEFAULT', uvmin=250.0):
    """Makes a BBS parset for peeling

    For best results, the sources should be peeled in order of decreasing flux.
    If phase_only is False, amplitudes are also solved for.
    """
    if os.path.exists(parset):
        return
    sol_int_list = []
    if time_block is not None:
        # Set all chunk sizes to time_block
        for peel_bin in peel_bins:
            peel_bin['sol_int'] = time_block + 2
    for peel_bin in peel_bins:
        sol_int_list.append(peel_bin['sol_int'])
    if not phase_only:
        sol_int_list.append(sol_int_amp)
    sol_int_list.append(250) # don't let chunk size fall below 250 for performance reasons

    # Set overall strategy
    nbins = len(peel_bins)
    newlines = ['Strategy.InputColumn = DATA\n',
        'Strategy.ChunkSize = {0}\n'.format(int(max(sol_int_list))),
        'Strategy.Baselines = [CR]S*&\n',
        'Strategy.UseSolver = F\n']
    if phase_only:
        pstr = ''
        strategy_str = 'Strategy.Steps = [subtractfield'
        for i, peel_bin in enumerate(peel_bins):
            strategy_str += ', add{0}, solve{0}'.format(i+1)
            if i < nbins - 1:
                strategy_str += ', subtract{0}'.format(i+1)
        strategy_str += ']\n'
    else:
        pstr = 'p'
        strategy_str = 'Strategy.Steps = [subtractfield'
        for i, peel_bin in enumerate(peel_bins):
            strategy_str += ', add{0}, solvep{0}, solvea{0}'.format(i+1)
            if i < nbins - 1:
                strategy_str += ', subtract{0}'.format(i+1)
        strategy_str += ']\n'
    newlines += strategy_str

    # Handle beam
    if beam_mode.lower() == 'off':
        beam_enable = 'F'
        beam_mode = 'DEFAULT'
    else:
        beam_enable = 'T'

    # Subtract field (all sources)
    newlines += ['\n', 'Step.subtractfield.Operation = SUBTRACT\n',
        'Step.subtractfield.Model.Sources = []\n',
        'Step.subtractfield.Model.Beam.Enable = {0}\n'.format(beam_enable),
        'Step.subtractfield.Model.Beam.Mode = {0}\n'.format(beam_mode),
        '\n']

    for i, peel_bin in enumerate(peel_bins):
        # Add sources in current bin
        newlines += ['Step.add{0}.Operation = ADD\n'.format(i+1),
            'Step.add{0}.Model.Sources = '.format(i+1) + str(peel_bin['names']) + '\n',
            'Step.add{0}.Model.Beam.Enable = {1}\n'.format(i+1, beam_enable),
            'Step.add{0}.Model.Beam.Mode = {1}\n'.format(i+1, beam_mode),
            '\n']

        # Phase-only solve
        newlines += ['Step.solve{0}{1}.Operation = SOLVE\n'.format(pstr, i+1),
            'Step.solve{0}{1}.Model.Sources = '.format(pstr, i+1) + str(peel_bin['names']) + '\n',
            'Step.solve{0}{1}.Model.Cache.Enable = T\n'.format(pstr, i+1),
            'Step.solve{0}{1}.Model.Beam.Enable = {2}\n'.format(pstr, i+1, beam_enable),
            'Step.solve{0}{1}.Model.Beam.Mode = {2}\n'.format(pstr, i+1, beam_mode),
            'Step.solve{0}{1}.Model.Beam.UseChannelFreq = T\n'.format(pstr, i+1),
            'Step.solve{0}{1}.Model.DirectionalGain.Enable = T\n'.format(pstr, i+1),
            'Step.solve{0}{1}.Model.Phasors.Enable = T\n'.format(pstr, i+1)]
        if scalar_phase:
            newlines += ['Step.solve{0}{1}.Solve.Mode = COMPLEX\n'.format(pstr, i+1),
                'Step.solve{0}{1}.Model.ScalarPhase.Enable= T\n'.format(pstr, i+1),
                'Step.solve{0}{1}.Solve.Parms = ["ScalarPhase:*"]\n'.format(pstr, i+1)]
        else:
            newlines += ['Step.solve{0}{1}.Solve.Mode = COMPLEX\n'.format(pstr, i+1),
                'Step.solve{0}{1}.Solve.Parms = ["DirectionalGain:0:0:Phase:*",'
                '"DirectionalGain:1:1:Phase:*"]\n'.format(pstr, i+1)]
        newlines += ['Step.solve{0}{1}.Solve.CellSize.Freq = 0\n'.format(pstr, i+1),
            'Step.solve{0}{1}.Solve.CellSize.Time = {2}\n'.format(pstr, i+1, int(peel_bin['sol_int'])),
            'Step.solve{0}{1}.Solve.CellChunkSize = {2}\n'.format(pstr, i+1, int(peel_bin['sol_int'])),
            'Step.solve{0}{1}.Solve.Options.MaxIter = 150\n'.format(pstr, i+1),
            'Step.solve{0}{1}.Solve.Options.EpsValue = 1e-9\n'.format(pstr, i+1),
            'Step.solve{0}{1}.Solve.Options.EpsDerivative = 1e-9\n'.format(pstr, i+1),
            'Step.solve{0}{1}.Solve.Options.ColFactor = 1e-9\n'.format(pstr, i+1),
            'Step.solve{0}{1}.Solve.Options.LMFactor = 1.0\n'.format(pstr, i+1),
            'Step.solve{0}{1}.Solve.Options.BalancedEqs = F\n'.format(pstr, i+1),
            'Step.solve{0}{1}.Solve.Options.UseSVD = T\n'.format(pstr, i+1),
            'Step.solve{0}{1}.Solve.UVRange = [{2}]\n'.format(pstr, i+1, uvmin)]

        # Ampl-only solve
        if not phase_only and time_block is None:
            newlines += ['\n',
                'Step.solvea{0}.Operation = SOLVE\n'.format(i+1),
                'Step.solvea{0}.Model.Sources = '.format(i+1) + str(peel_bin['names']) + '\n',
                'Step.solvea{0}.Model.Cache.Enable = T\n'.format(i+1),
                'Step.solvea{0}.Model.Beam.Enable = {1}\n'.format(i+1, beam_enable),
                'Step.solvea{0}.Model.Beam.Mode = {1}\n'.format(i+1, beam_mode),
                'Step.solvea{0}.Model.Beam.UseChannelFreq = T\n'.format(i+1),
                'Step.solvea{0}.Model.Phasors.Enable = T\n'.format(i+1),
                'Step.solvea{0}.Solve.Mode = COMPLEX\n'.format(i+1),
                'Step.solvea{0}.Model.ScalarPhase.Enable= T\n'.format(i+1),
                'Step.solvea{0}.Model.DirectionalGain.Enable = T\n'.format(i+1),
                'Step.solvea{0}.Solve.Parms = ["DirectionalGain:0:0:Ampl:*",'
                '"DirectionalGain:1:1:Ampl:*"]\n'.format(i+1),
                'Step.solvea{0}.Solve.CellSize.Freq = 0\n'.format(i+1),
                'Step.solvea{0}.Solve.CellSize.Time = {1}\n'.format(i+1, int(sol_int_amp)),
                'Step.solvea{0}.Solve.CellChunkSize = {1}\n'.format(i+1, int(sol_int_amp)),
                'Step.solvea{0}.Solve.Options.MaxIter = 50\n'.format(i+1),
                'Step.solvea{0}.Solve.Options.EpsValue = 1e-9\n'.format(i+1),
                'Step.solvea{0}.Solve.Options.EpsDerivative = 1e-9\n'.format(i+1),
                'Step.solvea{0}.Solve.Options.ColFactor = 1e-9\n'.format(i+1),
                'Step.solvea{0}.Solve.Options.LMFactor = 1.0\n'.format(i+1),
                'Step.solvea{0}.Solve.Options.BalancedEqs = F\n'.format(i+1),
                'Step.solvea{0}.Solve.Options.UseSVD = T\n'.format(i+1),
                'Step.solvea{0}.Solve.UVRange = [{1}]\n'.format(i+1, uvmin)]

        # Subtract sources in current bin
        if i < nbins - 1:
            newlines += ['\n',
                'Step.subtract{0}.Operation = SUBTRACT\n'.format(i+1),
                'Step.subtract{0}.Model.Sources = '.format(i+1) + str(peel_bin['names']) + '\n',
                'Step.subtract{0}.Model.Beam.Enable = {1}\n'.format(i+1, beam_enable)]
            if scalar_phase:
                newlines += ['Step.subtract{0}.Model.ScalarPhase.Enable = T\n'.format(i+1)]
            if not scalar_phase or not phase_only:
                newlines += ['Step.subtract{0}.Model.DirectionalGain.Enable = T\n'.format(i+1)]
            newlines += ['Step.subtract{0}.Model.Beam.Mode = {1}\n'.format(i+1, beam_mode),
                '\n']

    f = open(parset, 'w')
    f.writelines(newlines)
    f.close()


def write_sols(field_list, h5file, flag_outliers=True):
    """Collects all solutions and writes them to an H5parm file"""
    log = logging.getLogger("Writer")
    log.info('Collecting all solutions in H5parm file {0}'.format(h5file))

    # Use LoSoTo's H5parm_importer.py tool to make H5parm files for each
    # MS/instrument_out parmdb. Then use H5parm_merge.py to merge them all.
    create_master = True
    source_str = ''
    if len(field_list) > 1:
        get_dirindep = True
    else:
        get_dirindep = False
    for field in field_list:
        for band in field.bands:
            if get_dirindep:
                # Collect direction-independent solutions from original MSes. These
                # are assumed to be in the MS/instrument parmdbs
                ms_file = band.file
                ms_h5file = '{0}.h5parm'.format(ms_file)
                ms_solset = '{0}Band{1}MHzDirIndep'.format(field.name, int(band.freq/1e6))
                subprocess.call("H5parm_importer.py {0} {1} --solset={2} -i instrument".format(ms_h5file,
                    ms_file, ms_solset), shell=True)

            # Collect direction-dependent solutions from peeled MSes
            ms_file = band.peeled_file
            peeled_ms_h5file = '{0}.h5parm'.format(ms_file)
            peeled_ms_solset = '{0}Band{1}MHzDirDep'.format(field.name, int(band.freq/1e6))
            if band.use_timecorr:
                instrumentdb = 'instrument_out'
            else:
                instrumentdb = 'instrument'
            subprocess.call("H5parm_importer.py {0} {1} --solset={2} -i {3}".format(peeled_ms_h5file,
                ms_file, peeled_ms_solset, instrumentdb), shell=True)

            # Run LoSoTo to flag outliers in the phases
            if flag_outliers:
                parset = makeFlagParset(peeled_ms_h5file, peeled_ms_solset)
                subprocess.call("losoto.py {0} {1}".format(peeled_ms_h5file,
                    parset), shell=True)

            # Use LoSoTo's h5parm_merge.py to merge the H5parm files into the
            # master H5parm file.
            if create_master:
                if get_dirindep:
                    subprocess.call("mv {0} {1}".format(ms_h5file, h5file), shell=True)
                    subprocess.call("H5parm_merge.py {0}:{1} {2}:{3}".format(peeled_ms_h5file,
                        peeled_ms_solset, h5file, peeled_ms_solset), shell=True)
                else:
                    subprocess.call("mv {0} {1}".format(peeled_ms_h5file, h5file), shell=True)
                create_master = False
            else:
                if get_dirindep:
                    subprocess.call("H5parm_merge.py {0}:{1} {2}:{3}".format(ms_h5file, ms_solset,
                        h5file, ms_solset), shell=True)
                subprocess.call("H5parm_merge.py {0}:{1} {2}:{3}".format(peeled_ms_h5file,
                    peeled_ms_solset, h5file, peeled_ms_solset), shell=True)


def makeFlagParset(h5file, solset):
    """Runs LoSoTo to flag outliers"""
    parset = '{0}.losoto_flag.parset'.format(h5file)
    newlines = [
        'LoSoTo.Steps = [flag]\n',
        'LoSoTo.Solset = [{0}]\n'.format(solset),
        'LoSoTo.Soltab = [sol000/phase000]\n',
        'LoSoTo.Steps.flag.Operation = FLAG\n',
        'LoSoTo.Steps.flag.Axis = time\n',
        'LoSoTo.Steps.flag.MaxCycles = 5\n',
        'LoSoTo.Steps.flag.MaxRms = 5.\n',
        'LoSoTo.Steps.flag.Window = 100\n',
        'LoSoTo.Steps.flag.Order = 2\n',
        'LoSoTo.Steps.flag.MaxGap = 300\n'
        'LoSoTo.Steps.flag.Replace = True\n']
    f = open(parset, 'w')
    f.writelines(newlines)
    f.close()
    return parset


def calibrate(msname, parset, skymodel, logname_root, use_timecorr=False,
    time_block=None, ionfactor=0.5, outdir='.', instrument='instrument',
    solint=None, flag_filler=False, ncores=1):
    """Calls BBS to calibrate with optional time-correlated fitting"""
    log = logging.getLogger("Calib")

    if not use_timecorr:
        subprocess.call("calibrate-stand-alone {0} {1} {2} > {3}/logs/"
            "{4}_peeling_calibrate.log 2>&1".format(msname, parset, skymodel,
            outdir, logname_root), shell=True)
        subprocess.call("cp -r {0}/instrument {0}/instrument_out".format(msname),
            shell=True)
    else:
        # Perform a time-correlated solve
        dataset = msname
        blockl = time_block
        anttab = pt.table(dataset + '/ANTENNA', ack=False)
        antlist = anttab.getcol('NAME')
        instrument_orig  = msname+'/instrument'
        instrument_out = msname+'/instrument_out'
        if solint < 1:
            solint = 1


        # Get time per sample and number of times
        t = pt.table(dataset, readonly=True, ack=False)
        for t2 in t.iter(["ANTENNA1","ANTENNA2"]):
            if (t2.getcell('ANTENNA1',0)) < (t2.getcell('ANTENNA2',0)):
                timepersample = t2[1]['TIME']-t2[0]['TIME'] # sec
                trows = t2.nrows()
        t.close()

        # Calculate various intervals
        fwhm_min, fwhm_max = modify_weights(msname, ionfactor, dryrun=True) # s
        if time_block is None:
            # Set blockl to enclose the max FWHM and be divisible by 2 and by solint
            blockl = int(np.ceil(fwhm_max / timepersample / 2.0 / solint) * 2 * solint)
        tdiff = solint * timepersample / 3600. # difference between solutions in hours
        tlen = timepersample * np.float(blockl) / 3600. # length of block in hours
        nsols = int(np.ceil(trows / solint)) # number of solutions

        log.info('Performing time-correlated peeling for {0}...\n'
            '      Time per sample: {1} (s)\n'
            '      Samples in total: {2}\n'
            '      Block size: {3} (samples)\n'
            '                  {4} (s)\n'
            '      Number of solutions: {5}\n'
            '      Ionfactor: {6}\n'
            '      FWHM range: {7} - {8} (s)'.format(msname, timepersample,
            trows, blockl, tlen*3600.0, nsols, ionfactor, fwhm_min, fwhm_max))

        # Update cellsize and chunk size of parset
        update_parset(parset)

        # Make a copy of the master parmdb to store time-correlated solutions
        # in, resetting and flagging as needed
        os.system('rm -rf ' +instrument_out)
        clean_and_copy_parmdb(instrument_orig, instrument_out, blockl,
            flag_filler=flag_filler, msname=msname, timepersample=timepersample)

        # Calibrate the chunks
        chunk_list = []
        tlen_mod = tlen / 2.0 # hours
        chunk_mid_start = blockl / 2 / solint
        chunk_mid_end = nsols - blockl / 2 / solint
        for c in range(nsols):
            chunk_obj = Chunk(dataset)
            chunk_obj.chunk = c
            chunk_obj.outdir = outdir
            if c < chunk_mid_start:
                chunk_obj.trim_start = True
                chunk_obj.t0 = 0.0 # hours
                chunk_obj.t1 = np.float(chunk_obj.t0) + tlen_mod # hours
                tlen_mod += tdiff # add one solution interval (in hours)
            elif c > chunk_mid_end:
                tlen_mod -= tdiff # subtract one solution interval (in hours)
                chunk_obj.trim_start = False
                chunk_obj.t0 = tdiff*float(chunk_obj.chunk - chunk_mid_start) # hours
                chunk_obj.t1 = np.float(chunk_obj.t0) + tlen_mod # hours
            else:
                chunk_obj.trim_start = False
                chunk_obj.t0 = tdiff*float(chunk_obj.chunk - chunk_mid_start) # hours
                chunk_obj.t1 = np.float(chunk_obj.t0) + tlen # hours
            chunk_obj.ionfactor = ionfactor
            chunk_obj.parset = parset
            chunk_obj.skymodel = skymodel
            chunk_obj.logname_root = logname_root + '_part' + str(c)
            chunk_obj.solnum = chunk_obj.chunk
            chunk_obj.output = chunk_obj.outdir + '/part' + str(chunk_obj.chunk) + os.path.basename(chunk_obj.dataset)
            chunk_obj.ntot = nsols
            chunk_list.append(chunk_obj)

        # Split the dataset into parts
#         for chunk_obj in chunk_list:
#             split_ms(chunk_obj.dataset, chunk_obj.output, chunk_obj.t0, chunk_obj.t1)
#
#         # Calibrate in parallel
#         pool = multiprocessing.Pool(ncores)
#         pool.map(calibrate_chunk, chunk_list)
#         pool.close()
#         pool.join()
#
#         # Copy over the solutions to the final parmdb
#         pdb = lofar.parmdb.parmdb(instrument_out)
#         parms = pdb.getValuesGrid("*")
#         for chunk_obj in chunk_list:
#             instrument_input = chunk_obj.output + '/instrument'
#             pdb_part = lofar.parmdb.parmdb(instrument_input)
#             parms_part = pdb_part.getValuesGrid("*")
#             keynames = parms_part.keys()
#             for key in keynames:
#                 if 'Phase' in key:
#                     tmp1=np.copy(parms[key]['values'][:,0])
#                     tmp1[chunk_obj.solnum] = np.copy(parms_part[key]['values'][0,0])
#                     parms[key]['values'][:,0] = tmp1
#         os.system('rm -rf ' + instrument_out)
#         lofar.expion.parmdbmain.store_parms(instrument_out, parms, create_new=True)
#
#         # Clean up
#         for chunk_obj in chunk_list:
#             os.system('rm -rf {0}*'.format(chunk_obj.output))
#         os.system('rm calibrate-stand-alone*.log')
#
#         # Move the solutions to original parmdb
#         subprocess.call('cp -r {0} {1}'.format(instrument_out, instrument_orig),
#             shell=True)


        manager = multiprocessing.Manager()
        pool = multiprocessing.Pool(ncores)
        lock = manager.Lock()
        for chunk_obj in chunk_list:
            pool.apply_async(func=run_chunk, args=(chunk_obj,lock))
        pool.close()
        pool.join()


def run_chunk(chunk_obj, lock):
    """
    run time correlated calibration process for a single chunk.
    1. split data
    2. run bbs
    3. copy solutions to final parmdb
    4. clean directory of files created
    """

    instrument_orig  = chunk_obj.dataset+'/instrument'
    instrument_out = chunk_obj.dataset+'/instrument_out'

    # Split the dataset into parts
    split_ms(chunk_obj.dataset, chunk_obj.output, chunk_obj.t0, chunk_obj.t1)

    # Calibrate
    calibrate_chunk(chunk_obj)

    # Copy over the solutions to the final parmdb
    log.debug('run_chunk(): Copy sols, run lofar.parmdb.parmdb' + str(chunk_obj.solnum) + ' - ' + str(multiprocessing.current_process().name))
    # Lock parmdb
    lock.acquire()
    log.debug('run_chunk(): Copy sols, lock ' + str(chunk_obj.solnum))
    pdb = lofar.parmdb.parmdb(instrument_out)
    parms = pdb.getValuesGrid("*")
    parms_old = pdb.getValuesGrid("*")

    instrument_input = chunk_obj.output + '/instrument'
    pdb_part = lofar.parmdb.parmdb(instrument_input)
    parms_part = pdb_part.getValuesGrid("*")
    keynames = parms_part.keys()
    log.debug('run_chunk(): Copy sols, run -for- loop' + str(chunk_obj.solnum))
    # Replace old value with new
    for key in keynames:
	# Hard-coded to look for Phase and/or TEC parms
	# Presumably OK to use other parms with additional 'or' statments
        if 'Phase' in key or 'TEC' in key:
            tmp1=np.copy(parms[key]['values'][:,0])
            tmp1[chunk_obj.solnum] = np.copy(parms_part[key]['values'][0,0])
            parms[key]['values'][:,0] = tmp1
    log.debug('run_chunk(): Copy sols, end -for- loop' + str(chunk_obj.solnum))

    # Remove previous parmdb values
    for line in parms_old:
        pdb.deleteValues(line)
    # Add new values
    pdb.addValues(parms)
    log.debug('run_chunk(): Copy sols, unlock ' + str(chunk_obj.solnum))
    lock.release()

    # Clean up
    log.debug('run_chunk(): Clean up...')
    os.system('rm -rf {0}*'.format(chunk_obj.output))
    os.system('rm calibrate-stand-alone*.log')


def calibrate_chunk(chunk_obj):
    """Calibrates a single MS chunk using a time-correlated solve"""
    # Modify weights
    fwhm_min, fwhm_max = modify_weights(chunk_obj.output, chunk_obj.ionfactor,
        ntot=chunk_obj.ntot, trim_start=chunk_obj.trim_start)

    # Run bbs
    subprocess.call("calibrate-stand-alone {0} {1} {2} > {3}/logs/"
        "{4}_peeling_calibrate_timecorr.log 2>&1".format(chunk_obj.output, chunk_obj.parset,
        chunk_obj.skymodel, chunk_obj.outdir, chunk_obj.logname_root), shell=True)


def clean_and_copy_parmdb(instrument_name, instrument_out, blockl,
    flag_filler=False, msname=None, timepersample=10.0):
    """Resets and copies a parmdb

    instrument_name - parmdb to copy
    instrument_out - output parmdb
    blockl - block length in time slots
    flag_filler = flag parts that won't be filled with time-correlated solutions
    """
    pdb = lofar.parmdb.parmdb(instrument_name)
    parms = pdb.getValuesGrid("*")
    keynames = parms.keys()
    filler = int(blockl/2.0)

    # Set phases to zero
    for key in keynames:
        if 'Phase' in key:
            tmp1 = np.copy(parms[key]['values'][filler-1:-filler-2,0])
            tmp1 = tmp1*0.0
            parms[key]['values'][filler-1:-filler-2,0] = tmp1
    pdb_out = lofar.parmdb.parmdb(instrument_out, create=True)
    pdb_out.addValues(parms)

    if flag_filler:
        # Flag data for times that won't be filled with time-correlated solutions
        ms = pt.table(msname, readonly=False)
        starttime = ms[0]['TIME']
        endtime   = ms[ms.nrows()-1]['TIME']

        end_block = blockl / 2.0 * timepersample
        tabStationSelection = ms.query('TIME > ' + str(starttime)
            + ' && TIME < ' + str(starttime+end_block),
            sortlist='TIME,ANTENNA1,ANTENNA2')
        tabStationSelection.putcol("FLAG_ROW", numpy.ones(filler, dtype=bool))
        tabStationSelection.putcol("FLAG", numpy.ones((filler, 4), dtype=bool))
        tabStationSelection.close()

        start_block = endtime - (blockl / 2.0 * timepersample)
        tabStationSelection = ms.query('TIME > ' + str(starttime+start_block)
            + ' && TIME < ' + str(endtime),
            sortlist='TIME,ANTENNA1,ANTENNA2')
        tabStationSelection.putcol("FLAG_ROW", numpy.ones(filler, dtype=bool))
        tabStationSelection.putcol("FLAG", numpy.ones((filler, 4), dtype=bool))
        tabStationSelection.close()
        ms.close()


def update_parset(parset):
    """
    Update the parset to set cellsize and chunksize = 0
    where a value of 0 forces all time/freq/cell intervals to be considered
    """
    log.info('Beginning update_parset()...')

    f = open(parset, 'r')
    newlines = f.readlines()
    f.close()
    for i in range(0, len(newlines)):
	if 'ChunkSize' in newlines[i] or 'CellSize.Time' in newlines[i]:
	    log.debug('update_parset(): Updated a line in the parset')
	    vars = newlines[i].split()
	    #newlines[i] = vars[0]+' '+vars[1]+' '+str(blockl)+'\n'
	    newlines[i] = vars[0]+' '+vars[1]+' 0\n'
    f = open(parset,'w')
    f.writelines(newlines)
    f.close()


def split_ms(msin, msout, start_out, end_out):
    """Splits an MS between start and end times in hours relative to first time"""
    t = pt.table(msin, ack=False)

    starttime = t[0]['TIME']
    t1 = t.query('TIME > ' + str(starttime+start_out*3600) + ' && '
      'TIME < ' + str(starttime+end_out*3600), sortlist='TIME,ANTENNA1,ANTENNA2')

    t1.copy(msout, True)
    t1.close()
    t.close()


def modify_weights(msname, ionfactor, dryrun=False, ntot=None, trim_start=True):
    """Modifies the WEIGHTS column of the input MS"""
    t = pt.table(msname, readonly=False, ack=False)
    freq_tab = pt.table(msname + '/SPECTRAL_WINDOW', ack=False)
    freq = freq_tab.getcol('REF_FREQUENCY')
    wav = 3e8 / freq
    anttab = pt.table(msname + '/ANTENNA', ack=False)
    antlist = anttab.getcol('NAME')
    fwhm_list = []

    for t2 in t.iter(["ANTENNA1","ANTENNA2"]):
        if (t2.getcell('ANTENNA1',0)) < (t2.getcell('ANTENNA2',0)):
            weightscol = t2.getcol('WEIGHT_SPECTRUM')
            uvw = t2.getcol('UVW')
            uvw_dist = np.sqrt(uvw[:,0]**2 + uvw[:,1]**2 + uvw[:,2]**2)
            weightscol_modified = np.copy(weightscol)
            timepersample = t2[1]['TIME'] - t2[0]['TIME']
            dist = np.mean(uvw_dist) / 1e3
            stddev = ionfactor * np.sqrt((25e3 / dist)) * (freq / 60e6) # in sec
            fwhm = 2.3548 * stddev
            fwhm_list.append(fwhm[0])

            if not dryrun:
                for pol in range(0,len(weightscol[0,0,:])):
                    for chan in range(0,len(weightscol[0,:,0])):
                        weights = weightscol[:,chan,pol]

                if ntot is None:
                    ntot = len(weights)
                gauss = scipy.signal.gaussian(ntot, stddev/timepersample)
                if trim_start:
                    weightscol_modified[:,chan,pol] = weights * gauss[ntot-len(weights):]
                else:
                    weightscol_modified[:,chan,pol] = weights * gauss[:len(weights)]
                t2.putcol('WEIGHT_SPECTRUM', weightscol_modified)
    t.close()
    freq_tab.close()
    return (min(fwhm_list), max(fwhm_list))


class Field(object):
    """The Field object contains the basic parameters for each field."""
    def __init__(self, name, ra, dec, bands):
        self.name = name
        self.ra = ra
        self.dec = dec
        self.bands = bands


class Band(object):
    """The Band object contains parameters needed for each band (MS)."""
    def __init__(self, MSfile, outdir):
        self.file = MSfile
        self.outdir = outdir
        self.msname = self.file.split('/')[-1]
        self.peeled_file = "{0}/{1}.peeled".format(self.outdir, self.msname)
        sw = pt.table(self.file + '/SPECTRAL_WINDOW', ack=False)
        self.freq = sw.col('REF_FREQUENCY')[0]
        sw.close()
        obs = pt.table(self.file + '/FIELD', ack=False)
        self.ra = np.degrees(float(obs.col('REFERENCE_DIR')[0][0][0]))
        if self.ra < 0.:
            self.ra=360.+(self.ra)
        self.dec = np.degrees(float(obs.col('REFERENCE_DIR')[0][0][1]))
        obs.close()
        ant = pt.table(self.file + '/ANTENNA', ack=False)
        diam = float(ant.col('DISH_DIAMETER')[0])
        ant.close()
        self.fwhm_deg = 1.1*((3.0e8/self.freq)/diam)*180./np.pi
        self.name = str(self.freq)


    def __jug_hash__(self):
        return self.msname


class Chunk(object):
    """The Chunk object contains parameters for time-correlated calibration
    (most of which are set later during calibration).
    """
    def __init__(self, MSfile):
        self.dataset = MSfile


if __name__=='__main__':
    # Get command-line options.
    import optparse
    opt = optparse.OptionParser(usage='%prog <RA (deg)> <Dec (deg)> '
        '<radius (deg)> <outfile>', version='%prog '+_version,
        description=__doc__)
    opt.add_option('-i', '--indir', help='Input directory [default: %default]',
        type='string', default='.')
    opt.add_option('-o', '--outdir', help='Output directory [default: %default]',
        type='string', default='Peeled')
    opt.add_option('-f', '--fluxcut', help='Minimum apparent flux at 60 MHz in '
        'Jy for calibrators [default: %default]', type='float', default='15.0')
    opt.add_option('-g', '--gsm', help='Global sky model file, to use instead '
        'of gsm.py [default: %default]; if given, the RA, Dec and radius '
        'arguments are ignored', type='str', default=None)
    opt.add_option('-l', '--lsm', help='Use local sky model files instead '
        'of the global sky model [default: %default]. Local sky models must be '
        'named MS.skymodel and be in the input directory (e.g. SB50.MS.skymodel).',
        action='store_true', default=False)
    opt.add_option('-m', '--majcut', help='Maximum major axis size in '
        'arcmin for calibrators [default: %default]', type='float', default=None)
    opt.add_option('-B', '--beam', help='Beam mode to use during peeling. Use OFF '
        'to disable the beam [default: %default]', type='str', default='ARRAY_FACTOR')
    opt.add_option('-b', '--nbands', help='Minimum number of bands that a '
        'calibrator must have to be used [default: %default]', type='int', default='8')
    opt.add_option('-a', '--navg', help='Number of frequency channels to '
        'average together before calibration with NDPPP (1 = no averaging) '
        ' [default: %default]', type='int', default='8')
    opt.add_option('-n', '--ncores', help='Number of simultaneous bands '
        'to calibrate [default: %default]', type='int', default='8')
    opt.add_option('-v', '--verbose', help='Set verbose output and interactive '
        'mode [default: %default]', action='store_true', default=False)
    opt.add_option('-D', '--dirindep', help='Perform a dir-independent calibration '
        'before peeling? [default: %default]', action='store_true', default=False)
    opt.add_option('-p', '--patches', help='Group model into patches (any existing '
        'patches are replaced)? [default: %default]', action='store_true', default=False)
    opt.add_option('-P', '--patchtype', help='Type of grouping to use to make '
        'patches (tesselate or cluster) [default: %default]', default='tessellate')
    opt.add_option('-x', '--fluxbin', help='Target flux per bin at 60 MHz in '
        'Jy for tesselation [default: %default]', type='float', default=10.0)
    opt.add_option('-u', '--uvmin', help='UVmin to use during peeling in lambda '
        '[default: %default]', type='float', default=250.0)
    opt.add_option('-N', '--numclusters', help='Number of clusters for clustering '
        '[default: %default]', type='float', default='20')
    opt.add_option('-t', '--timecorr', help='Use time-correlated solutions? '
        '[default: %default]', action='store_true', default=False)
    opt.add_option('-I', '--ionfactor', help='Ionfactor: sets characteristic time '
        'scale of time-correlated solve at 60 MHz and 25 km baseline. I.e.: '
        'FWHM (# of time slots) = 2.35 * ionfactor * sqrt((25e3 / dist)) * (freq / 60e6), '
        'where dist is the baseline length in meters, and freq is the frequency '
        'in Hz [default: %default]', type='float', default=0.25)
    opt.add_option('-d', '--dryrun', help='Do a dry run (no calibration is '
        'done) [default: %default]', action='store_true', default=False)
    opt.add_option('-s', '--solint', help='Solution interval to use '
        'for phase solutions (# of time slots) [default: %default]',
        type='float', default='5')
    opt.add_option('-S', '--scale', help='Scale solution interval as '
        'solint*(fluxMax/flux)^2 [default: %default]', action='store_true', default=False)
    opt.add_option('-F', '--flag', help='Flag outliers in peeling solutions? '
        '[default: %default]', action='store_true', default=False)
    opt.add_option('-j', '--jug', help='Use jug? '
        '[default: %default]', action='store_true', default=False)
    opt.add_option('-c', '--clobber', help='Clobber existing output files? '
        '[default: %default]', action='store_true', default=False)
    try:
        import cfgparse
        cfg = cfgparse.ConfigParser()
        cfg.add_optparse_help_option(opt)
        cfg.add_option('sky_ra')
        cfg.add_option('sky_dec')
        cfg.add_option('sky_radius')
        cfg.add_option('outfile')
        cfg.add_option('indir')
        cfg.add_option('outdir')
        cfg.add_option('nbands')
        cfg.add_option('ncores')
        cfg.add_option('navg')
        cfg.add_option('lsm')
        cfg.add_option('gsm')
        cfg.add_option('fluxcut')
        cfg.add_option('solint')
        cfg.add_option('majcut')
        cfg.add_option('beam')
        cfg.add_option('timecorr')
        cfg.add_option('jug')
        cfg.add_file('ion_peel.ini')
        (options, args) = cfg.parse(opt)
    except ImportError:
        (options, args) = opt.parse_args()

    # Get inputs
    if len(args) != 4 and len(args) != 0:
        opt.print_help()
    else:
        if len(args) == 0:
            try:
                sky_ra = float(options.sky_ra)
                sky_dec = float(options.sky_dec)
                sky_radius = float(options.sky_radius)
                outfile = options.outfile
            except:
                opt.print_help()
                sys.exit()
        else:
            sky_ra = float(args[0])
            sky_dec = float(args[1])
            sky_radius = float(args[2])
            outfile = args[3]

        # Set up output directories and initialize logging
        outdir = options.outdir
        if not os.path.isdir(outdir):
            os.mkdir(outdir)
            logfilename = outdir + '/' + outfile + '.log'
            init_logger(logfilename, debug=options.verbose)
            log = logging.getLogger("Main")
        elif options.clobber:
            subprocess.call("rm -rf {0}".format(outdir), shell=True)
            os.mkdir(outdir)
            logfilename = outdir + '/' + outfile + '.log'
            init_logger(logfilename, debug=options.verbose)
            log = logging.getLogger("Main")
        elif options.jug:
            logfilename = outdir + '/' + outfile + '.log'
            init_logger(logfilename, debug=options.verbose)
            log = logging.getLogger("Main")
        else:
            logfilename = outdir + '/' + outfile + '.log'
            init_logger(logfilename, debug=options.verbose)
            log = logging.getLogger("Main")
            log.error("The peeling results directory already exists! Please\n"
                "rename/move/delete it, or set the clobber (-c) flag.")
            sys.exit()

        if not os.path.isdir(outdir+"/skymodels"):
            os.mkdir(outdir+"/skymodels")
        if not os.path.isdir(outdir+"/logs"):
            os.mkdir(outdir+"/logs")
        if not os.path.isdir(outdir+"/parsets"):
            os.mkdir(outdir+"/parsets")
        if os.path.exists(outfile):
            if options.clobber:
                subprocess.call("rm -rf {0}".format(outfile), shell=True)
            else:
                log.error("The output H5parm file already exists! Please\n"
                "rename/move/delete it, or set the clobber (-c) flag.")
                sys.exit()

        # Scan the directories to determine fields.
        field_list = scan_directory(options.indir, outdir)
        if len(field_list) == 0:
            log.error('No measurement sets found in input directory!')
            sys.exit()

        # Make a master sky model from which calibrators will be chosen. The flux
        # cut is set to 10% of the input flux cutoff.
        log.info('Searching sky model for suitable calibrators...')
        if options.gsm is not None:
            master_skymodel = options.gsm
        else:
            master_skymodel = outdir + '/skymodels/potential_calibrators.skymodel'
            if os.exists(master_skymodel) and options.jug:
                pass
            else:
                subprocess.call("gsm.py {0} {1} {2} {3} {4} 2>/dev/null".format(master_skymodel,
                    sky_ra, sky_dec, sky_radius, options.fluxcut/10.0), shell=True)

        if options.patches:
            log.info('  Tessellating the sky model...')
            patch_skymodel = outdir + '/skymodels/potential_calibrators_patches.skymodel'
            s = lsmtool.load(master_skymodel)
            s.group(options.patchtype, targetFlux=options.fluxbin,
                numClusters=options.numclusters)
            s.setPatchPositions(method='mid')
            if options.verbose:
                print('Showing tessellated sky model. Close the plot window to continue.')
                s.plot()
            s.write(patch_skymodel, clobber=options.clobber)

        # Determine potential calibrators for each band and field.
        if options.patches:
            master_skymodel = patch_skymodel
        if options.majcut is not None:
            majcut_arcsec = options.majcut * 60.0
        else:
            majcut_arcsec = None
        if options.beam.lower() == 'off':
            applyBeam = False
        else:
            applyBeam = True
        cal_sets = []
        for field in field_list:
            for band in field.bands:
                band.master_skymodel = master_skymodel
                if options.lsm:
                    skymodel = band.file + '.skymodel'
                    if not os.path.exists(skymodel):
                        skymodel = band.file.split('.')[0] + '.skymodel'
                    band.skymodel = skymodel
                else:
                    band.skymodel = master_skymodel
                cal_names, cal_fluxes, cal_sizes, hasPatches = find_calibrators(master_skymodel,
                    band.file, options.fluxcut, majcut_arcsec, plot=options.verbose,
                    applyBeam=applyBeam, band_skymodel=band.skymodel)
                if options.verbose:
                    prompt = "Press enter to continue or 'q' to quit... : "
                    answ = raw_input(prompt)
                    while answ != '':
                        if answ == 'q':
                            sys.exit()
                        answ = raw_input(prompt)

                band.cal_names = cal_names
                band.cal_apparent_fluxes = cal_fluxes
                band.cal_sizes = cal_sizes
                band.master_skymodel = master_skymodel
                band.use_patches = hasPatches

        # Make a list of all calibrators and bands.
        cal_names = []
        band_names = []
        for field in field_list:
            for band in field.bands:
                cal_names += band.cal_names
                band_names.append(band.name)
        cal_names_set = set(cal_names)
        band_names_set = set(band_names)

        # Eliminate duplicate calibrators by selecting brightest one in each band
        # if there is more than one field.
        if len(field_list) > 1:
            # Loop over calibrators and remove them from bands in which they are
            # not the brightest.
            for band_name in band_names_set:
                for cal_name in cal_names_set:
                    max_cal_apparent_flux = 0.0

                    # For each band and calibrator, find the field in which it
                    # is brightest.
                    max_cal_field_name = field_list[0].name
                    for field in field_list:
                        for band in field.bands:
                            if cal_name in band.cal_names and band.name == band_name:
                                indx = band.cal_names.index(cal_name)
                                if band.cal_apparent_fluxes[indx] > max_cal_apparent_flux:
                                    max_cal_apparent_flux = band.cal_apparent_fluxes[indx]
                                    max_cal_field_name = field.name

                    # Now remove all but the brightest calibrator.
                    for field in field_list:
                        if field.name != max_cal_field_name:
                            for band in field.bands:
                                if cal_name in band.cal_names and band.name == band_name:
                                    indx = band.cal_names.index(cal_name)
                                    band.cal_names.pop(indx)
                                    band.cal_apparent_fluxes.pop(indx)
                                    band.cal_sizes.pop(indx)

        # Remove calibrators that do not appear in enough bands (set by
        # options.nbands)
        cals_to_remove = []
        if len(cal_names_set) == 1:
            caltxt = 'calibrator'
        else:
            caltxt = 'calibrators'
        if options.majcut is None:
            logtxt = '{0} {1} found (Sint > {2} Jy):'.format(
                len(cal_names_set), caltxt, options.fluxcut)
        else:
            logtxt = '{0} {1} found (Sint > {2} Jy; Maj < {3} arcmin):'.format(
                len(cal_names_set), caltxt, options.fluxcut, options.majcut)
        for cal_name in cal_names_set:
            nbands = 0
            for field in field_list:
                for band in field.bands:
                    if cal_name in band.cal_names:
                        nbands += 1
            if nbands < options.nbands:
                cals_to_remove.append(cal_name)
            logtxt += "\n      '{0}': {1} bands".format(cal_name, nbands)
        log.info(logtxt)

        if len(cals_to_remove) > 0:
            for field in field_list:
                for band in field.bands:
                    for cal_name in band.cal_names[:]:
                        if cal_name in cals_to_remove:
                            indx = band.cal_names.index(cal_name)
                            band.cal_names.pop(indx)
                            band.cal_apparent_fluxes.pop(indx)
                            band.cal_sizes.pop(indx)
        cal_names_final = []
        for cal_name in cal_names_set:
            nbands = 0
            for field in field_list:
                for band in field.bands:
                    if cal_name in band.cal_names:
                        cal_names_final.append(cal_name)
        cal_names_set = set(cal_names_final)
        log.info('{0} {1} available in {2} or more bands'.format(
            len(cal_names_set), caltxt, options.nbands))
        if len(cal_names_set) == 0:
            sys.exit()
        if options.verbose:
            prompt = "Press enter to continue or 'q' to quit... : "
            answ = raw_input(prompt)
            while answ != '':
                if answ == 'q':
                    sys.exit()
                answ = raw_input(prompt)

        # Make list of bands to peel and set their options
        band_list = []
        freq_list = []
        for field in field_list:
            for band in field.bands:
                band_list.append(band)
                freq_list.append(band.freq)
        for band in band_list:
            # For each Band instance, set options
            band.beam_mode = options.beam
            band.do_peeling = True
            band.nsrc_per_bin = 1
            band.use_scalar_phase = True
            band.use_timecorr = options.timecorr
            band.phase_only = False
            band.solint_min = options.solint
            band.navg = options.navg
            band.solint_amp = 330
            band.time_block = 60 # number of time samples in a block
            band.flag_filler = False # flag filler solutions
            band.ionfactor = options.ionfactor
            band.ncores_per_cal = 3
            band.do_each_cal_sep = False
            band.scale_solint = options.scale
            band.do_dirindep = options.dirindep
            band.uvmin = options.uvmin
            if band.use_timecorr and (np.remainder(band.time_block, 2) or
                np.remainder(band.time_block, band.solint_min)):
                log.warning('For best results, the number of time samples in a '
                    'block should be evenly divisble both by 2 and by the '
                    'solution interval')

        # Setup peeling
        for band in band_list:
            setup_peeling(band)

        # Perform peeling for each band. The peel_band script will split up the
        # calibrators for each band into sets for peeling.
        for band in band_list[:]:
            if not band.do_peeling:
                band_list.remove(band)
        if not options.dryrun:
            if not has_jug or not options.jug:
                pool = MyPool(options.ncores)
                pool.map(peel_band, band_list)
                pool.close()
                pool.join()

                # Write all the solutions to an H5parm file for later use in LoSoTo.
                write_sols(field_list, outdir+'/'+outfile, flag_outliers=options.flag)
                log.info('Peeling complete.')
            else:
                for band in band_list:
                    Task(peel_band, band)

                # Write all the solutions to an H5parm file for later use in LoSoTo.
                Task(write_sols, field_list, outdir+'/'+outfile, options.flag)
                log.info('Peeling complete.')
        else:
            log.info('Dry-run flag (-d) was set. No peeling done.')

