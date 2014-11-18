#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Defines the various functions and classes needed by the ionospheric scripts
"""
import logging
import os
import commands
import subprocess
import glob
import shutil
import sys
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
import socket
import time


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
    log = logging.getLogger("Find_calibrators")
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
    if band.init_logger:
        logfilename = band.outdir + '/logs/' + band.msname + '.peel_band.log'
        init_logger(logfilename)
    log = logging.getLogger("Peeler")

    # Wrap everything in a try-except block to be sure any exception is caught
    try:
        # Check if peeling is required
        if not band.do_peeling:
            return

        time.sleep(band.peel_start_delay)

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
        if not band.resume or (band.resume and not os.path.exists(newmsname)):
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
        if band.do_dirindep and not band.resume or (band.do_dirindep and band.resume
            and not os.path.exists('{0}/state/{1}_dirindep.done'.format(band.outdir,
            band.msname))):
            dirindep_parset = '{0}/parsets/{1}.dirindep.parset'.format(
                band.outdir, msname)
            make_dirindep_parset(dirindep_parset, scalar_phase=band.use_scalar_phase,
                sol_int=band.solint_min, beam_mode=band.beam_mode, uvmin=band.uvmin)
            subprocess.call("calibrate-stand-alone -f {0} {1} {2} > {3}/logs/"
                "{4}_dirindep_calibrate.log 2>&1".format(newmsname, dirindep_parset,
                skymodel, band.outdir, msname), shell=True)

            # Save state
            cmd = 'touch {0}/state/{1}_dirindep.done'.format(band.outdir,
                band.msname)
            subprocess.call(cmd, shell=True)

        # Perform the peeling. Do this step even if time-correlated solutions
        # are desired so that the proper parmdb is made and so that the correlation
        # time can be estimated
        if not band.resume or (band.resume and
            not os.path.exists('{0}/state/{1}_initialpeel.done'.format(band.outdir,
            band.msname))):
            subprocess.call("calibrate-stand-alone -f {0} {1} {2} > {3}/logs/"
                "{4}_peeling_calibrate.log 2>&1".format(newmsname, peelparset,
                skymodel, band.outdir, msname), shell=True)

            # Save state
            cmd = 'touch {0}/state/{1}_initialpeel.done'.format(band.outdir,
                band.msname)
            subprocess.call(cmd, shell=True)

        if band.use_timecorr:
            # Do time-correlated peeling.

            # Estimate ionfactor from the non-time-correlated peeling solutions
            # TODO: allow ionfactor to vary with time -- maybe a dictionary of
            # {'ionfactor': [0.5, 0.3, 0.2], 'start_sol_num': [0, 24, 256]}?
    #        band.ionfactor = get_ionfactor(msname, instrument='instrument')

            if band.subfield_first:
                # Subtract the field before peeling
                if not band.resume or (band.resume and
                    not os.path.exists('{0}/state/{1}_subtract.done'.format(band.outdir,
                    band.msname))):
                    subparset = '{0}/parsets/{1}.subtract_field.parset'.format(
                        band.outdir, msname)
                    make_subtract_parset(subparset, source_list=None, beam_mode=band.beam_mode,
                        output_column='SUBTRACTED_DATA')
                    subprocess.call("calibrate-stand-alone {0} {1} {2} > {3}/logs/"
                        "{4}_subtract_field.log 2>&1".format(newmsname, subparset,
                        skymodel, band.outdir, msname), shell=True)

                    # Save state
                    cmd = 'touch {0}/state/{1}_subtract.done'.format(band.outdir,
                        band.msname)
                    subprocess.call(cmd, shell=True)

                # Make a new sky model with only the calibrators
                cal_skymodel = skymodel + '.cals_only'
                cal_list = []
                for peel_bin in band.peel_bins:
                    cal_list += peel_bin['names']
                s = lsmtool.load(skymodel)
                if s.hasPatches:
                    s.select('Patch == [{0}]'.format(','.join(cal_list)))
                else:
                    s.select('Name == [{0}]'.format(','.join(cal_list)))
                s.write(cal_skymodel, clobber=True)

            # Do the peeling
            peelparset_timecorr = '{0}/parsets/{1}.timecorr_peeling.parset'.format(
                band.outdir, msname)
            if band.subfield_first:
                make_peeling_parset(peelparset_timecorr, band.peel_bins,
                    scalar_phase=band.use_scalar_phase, phase_only=True,
                    time_block=band.time_block, beam_mode=band.beam_mode,
                    uvmin=band.uvmin, skip_field=True, input_column='SUBTRACTED_DATA')
                calibrate(newmsname, peelparset_timecorr, cal_skymodel, msname,
                    use_timecorr=True, outdir=band.outdir, instrument='instrument',
                    time_block=band.time_block, ionfactor=band.ionfactor,
                    solint=band.solint_min,
                    ncores=band.ncores_per_cal, resume=band.resume)
            else:
                make_peeling_parset(peelparset_timecorr, band.peel_bins,
                    scalar_phase=band.use_scalar_phase, phase_only=True,
                    time_block=band.time_block, beam_mode=band.beam_mode,
                    uvmin=band.uvmin)
                calibrate(newmsname, peelparset_timecorr, skymodel, msname,
                    use_timecorr=True, outdir=band.outdir, instrument='instrument',
                    time_block=band.time_block, ionfactor=band.ionfactor,
                    solint=band.solint_min,
                    ncores=band.ncores_per_cal, resume=band.resume)

        return {'host':socket.gethostname(), 'name':band.msname}
    except Exception as e:
        log.error(str(e))


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


def make_subtract_parset(parset, source_list=None, beam_mode='DEFAULT',
    output_column='SUBTRACTED_DATA'):
    """Makes a BBS parset for subtraction
    """
    if os.path.exists(parset):
        return
    if source_list is None:
        source_list = ''

    # Set overall strategy
    newlines = ['Strategy.InputColumn = DATA\n',
        'Strategy.ChunkSize = 250\n',
        'Strategy.Baselines = [CR]S*&\n',
        'Strategy.UseSolver = F\n']
    strategy_str = 'Strategy.Steps = [subtract]\n'
    newlines += strategy_str

    # Handle beam
    if beam_mode.lower() == 'off':
        beam_enable = 'F'
        beam_mode = 'DEFAULT'
    else:
        beam_enable = 'T'

    # Subtract sources
    newlines += ['\n', 'Step.subtract.Operation = SUBTRACT\n',
        'Step.subtract.Model.Sources = [{0}]\n'.format(source_list),
        'Step.subtract.Model.Beam.Enable = {0}\n'.format(beam_enable),
        'Step.subtract.Model.Beam.Mode = {0}\n'.format(beam_mode),
        'Step.subtract.Output.Column = {0}\n'.format(output_column)]

    f = open(parset, 'w')
    f.writelines(newlines)
    f.close()


def make_peeling_parset(parset, peel_bins, scalar_phase=True, phase_only=True,
    sol_int_amp=500, time_block=None, beam_mode='DEFAULT', uvmin=250.0,
    skip_field=False, input_column='DATA'):
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
    newlines = ['Strategy.InputColumn = {0}\n'.format(input_column),
        'Strategy.ChunkSize = {0}\n'.format(int(max(sol_int_list))),
        'Strategy.Baselines = [CR]S*&\n',
        'Strategy.UseSolver = F\n']
    if phase_only:
        pstr = ''
        if not skip_field:
            strategy_str = 'Strategy.Steps = [subtractfield'
        else:
            strategy_str = 'Strategy.Steps = ['
        for i, peel_bin in enumerate(peel_bins):
            if i == 0 and skip_field:
                strategy_str += 'add{0}, solve{0}'.format(i+1)
            else:
                strategy_str += ', add{0}, solve{0}'.format(i+1)

            if i < nbins - 1:
                strategy_str += ', subtract{0}'.format(i+1)
        strategy_str += ']\n'
    else:
        pstr = 'p'
        if not skip_field:
            strategy_str = 'Strategy.Steps = [subtractfield'
        else:
            strategy_str = 'Strategy.Steps = ['
        for i, peel_bin in enumerate(peel_bins):
            if i == 0 and skip_field:
                strategy_str += 'add{0}, solvep{0}, solvea{0}'.format(i+1)
            else:
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
    if not skip_field:
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


def make_screen_parset(parset, solint=1, uvmin=80, freqint=10,
    beam_mode='DEFAULT'):
    """Makes BBS parset for dir-indep calibration with screen included"""

    # Handle beam
    if beam_mode.lower() == 'off':
        beam_enable = 'F'
        beam_mode = 'DEFAULT'
    else:
        beam_enable = 'T'

    newlines = ['Strategy.Stations = []\n',
        'Strategy.InputColumn = DATA\n',
        'Strategy.ChunkSize = 250\n',
        'Strategy.UseSolver = F\n',
        'Strategy.Steps = [solve]\n',
        'Strategy.Baselines = *& \n',
        'Step.solve.Operation = SOLVE\n',
        'Step.solve.Model.Sources = []\n',
        'Step.solve.Model.Ionosphere.Enable = T\n',
        'Step.solve.Model.Ionosphere.Type = EXPION\n',
        'Step.solve.Model.Beam.Enable = {0}\n'.format(beam_enable),
        'Step.solve.Model.Beam.Mode = {0}\n'.format(beam_mode),
        'Step.solve.Model.Beam.UseChannelFreq = T\n',
        'Step.solve.Model.Cache.Enable = T\n',
        'Step.solve.Model.Gain.Enable = F\n',
        'Step.solve.Model.CommonScalarPhase.Enable= T\n',
        'Step.solve.Solve.Mode = COMPLEX\n',
        'Step.solve.Solve.UVRange = [{0}]\n'.format(uvmin),
        'Step.solve.Solve.Parms = ["CommonScalarPhase:*"]\n',
        'Step.solve.Solve.CellSize.Freq = {0}\n'.format(freqint),
        'Step.solve.Solve.CellSize.Time = {0}\n'.format(solint),
        'Step.solve.Solve.CellChunkSize = 1\n',
        'Step.solve.Solve.PropagateSolutions = T']
    f = open(parset, 'w')
    f.writelines(newlines)
    f.close()
    return parset


def apply_band(band):
    """Apply TEC screen to band"""
    if band.init_logger:
        logfilename = band.outdir + '/logs/' + band.msname + '.apply_band.log'
        init_logger(logfilename)
    log = logging.getLogger("Applier")

    # Wrap everything in a try-except block to be sure any exception is caught
    try:
        # Define file names
        msname = band.msname
        skymodel =  "{0}/skymodels/{1}.apply.skymodel".format(band.outdir, msname)
        screen_parset = "{0}/parsets/{1}.screen.parset".format(band.outdir, msname)
#         noscreen_parset = "{0}/parsets/{1}.noscreen.parset".format(band.outdir, msname)

        # Perform dir-independent calibration without the TEC screen to set up
        # instrument db
#         if not band.resume or (band.resume
#             and not os.path.exists('{0}/state/{1}_dirindep_noscreen.done'.format(band.outdir,
#             band.msname))):
#             make_noscreen_parset(noscreen_parset, scalar_phase=band.use_scalar_phase,
#                 sol_int=band.solint_min, beam_mode=band.beam_mode, uvmin=band.uvmin)
#             subprocess.call("calibrate-stand-alone -f {0} {1} {2} > {3}/logs/"
#                 "{4}_dirindep_noscreen_calibrate.log 2>&1".format(newmsname, noscreen_parset,
#                 skymodel, band.outdir, msname), shell=True)
#
#             # Save state
#             cmd = 'touch {0}/state/{1}_dirindep.done'.format(band.outdir,
#                 band.msname)
#             subprocess.call(cmd, shell=True)

        # Perform dir-independent calibration with the TEC screen
        make_screen_parset(screen_parset, solint=band.solint_min,
            beam_mode=band.beam_mode, uvmin=band.uvmin)
        chunk_list, chunk_list_orig = calibrate(band.outdir+'/'+msname, screen_parset, skymodel, msname,
            use_timecorr=True, outdir=band.outdir, instrument=band.parmdb,
            time_block=band.time_block, ionfactor=None, ncores=band.ncores_per_cal,
            resume=band.resume)
        return chunk_list, chunk_list_orig
    except Exception as e:
        log.error(str(e))


def calibrate(msname, parset, skymodel, logname_root, use_timecorr=False,
    time_block=None, ionfactor=0.5, outdir='.', instrument='instrument',
    solint=None, ncores=1, resume=False):
    """Calls BBS to calibrate with optional time-correlated or distributed fitting"""
    log = logging.getLogger("Calib")

    instrument_orig = '{0}/{1}'.format(msname, instrument)
    instrument_out = '{0}/{1}_out'.format(msname, instrument)

    # Make sure output parmdb does not exist
    if os.path.exists(instrument_out):
        shutil.rmtree(instrument_out)

    if not use_timecorr:
        subprocess.call("calibrate-stand-alone {0} {1} {2} > {3}/logs/"
            "{4}_peeling_calibrate.log 2>&1".format(msname, parset, skymodel,
            outdir, logname_root), shell=True)
        subprocess.call("cp -r {0} {1}".format(instrument_orig, instrument_out),
            shell=True)
    else:
        # Perform a time-correlated or distributed solve
        dataset = msname
        blockl = time_block
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
        if ionfactor is not None:
            fwhm_min, fwhm_max = modify_weights(msname, ionfactor, dryrun=True) # s
            if time_block is None:
                # Set blockl to enclose the max FWHM and be divisible by 2 and by solint
                blockl = int(np.ceil(fwhm_max / timepersample / 2.0 / solint) * 2 * solint)
        tdiff = solint * timepersample / 3600. # difference between chunk start times in hours
        tlen = timepersample * np.float(blockl) / 3600. # length of block in hours
        if ionfactor is None:
            nsols = int(np.ceil(trows / solint / blockl)) # number of solutions/chunks
        else:
            nsols = int(np.ceil(trows / solint)) # number of solutions/chunks

        if ionfactor is not None:
            log.info('Performing time-correlated peeling for {0}...\n'
                '      Time per sample: {1} (s)\n'
                '      Samples in total: {2}\n'
                '      Block size: {3} (samples)\n'
                '                  {4} (s)\n'
                '      Solution interval: {5} (samples)\n'
                '      Number of solutions: {6}\n'
                '      Ionfactor: {7}\n'
                '      FWHM range: {8} - {9} (s)'.format(msname, timepersample,
                trows, blockl, tlen*3600.0, solint, nsols, ionfactor, fwhm_min, fwhm_max))
        else:
            log.info('Performing distributed calibration for {0}...\n'
                '      Time per sample: {1} (s)\n'
                '      Samples in total: {2}\n'
                '      Block size: {3} (samples)\n'
                '                  {4} (s)\n'
                '      Solution interval: {5} (samples)\n'
                '      Number of solutions: {6}\n'
                '      Ionfactor: {7}'.format(msname, timepersample,
                trows, blockl, tlen*3600.0, solint, nsols, ionfactor))


        # Update cellsize and chunk size of parset
        if ionfactor is not None:
            update_parset(parset)

        # Set up the chunks
        chunk_list = []
        if ionfactor is None:
            chunk_mid_start = 0
            chunk_mid_end = nsols
            tdiff = tlen
        else:
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
            chunk_obj.solrange = range(chunk_obj.solnum, chunk_obj.solnum+int(np.ceil(blockl/solint)))
            chunk_obj.output = chunk_obj.outdir + '/part' + str(chunk_obj.chunk) + os.path.basename(chunk_obj.dataset)
            chunk_obj.output_instrument = '{0}/parmdbs/part{1}{2}_instrument'.format(chunk_obj.outdir,
                    chunk_obj.chunk, os.path.basename(chunk_obj.dataset))
            chunk_obj.state_file = '{0}/state/part{1}{2}.done'.format(chunk_obj.outdir,
                    chunk_obj.chunk, os.path.basename(chunk_obj.dataset))
            chunk_obj.ntot = blockl
            chunk_obj.start_delay = 0.0
            chunk_list.append(chunk_obj)

        chunk_list_orig = chunk_list[:]
        if resume:
            # Determine which chunks need to be calibrated
            for chunk_obj in chunk_list_orig:
                if os.path.exists(chunk_obj.state_file):
                    chunk_list.remove(chunk_obj)
            if len(chunk_list) > 0:
                log.info('Resuming time-correlated calibration for {0}...'.format(msname))
                log.debug('Chunks remaining to be calibrated:')
                for chunk_obj in chunk_list:
                    log.debug('  Solution #{0}'.format(chunk_obj.chunk))
            else:
                log.info('Peeling complete for {0}.'.format(msname))

        if len(chunk_list) > 0:
            if ionfactor is None:
                return chunk_list, chunk_list_orig

            # Run chunks in parallel
            pool = multiprocessing.Pool(ncores)
            pool.map(run_chunk, chunk_list)
            pool.close()
            pool.join()

        # Copy over the solutions to the final output parmdb
        try:
            log.info('Copying time-correlated solutions to output parmdb...')
            pdb = lofar.parmdb.parmdb(instrument_orig)
            parms = pdb.getValuesGrid("*")
            for chunk_obj in chunk_list_orig:
                chunk_instrument = chunk_obj.output_instrument
                try:
                    pdb_part = lofar.parmdb.parmdb(chunk_instrument)
                except:
                    continue
                parms_part = pdb_part.getValuesGrid("*")
                keynames = parms_part.keys()

                # Replace old value with new
                for key in keynames:
                # Hard-coded to look for Phase and/or TEC parms
                # Presumably OK to use other parms with additional 'or' statments
                    if 'Phase' in key or 'TEC' in key:
                        parms[key]['values'][chunk_obj.solnum, 0] = np.copy(
                            parms_part[key]['values'][0, 0])

            # Add new values to final output parmdb
            pdb_out = lofar.parmdb.parmdb(instrument_out, create=True)
            pdb_out.addValues(parms)
        except Exception as e:
            log.error(str(e))


def run_chunk(chunk_obj):
    """
    run time correlated calibration process for a single chunk.
    1. split data
    2. run bbs
    3. copy solutions to final parmdb
    4. clean directory of files created
    """
    log = logging.getLogger("Run_chunk")
    time.sleep(chunk_obj.start_delay)

    # Wrap everything in a try-except block to be sure any exception is caught
    try:
        # Split the dataset into parts
        split_ms(chunk_obj.dataset, chunk_obj.output, chunk_obj.t0, chunk_obj.t1)

        # Calibrate
        calibrate_chunk(chunk_obj)

        # Clean up, copying instrument parmdb for later collection
        subprocess.call('cp -r {0}/instrument {1}'.
            format(chunk_obj.output, chunk_obj.output_instrument), shell=True)
        shutil.rmtree(chunk_obj.output)

        # Record successful completion
        success_file = chunk_obj.state_file
        cmd = 'touch {0}'.format(success_file)
        subprocess.call(cmd, shell=True)
    except Exception as e:
        log.error(str(e))


def calibrate_chunk(chunk_obj):
    """Calibrates a single MS chunk using a time-correlated solve"""
    if chunk_obj.ionfactor is not None:
        # Modify weights
        fwhm_min, fwhm_max = modify_weights(chunk_obj.output, chunk_obj.ionfactor,
            ntot=chunk_obj.ntot, trim_start=chunk_obj.trim_start)

    # Run bbs
    subprocess.call("calibrate-stand-alone {0} {1} {2} > {3}/logs/"
        "{4}_peeling_calibrate_timecorr.log 2>&1".format(chunk_obj.output, chunk_obj.parset,
        chunk_obj.skymodel, chunk_obj.outdir, chunk_obj.logname_root), shell=True)


def clean_and_copy_parmdb(instrument_name, instrument_out, blockl):
    """Resets and copies a parmdb

    instrument_name - parmdb to copy
    instrument_out - output parmdb
    blockl - block length in time slots
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


def update_parset(parset):
    """
    Update the parset to set cellsize and chunksize = 0
    where a value of 0 forces all time/freq/cell intervals to be considered
    """
    f = open(parset, 'r')
    newlines = f.readlines()
    f.close()
    for i in range(0, len(newlines)):
	if 'ChunkSize' in newlines[i] or 'CellSize.Time' in newlines[i]:
	    vars = newlines[i].split()
	    newlines[i] = vars[0]+' '+vars[1]+' 0\n'
    f = open(parset,'w')
    f.writelines(newlines)
    f.close()


def split_ms(msin, msout, start_out, end_out):
    """Splits an MS between start and end times in hours relative to first time"""
    if os.path.exists(msout):
        os.system('rm -rf {0}'.format(msout))
    if os.path.exists(msout):
        os.system('rm -rf {0}'.format(msout))

    t = pt.table(msin, ack=False)

    starttime = t[0]['TIME']
    t1 = t.query('TIME > ' + str(starttime+start_out*3600) + ' && '
      'TIME < ' + str(starttime+end_out*3600), sortlist='TIME,ANTENNA1,ANTENNA2')

    t1.copy(msout, True)
    t1.close()
    t.close()


def modify_weights(msname, ionfactor, dryrun=False, ntot=None, trim_start=True):
    """Modifies the WEIGHTS column of the input MS"""
    log = logging.getLogger("Modify_weights")

    t = pt.table(msname, readonly=False, ack=False)
    freqtab = pt.table(msname + '/SPECTRAL_WINDOW', ack=False)
    freq = freqtab.getcol('REF_FREQUENCY')
    freqtab.close()
    wav = 3e8 / freq
    fwhm_list = []

    for t2 in t.iter(["ANTENNA1", "ANTENNA2"]):
        if (t2.getcell('ANTENNA1', 0)) < (t2.getcell('ANTENNA2', 0)):
            weightscol = t2.getcol('WEIGHT_SPECTRUM')
            uvw = t2.getcol('UVW')
            uvw_dist = np.sqrt(uvw[:, 0]**2 + uvw[:, 1]**2 + uvw[:, 2]**2)
            weightscol_modified = np.copy(weightscol)
            timepersample = t2[1]['TIME'] - t2[0]['TIME']
            dist = np.mean(uvw_dist) / 1e3
            stddev = ionfactor * np.sqrt((25e3 / dist)) * (freq / 60e6) # in sec
            fwhm = 2.3548 * stddev
            fwhm_list.append(fwhm[0])
            nslots = len(weightscol[:, 0, 0])
            if ntot is None:
                ntot = nslots
            elif ntot < nslots:
                log.debug('Number of samples for Gaussian is {0}, but number '
                    'in chunk is {1}. Setting number for Gaussian to {1}.'.format(ntot, nslots))
                ntot = nslots
            gauss = scipy.signal.gaussian(ntot, stddev/timepersample)

            if not dryrun:
                for pol in range(0, len(weightscol[0, 0, :])):
                    for chan in range(0, len(weightscol[0, :, 0])):
                        weights = weightscol[:, chan, pol]
                        if trim_start:
                            weightscol_modified[:, chan, pol] = weights * gauss[ntot - len(weights):]
                        else:
                            weightscol_modified[:, chan, pol] = weights * gauss[:len(weights)]
                t2.putcol('WEIGHT_SPECTRUM', weightscol_modified)
    t.close()
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


class Chunk(object):
    """The Chunk object contains parameters for time-correlated calibration
    (most of which are set later during calibration).
    """
    def __init__(self, MSfile):
        self.dataset = MSfile
