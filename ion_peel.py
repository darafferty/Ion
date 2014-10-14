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

The script supports running over multiple nodes using IPython and PBS. To run
in this way, make a PBS script and run it with qsub as follows:

run_peel.pbs:

    #!/bin/bash
    #PBS -N peeling_10SB
    #PBS -l walltime=100:00:00
    #PBS -l nodes=10                --> each node gets one band
    #PBS -j oe
    #PBS -o output-$PBS_JOBNAME-$PBS_JOBID
    #PBS -m bea
    #PBS -M drafferty@hs.uni-hamburg.de

    cd $PBS_O_WORKDIR
    python ion_peel.py 90.80229 42.28977 10 peeled_10SB.h5 -n 1

Run with:

    $ qsub run_peel.pbs

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
try:
    import loadbalance
    has_ipy_parallel = True
except ImportError:
    has_ipy_parallel = False
from ion_libs import *

_version = '1.0'


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
    opt.add_option('-T', '--torque', help='Use torque? '
        '[default: %default]', action='store_true', default=False)
    opt.add_option('-c', '--clobber', help='Clobber existing output files? '
        '[default: %default]', action='store_true', default=False)
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
            if has_ipy_parallel and options.torque:
                log.info('Distributing peeling over PBS nodes...')
                lb = loadbalance.LoadBalance(ppn=options.ncores)
                lb.set_retries(5)
                dview = lb.rc[:]
                dview.execute('from Ion.ion_libs import *')
                lb.lview.map(peel_band, band_list)
            else:
                pool = MyPool(options.ncores)
                pool.map(peel_band, band_list)
                pool.close()
                pool.join()

            # Write all the solutions to an H5parm file for later use in LoSoTo.
            write_sols(field_list, outdir+'/'+outfile, flag_outliers=options.flag)
            log.info('Peeling complete.')

        else:
            log.info('Dry-run flag (-d) was set. No peeling done.')

