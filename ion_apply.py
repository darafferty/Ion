#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script applies TEC screens generated by LoSoTo's TECFIT and TECSCREEN
operations.

Steps:
  - runs H5parmExporter.py to export TEC screens to parmdbs
  - runs BBS for dir-indep. calibration with screen included. Calibration is
    done on the DATA column, so any direction-independent selfcal solutions
    must have been applied to the DATA column before running this script
"""

import os
import sys
import glob
import numpy
import lofar.parmdb
from multiprocessing import Pool
import lofar.parameterset
from losoto.h5parm import h5parm
import pyrap.tables
import logging
try:
    import loadbalance
    has_ipy_parallel = True
except ImportError:
    has_ipy_parallel = False
from Ion.ion_libs import *

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


def makeNonDirParset(outdir, solint=1, uvmin=80, freqint=10,
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
    parset = outdir + '/bbs_nondir.parset'
    f = open(parset, 'w')
    f.writelines(newlines)
    f.close()
    return parset


def makePredictParset(outdir):
    """Makes BBS parset for predict with screen included"""
    newlines = ['Strategy.Stations = []\n',
        'Strategy.ChunkSize = 250\n',
        'Strategy.Steps = [predict]\n',
        'Strategy.Baselines = *& \n',
        'Step.predict.Operation = PREDICT\n',
        'Step.predict.Model.Sources = []\n',
        'Step.predict.Model.Ionosphere.Enable = T\n',
        'Step.predict.Model.Ionosphere.Type = EXPION\n',
        'Step.predict.Model.Beam.Enable = T\n',
        'Step.predict.Model.Beam.Mode = ARRAY_FACTOR\n',
        'Step.predict.Model.Beam.UseChannelFreq = T\n',
        'Step.predict.Model.Cache.Enable = T\n',
        'Step.predict.Output.Column = MODEL_DATA_TEC\n']
    parset = outdir + '/bbs_predict.parset'
    f = open(parset, 'w')
    f.writelines(newlines)
    f.close()
    return parset


def makeGainCalParset(msname, parmdb, solint=1, uvmin=80, freqint=10):
    """Makes an NDPPP parset to perform a dir-independent calibration using
    the MODEL_DATA_TEC column"""
    outdir = '/'.join(msname.split('/')[:-1])

    newlines = ['msin = {0}\n'.format(msname),
        'msout=.\n',
        'msin.modelcolumn = MODEL_DATA_TEC\n',
        'steps = [solve]\n',
        'solve.type = gaincal\n',
        'solve.sourcedb = {0}/sky\n'.format(msname),
        'solve.parmdb = {0}/{1}\n'.format(msname, parmdb),
        'solve.usebeammodel = False\n',
        'solve.maxiter = 50\n',
        'solve.solint = {0}\n'.format(solint),
        'solve.debuglevel = 2\n',
        'solve.tolerance = 1.e-4\n',
        'solve.stefcalvariant = 1c\n',
        'solve.detectstalling = False\n',
        'solve.usemodelcolumn = True\n',
        'solve.caltype = scalarphase\n']
    parset = outdir + '/gaincal_nondir.parset'
    f = open(parset, 'w')
    f.writelines(newlines)
    f.close()
    return parset


def makeCorrectParset(outdir, noTEC=False, beam_mode='DEFAULT'):
    """Makes BBS parset for correct with screen included"""

    # Handle beam
    if beam_mode.lower() == 'off':
        beam_enable = 'F'
        beam_mode = 'DEFAULT'
    else:
        beam_enable = 'T'

    newlines = ['Strategy.Stations = []\n',
        'Strategy.InputColumn = DATA\n',
        'Strategy.ChunkSize = 1500\n',
        'Strategy.UseSolver = F\n']
    if noTEC:
        newlines += ['Strategy.Steps = [correct1]\n']
    else:
        newlines += ['Strategy.Steps = [correct1, correct2]\n']
    newlines += ['Step.correct1.Operation = CORRECT\n',
        'Step.correct1.Model.Sources = []\n',
        'Step.correct1.Model.Beam.Enable = {0}\n'.format(beam_enable),
        'Step.correct1.Model.Beam.Mode = {0}\n'.format(beam_mode),
        'Step.correct1.Model.Beam.UseChannelFreq = T\n',
        'Step.correct1.Model.Gain.Enable = F\n']
    if noTEC:
        newlines += ['Step.correct1.Model.CommonScalarPhase.Enable = F\n',
        'Step.correct1.Output.Column = CORRECTED_DATA_NOTEC\n',
        'Step.correct1.Output.WriteFlags = F']
    else:
        newlines += ['Step.correct1.Model.CommonScalarPhase.Enable = T\n',
        'Step.correct2.Operation = CORRECT\n',
        'Step.correct2.Model.Sources = []\n',
        'Step.correct2.Model.Ionosphere.Enable = T\n',
        'Step.correct2.Model.Ionosphere.Type = EXPION\n',
        'Step.correct2.Output.Column = CORRECTED_DATA\n',
        'Step.correct2.Output.WriteFlags = F']
    if noTEC:
        parset = outdir + '/bbs_correct_notec.parset'
    else:
        parset = outdir + '/bbs_correct_tec.parset'
    f = open(parset, 'w')
    f.writelines(newlines)
    f.close()
    return parset


def calibrateBBS(msname_parmdb):
    """Runs BBS with a phase-only, dir-independent calibration with screen included"""
    msname, parmdb, skymodel, solint, beam = msname_parmdb
    root_dir = '/'.join(msname.split('/')[:-1])

    # Do dir-independent calibration with TEC screen included
    parset = makeNonDirParset(root_dir, solint=int(solint), beam_mode=beam)
    if skymodel == '':
        skymodel = root_dir + '/none'
        os.system("touch {0}".format(skymodel))
        replace_sourcedb = ''
    else:
        replace_sourcedb = '--replace-sourcedb'
    os.system("calibrate-stand-alone {0} --parmdb-name {1} {2} {3} {4} "
            "> {2}_calibrate.log 2>&1".format(replace_sourcedb, parmdb, msname,
            parset, skymodel))


def calibrateNDPPP(msname_parmdb):
    """Runs NDPPP with a phase-only, dir-independent calibration with screen
    included.

    Before running NDPPP, BBS is run to predict the model, corrected for TEC
    screen, and store in MODEL_DATA_TEC column.
    """
    msname, parmdb, skymodel, solint = msname_parmdb
    root_dir = '/'.join(msname.split('/')[:-1])

    # Do a predict in BBS with TEC screen included
    parset = makePredictParset(root_dir)
    if skymodel == '':
        skymodel = root_dir + '/none'
        os.system("touch {0}".format(skymodel))
        replace_sourcedb = ''
    else:
        replace_sourcedb = '--replace-sourcedb'
    os.system("calibrate-stand-alone {0} --parmdb-name {1} {2} {3} {4} "
            "> {2}_predict.log 2>&1".format(replace_sourcedb, parmdb, msname,
            parset, skymodel))

    # Do dir-independent calibration with TEC screen included
    parset = makeGainCalParset(msname, parmdb)
    os.system("NDPPP {0}".format(parset))


if __name__=='__main__':
    import optparse
    opt = optparse.OptionParser(usage='%prog <H5parm:solset>', version='%prog '+_version,
        description=__doc__)
    opt.add_option('-i', '--indir', help='Input directory [default: %default]',
        type='string', default='.')
    opt.add_option('-p', '--parmdb', help='Name of parmdb instument file to use '
        '[default: %default]', type='string', default='instrument')
    opt.add_option('-s', '--solint', help='Solution interval to use '
        'for phase solutions (# of time slots) [default: %default]',
        type='float', default='1')
    opt.add_option('-m', '--model', help='Name of sky model file to use. If no '
        'file is given, the model is taken from the sky parmdb '
        '[default: %default]', type='string', default='')
    opt.add_option('-g', '--gaincal', help='Use NDPPP GainCal for dir-independent '
        'calibration [default: %default]', action='store_true', default=False)
    opt.add_option('-e', '--skipexport', help='Skip export of screen from H5parm '
        '[default: %default]', action='store_true', default=False)
    opt.add_option('-n', '--ncores', help='Maximum number of simultaneous '
        'calibration runs [default: %default]', type='int', default='8')
    opt.add_option('-v', '--verbose', help='Set verbose output and interactive '
        'mode [default: %default]', action='store_true', default=False)
    opt.add_option('-B', '--beam', help='Beam mode to use during peeling. Use OFF '
        'to disable the beam [default: %default]', type='str', default='ARRAY_FACTOR')
    opt.add_option('-c', '--clobber', help='Clobber existing output files? '
        '[default: %default]', action='store_true', default=False)
    opt.add_option('-T', '--torque', help='Use torque? '
        '[default: %default]', action='store_true', default=False)
    opt.add_option('-r', '--resume', help='Try to resume interupted time-correlated '
        'calibration? [default: %default]', action='store_true', default=False)
    (options, args) = opt.parse_args()

    # Get inputs
    if len(args) != 1:
        opt.print_help()
    else:
        try:
            h5, solset = args[0].split(':')
        except ValueError:
            print('H5parm and/or solset name not understood')
            sys.exit()
        h5 = h5.strip()
        solset = solset.strip()

        ms_list = sorted(glob.glob(options.indir+"/*.MS"))
        if len(ms_list) == 0:
            ms_list = sorted(glob.glob(options.indir+"/*.ms"))
        if len(ms_list) == 0:
            ms_list = sorted(glob.glob(options.indir+"/*.ms.peeled"))
        if len(ms_list) == 0:
            ms_list = sorted(glob.glob(options.indir+"/*.MS.peeled"))
        if len(ms_list) == 0:
            print('No measurement sets found in input directory. They must end '
                'in .MS, .ms, .ms.peeled, .MS.peeled')
            sys.exit()

        outdir = options.indir
        if not os.path.isdir(outdir+"/logs"):
            os.mkdir(outdir+"/logs")
        if not os.path.isdir(outdir+"/state"):
            os.mkdir(outdir+"/state")
        if not os.path.isdir(outdir+"/parmdbs"):
            os.mkdir(outdir+"/parmdbs")
        if not os.path.isdir(outdir+"/parsets"):
            os.mkdir(outdir+"/parsets")

        logfilename = options.indir + '/ion_apply.log'
        init_logger(logfilename, debug=options.verbose)
        log = logging.getLogger("Main")

        if not options.skipexport:
            out_parmdb_list = ['ion_{0}'.format(options.parmdb)] * len(ms_list)
            log.info('Exporting screens...')
            for ms, out_parmdb in zip(ms_list, out_parmdb_list):
                # Export screens to parmdbs
                os.system('H5parm_exporter.py {0} {1} -c -r ion -s {2} -i {3} >> {4} '
                    '2>&1'.format(h5, ms, solset, options.parmdb, logfilename))
                log.info('Screens exported to {0}/{1}'.format(ms, out_parmdb))
        else:
            out_parmdb_list = ['{0}'.format(options.parmdb)] * len(ms_list)

        # Calibrate
        log.info('Performing calibration with screens included...')
        skymodel_list = [options.model] * len(ms_list)
        solint_list = [options.solint] * len(ms_list)
        beam_list = [options.beam] * len(ms_list)
        if options.gaincal:
            calibrateNDPPP(ms_list, out_parmdb_list,
                skymodel_list, solint_list)
        else:
            if has_ipy_parallel and options.torque:
#                 lb = loadbalance.LoadBalance(ppn=options.ncores, logfile=None,
#                     loglevel=logging.DEBUG, file_to_source='/home/sttf201/init-lofar.sh')
#                 lb.sync_import('from Ion.ion_libs import *')

                band_list = []
                for ms in ms_list:
                    band_list.append(Band(ms, outdir))
                for i, band in enumerate(band_list):
                    # For each Band instance, set options
                    band.beam_mode = options.beam
                    band.use_timecorr = True
                    band.time_block = 10 # number of time samples in a block
                    band.ionfactor = None
                    band.ncores_per_cal = 6
                    band.resume = options.resume
                    band.init_logger = True
                    band.parmdb = out_parmdb_list[i]
                    band.solint_min = solint_list[i]
                    band.uvmin = 0
                    band.skymodel = skymodel_list[i]
                    chunk_list, chunk_list_orig = apply_band(band)

                    for i, chunk in enumerate(chunk_list):
                        chunk.start_delay = i * 10.0 # start delay in seconds to avoid too much disk IO

                    # Map list of bands to the engines
#                     if len(chunk_list) > 0:
#                         lb.map(run_chunk, chunk_list)

                    # Copy over the solutions to the final output parmdb
                    try:
                        log.info('Copying distributed solutions to output parmdb...')
                        instrument_out = out_parmdb_list[i] + '_total'
                        os.system("rm %s -rf" % instrument_out)
                        pdb_out = lofar.parmdb.parmdb(instrument_out, create=True)
                        for chunk_obj in chunk_list_orig:
                            chunk_instrument = chunk_obj.output_instrument
                            try:
                                pdb_part = lofar.parmdb.parmdb(chunk_instrument)
                            except:
                                continue
                            for parmname in pdb_part.getNames():
                                v = pdb_part.getValuesGrid(parmname)
                                pdb_out.addValues(v)

#                         pdb = lofar.parmdb.parmdb(instrument_orig)
#                         parms = pdb.getValuesGrid("*")
#                         for chunk_obj in chunk_list_orig:
#                             chunk_instrument = chunk_obj.output_instrument
#                             try:
#                                 pdb_part = lofar.parmdb.parmdb(chunk_instrument)
#                             except:
#                                 continue
#                             parms_part = pdb_part.getValuesGrid("*")
#                             keynames = parms_part.keys()
#
#                             # Replace old value with new
#                             for key in keynames:
#                             # Hard-coded to look for Phase and/or TEC parms
#                             # Presumably OK to use other parms with additional 'or' statments
#                                 if 'Phase' in key or 'TEC' in key:
#                                     parms[key]['values'][chunk_obj.solrange, 0] = np.copy(
#                                         parms_part[key]['values'][0:len(chunk_obj.solrange), 0])
#
#                         # Add new values to final output parmdb
#                         pdb_out = lofar.parmdb.parmdb(instrument_out, create=True)
#                         pdb_out.addValues(parms)
                    except Exception as e:
                        log.error(str(e))
            else:
                calibrateBBS((ms_list, out_parmdb_list,
                    skymodel_list, solint_list, beam_list))

        log.info('TEC screen application complete.')

