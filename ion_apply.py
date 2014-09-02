#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script applies TEC screens generated by LoSoTo's TECFIT and TECSCREEN
operations.

Steps:
  - runs H5parmExporter.py to export TEC screens to parmdbs
  - runs BBS for dir-indep. calibration with screen included
  - clips amplitudes
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


def clip(msnames, station_selection, threshold=750):
    """Clip CORRECTED_DATA amplitudes above threshold and adjust flags"""

    for msname in msnames:
        t = pyrap.tables.table(msname, readonly = False)
        if os.path.exists(msname + '.flags'):
            t_flags = pyrap.tables.table(msname + '.flags')
        else:
            t_flags = t.select("FLAG").copy(msname + '.flags', deep=True)

        t_ant = pyrap.tables.table(msname + '/ANTENNA')
        ant_names = t_ant.getcol('NAME')

        ant1 = t.getcol("ANTENNA1")
        f1 = numpy.array([ant_names[ant] not in station_selection for ant in ant1])

        ant2 = t.getcol("ANTENNA2")
        f2 = numpy.array([ant_names[ant] not in station_selection for ant in ant2])

        f = t_flags.getcol("FLAG")
        f = numpy.logical_or(f, f1[:, numpy.newaxis, numpy.newaxis])
        f = numpy.logical_or(f, f2[:, numpy.newaxis, numpy.newaxis])

        d = t.getcol("CORRECTED_DATA")

        f = numpy.logical_or(f, abs(d)>threshold)
        t.putcol("FLAG", f)
        t.flush()


def makeNonDirParset(outdir):
    """Makes BBS parset for dir-indep calibration with screen included"""
    newlines = ['Strategy.Stations = []\n',
        'Strategy.InputColumn = DATA\n',
        'Strategy.ChunkSize = 1500\n',
        'Strategy.UseSolver = F\n',
        'Strategy.Steps = [solve]\n',
        'Strategy.Baselines = *& \n',
        'Step.solve.Operation = SOLVE\n',
        'Step.solve.Model.Sources = []\n',
        'Step.solve.Model.Ionosphere.Enable = T\n',
        'Step.solve.Model.Ionosphere.Type = EXPION\n',
        'Step.solve.Model.Beam.Enable = T\n',
        'Step.solve.Model.Beam.Mode = ARRAY_FACTOR\n',
        'Step.solve.Model.Beam.UseChannelFreq = T\n',
        'Step.solve.Model.Cache.Enable = T\n',
        'Step.solve.Model.Gain.Enable = F\n',
        'Step.solve.Model.CommonScalarPhase.Enable= T\n',
        'Step.solve.Solve.Mode = COMPLEX\n',
        'Step.solve.Solve.UVRange = [80]\n',
        'Step.solve.Solve.Parms = ["CommonScalarPhase:*"]\n',
        'Step.solve.Solve.CellSize.Freq = 10\n',
        'Step.solve.Solve.CellSize.Time = 5\n',
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
        'Strategy.ChunkSize = 1500\n',
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


def makeGainCalParset(msname, parmdb):
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
        'solve.solint = 1\n',
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


def makeCorrectParset(outdir, noTEC=False):
    """Makes BBS parset for correct with screen included"""
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
        'Step.correct1.Model.Beam.Enable = T\n',
        'Step.correct1.Model.Beam.Mode = ARRAY_FACTOR\n',
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
    msname, parmdb, skymodel = msname_parmdb
    root_dir = '/'.join(msname.split('/')[:-1])

    # Do dir-independent calibration with TEC screen included
    parset = makeNonDirParset(root_dir)
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

    Before running NDPPP, BBS is run to predict model, corrected for TEC
    screen, and store in MODEL_DATA_TEC column.
    """
    msname, parmdb, skymodel = msname_parmdb
    root_dir = '/'.join(msname.split('/')[:-1])

    # Do a predict in BBS with TEC screen included
    parset = makePredictParset(msname)
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
#    os.system("NDPPP {0}".format(parset))


def applyTEC(msname_parmdb):
    """Applies dir-independent calibration, beam and TEC screen at phase
    center to CORRECTED_DATA"""
    msname, parmdb, skymodel = msname_parmdb
    root_dir = '/'.join(msname.split('/')[:-1])

    parset = makeCorrectParset(root_dir)
    skymodel = root_dir + '/none'
    os.system("touch {0}".format(skymodel))
    os.system("calibrate-stand-alone --no-columns --parmdb-name {0} {1} {2} {3} "
            "> {1}_apply.log 2>&1".format(parmdb, msname, parset, skymodel))


def applyNoTEC(msname_parmdb):
    """Applies beam at phase center to CORRECTED_DATA_NOTEC"""
    msname, parmdb, skymodel = msname_parmdb
    root_dir = '/'.join(msname.split('/')[:-1])

    parset = makeCorrectParset(root_dir, noTEC=True)
    skymodel = root_dir + '/none'
    os.system("touch {0}".format(skymodel))
    os.system("calibrate-stand-alone --no-columns --parmdb-name {0} {1} {2} {3} "
            "> {1}_apply.log 2>&1".format(parmdb, msname, parset, skymodel))


if __name__=='__main__':
    import optparse
    opt = optparse.OptionParser(usage='%prog <H5parm:solset>', version='%prog '+_version,
        description=__doc__)
    opt.add_option('-i', '--indir', help='Input directory [default: %default]',
        type='string', default='.')
    opt.add_option('-p', '--parmdb', help='Name of parmdb instument file to use '
        '[default: %default]', type='string', default='instrument')
    opt.add_option('-s', '--skymodel', help='Name of sky model file to use. If no '
        'file is given, the model is taken from the sky parmdb '
        '[default: %default]', type='string', default='')
    opt.add_option('-b', '--bbs', help='Use BBS for dir-independent '
        'calibration [default: %default]', action='store_true', default=False)
    opt.add_option('-e', '--skipexport', help='Skip export of screen from H5parm '
        '[default: %default]', action='store_true', default=False)
    opt.add_option('-C', '--skipclip', help='Skip clipping on CORRECTED_DATA amplitudes '
        '[default: %default]', action='store_true', default=False)
    opt.add_option('-t', '--threshold', help='Clipping threshold in Jy '
        '[default: %default]', type='float', default=700.0)
    opt.add_option('-n', '--ncores', help='Maximum number of simultaneous '
        'calibration runs [default: %default]', type='int', default='8')
    opt.add_option('-v', '--verbose', help='Set verbose output and interactive '
        'mode [default: %default]', action='store_true', default=False)
    opt.add_option('-c', '--clobber', help='Clobber existing output files? '
        '[default: %default]', action='store_true', default=False)
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
        log.info('Calibrating and applying screens...')
        skymodel_list = [options.skymodel] * len(ms_list)
        workers = Pool(processes=min(len(ms_list), options.ncores))
        if options.bbs:
            workers.map(calibrateBBS, zip(ms_list, out_parmdb_list, skymodel_list))
        else:
            workers.map(calibrateNDPPP, zip(ms_list, out_parmdb_list, skymodel_list))
        workers.map(applyTEC, zip(ms_list, out_parmdb_list, skymodel_list))
        workers.map(applyNoTEC, zip(ms_list, out_parmdb_list, skymodel_list))

        # Clip high data amplitudes
        if not options.skipclip:
            H = h5parm(h5)
            station_selection = H.getAnt(solset).keys()
            log.info('Clipping CORRECTED_DATA amplitudes at {0} Jy...'.format(options.threshold))
            clip(ms_list, station_selection, threshold=options.threshold)

        log.info('TEC screen application complete.')

