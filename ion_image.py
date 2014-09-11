#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script makes images with and without TEC screens applied

Steps:
  - concatenates MSes for the snapshots
  - images, optionally using PyBDSM to generate clean masks

"""

import os
import sys
import pyrap.tables
import lofar.parmdb
import glob
import subprocess
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

def makeCorrectParset(outdir, noTEC=False):
    """Makes BBS parset to correct DATA at phase center for various effects"""
    newlines = ['Strategy.Stations = []\n',
        'Strategy.InputColumn = DATA\n',
        'Strategy.ChunkSize = 250\n',
        'Strategy.UseSolver = F\n']
    if noTEC:
        newlines += ['Strategy.Steps = [correct_beam]\n',
        'Step.correct_beam.Operation = CORRECT\n',
        'Step.correct_beam.Model.Sources = []\n',
        'Step.correct_beam.Model.Beam.Enable = T\n',
        'Step.correct_beam.Model.Beam.UseChannelFreq = T\n',
        'Step.correct_beam.Output.Column = CORRECTED_DATA\n',
        'Step.correct_beam.Output.WriteFlags = F']
    else:
        newlines += ['Strategy.Steps = [correct_beam_phase, correct_screen]\n',
        'Step.correct_beam_phase.Operation = CORRECT\n',
        'Step.correct_beam_phase.Model.Sources = []\n',
        'Step.correct_beam_phase.Model.Beam.Enable = T\n',
        'Step.correct_beam_phase.Model.Beam.Mode = ARRAY_FACTOR\n',
        'Step.correct_beam_phase.Model.Beam.UseChannelFreq = T\n',
        'Step.correct_beam_phase.Model.Gain.Enable = F\n',
        'Step.correct_beam_phase.Model.CommonScalarPhase.Enable = T\n',
        'Step.correct_screen.Operation = CORRECT\n',
        'Step.correct_screen.Model.Sources = []\n',
        'Step.correct_screen.Model.Ionosphere.Enable = T\n',
        'Step.correct_screen.Model.Ionosphere.Type = EXPION\n',
        'Step.correct_screen.Output.Column = CORRECTED_DATA\n',
        'Step.correct_screen.Output.WriteFlags = F']
    if noTEC:
        parset = outdir + '/bbs_correct_notec.parset'
    else:
        parset = outdir + '/bbs_correct_tec.parset'
    f = open(parset, 'w')
    f.writelines(newlines)
    f.close()
    return parset


def apply(msnames, parmdb, noTEC=False, logfilename='apply.log'):
    """Applies beam or dir-independent calibration, beam, and TEC screen at phase
    center to CORRECTED_DATA"""
    root_dir = '/'.join(msname.split('/')[:-1])
    parset = makeCorrectParset(root_dir, noTEC)
    skymodel = root_dir + '/none'
    os.system("touch {0}".format(skymodel))
    for msname in msnames:
        os.system("calibrate-stand-alone --no-columns --parmdb-name {0} {1} {2} {3} "
                "> {4} 2>&1".format(parmdb, msname, parset, skymodel, logfilename))


def clip(msnames, station_selection=None, threshold=750):
    """Clip CORRECTED_DATA amplitudes above threshold and adjust flags"""

    for msname in msnames:
        t = pyrap.tables.table(msname, readonly = False)
        if os.path.exists(msname + '.flags'):
            t_flags = pyrap.tables.table(msname + '.flags')
        else:
            t_flags = t.select("FLAG").copy(msname + '.flags', deep=True)

        t_ant = pyrap.tables.table(msname + '/ANTENNA')
        ant_names = t_ant.getcol('NAME')
        if station_selection is None:
            station_selection = ant_names

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


def concatenate(msnames, outdir, parmdb, noscreen=False, logfilename=None,
    cliplevel=750.0):
    """Coorrects, clips, and concatenates the input MSes"""
    noTEClist = [False]
    if noscreen:
        noTEClist.append(True)

    for noTEC in noTEClist:
        # Correct the DATA column for beam, phase, and screen in phase center
        # and write to CORRECTED_DATA column
        apply(msnames, parmdb, noTEC=noTEC, logfilename=logfilename)
        clip(msnames, threshold=cliplevel)

        # Concatenate
        if noTEC:
            concat_msname = outdir + "/concat_noscreen.ms"
        else:
            concat_msname = outdir + "/concat_screen.ms"
        pyrap.tables.msutil.msconcat(msnames, concat_msname)

        # Copy over parmdbs to concatenated MS
        pdb_concat_name = newmsname + "/ionosphere"
        os.system("rm %s -rf" % pdb_concat_name)
        pdb_concat = lofar.parmdb.parmdb(pdb_concat_name, create=True)
        for msname in msnames:
            pdb = lofar.parmdb.parmdb(msname + "/" + parmdb)
            for parmname in pdb.getNames():
                v = pdb.getValuesGrid(parmname)
                pdb_concat.addValues(v)


def createMask(msfile, skymodel, npix, cellsize, filename=None, logfilename=None):
    """Creates a CASA mask from an input sky model"""
    import lsmtool
    import pyrap.images as pi
    import numpy as np
    import re

    pad = 250. # increment in maj/min axes [arcsec]

    # Make blank mask image
    if filename is None:
        mask_file = "{0}_{1}pix_{2}cellsize.mask".format(skymodel, npix, cellsize)
    else:
        mask_file = filename
    mask_command = "awimager ms={0} image={1} operation=empty stokes='IQUV' "\
        "npix={2} cellsize={3}arcsec".format(msfile, mask_file, npix, cellsize)
    if logfilename is not None:
        mask_command += '>> {0} 2>&1 '.format(logfilename)

    subprocess.call(mask_command, shell=True)
    catalogue = skymodel

    # Open mask
    mask = pi.image(mask_file, overwrite = True)
    mask_data = mask.getdata()
    xlen, ylen = mask.shape()[2:]
    freq, stokes, null, null = mask.toworld([0, 0, 0, 0])

    # Open the skymodel
    lsm = lsmtool.load(skymodel)
    source_list = lsm.getColValues("Name")
    source_type = lsm.getColValues("Type")
    source_ra = lsm.getColValues("Ra", units='radian')
    source_dec = lsm.getColValues("Dec", units='radian')

    # Loop over the sources
    for source, srctype, ra, dec in zip(source_list, source_type, source_ra, source_dec):
        srcindx = lsm.getRowIndex(source)
        if srctype.lower() == 'guassian':
            maj_raw = lsm.getColValues('MajorAxis', units='radian')[srcindx]
            min_raw = lsm.getColValues("MinorAxis", units='radian')[srcindx]
            pa_raw = lsm.getColValues("Orientation", units='radian')[srcindx]
            if maj == 0 or min == 0: # wenss writes always 'GAUSSIAN' even for point sources -> set to wenss beam+pad
                maj = ((54. + pad) / 3600.) * np.pi / 180.
                min = ((54. + pad) / 3600.) * np.pi / 180.
        elif srctype.lower() == 'point': # set to wenss beam+pad
            maj = (((54. + pad) / 2.) / 3600.) * np.pi / 180.
            min = (((54. + pad) / 2.) / 3600.) * np.pi / 180.
            pa = 0.
        else:
            continue

        # define a small square around the source to look for it
        null, null, y1, x1 = mask.topixel([freq, stokes, dec - maj, ra - maj /
            np.cos(dec - maj)])
        null, null, y2, x2 = mask.topixel([freq, stokes, dec + maj, ra + maj /
            np.cos(dec + maj)])
        xmin = np.int(np.floor(np.min([x1, x2])))
        xmax = np.int(np.ceil(np.max([x1, x2])))
        ymin = np.int(np.floor(np.min([y1, y2])))
        ymax = np.int(np.ceil(np.max([y1, y2])))

        if xmin > xlen or ymin > ylen or xmax < 0 or ymax < 0:
            continue

        for x in xrange(xmin, xmax):
            for y in xrange(ymin, ymax):
                # skip pixels outside the mask field
                if x >= xlen or y >= ylen or x < 0 or y < 0:
                    continue
                # get pixel ra and dec in rad
                null, null, pix_dec, pix_ra = mask.toworld([0, 0, y, x])

                X = (pix_ra - ra) * np.sin(pa) + (pix_dec - dec) * np.cos(pa); # Translate and rotate coords.
                Y = -(pix_ra - ra) * np.cos(pa) + (pix_dec - dec) * np.sin(pa); # to align with ellipse
                if X ** 2 / maj ** 2 + Y ** 2 / min ** 2 < 1:
                    mask_data[0, 0, y, x] = 1

    mask.putdata(mask_data)
    return mask_file


def awimager(msname, imageroot, UVmax, cellsize, npix, threshold, mask_image=None,
    parmdbname='ionosphere', robust=0, use_ion=False, imagedir='.', clobber=False,
    logfilename=None, niter=1000000):
    """Calls the AWimager"""

    if clobber:
        os.system('rm %s/%s.residual -rf' % (imagedir, imageroot))
        os.system('rm %s/%s.residual.corr -rf' % (imagedir, imageroot))
        os.system('rm %s/%s.restored -rf' % (imagedir, imageroot))
        os.system('rm %s/%s.restored.corr -rf' % (imagedir, imageroot))
        os.system('rm %s/%s.model -rf'  % (imagedir, imageroot))
        os.system('rm %s/%s.model.corr -rf'  % (imagedir, imageroot))
        os.system('rm %s/%s.psf -rf'  % (imagedir, imageroot))
        os.system('rm %s/%s0.* -rf'  % (imagedir, imageroot))

    cellsize = '{0}arcsec'.format(cellsize)
    npix = str(npix)
    threshold = '{0}Jy'.format(threshold)

    callStr = 'awimager ms=%s data=CORRECTED_DATA image=%s/%s '\
        'operation=mfclark niter=%s UVmax=%f cellsize=%s npix=%s '\
        'threshold=%s weight=briggs robust=%f '\
        % (msname, imagedir, imageroot, niter, UVmax, cellsize, npix,
        threshold, robust)

    if use_ion:
        callStr += 'applyIonosphere=1 timewindow=10 SpheSupport=45 parmdbname=%s '\
            % (parmdbname)
    if mask_image is not None:
        callStr += 'mask=%s' % mask_image
    if logfilename is not None:
        callStr += '>> {0} 2>&1 '.format(logfilename)

    os.system(callStr)


if __name__=='__main__':
    import optparse
    opt = optparse.OptionParser(usage='%prog', version='%prog '+_version,
        description=__doc__)
    opt.add_option('-i', '--indir', help='Input directory [default: %default]',
        type='string', default='.')
    opt.add_option('-o', '--outdir', help='Output directory [default: %default]',
        type='string', default='.')
    opt.add_option('-t', '--threshold', help='Clean threshold in Jy '
        '[default: %default]', type='float', default=0.5)
    opt.add_option('-C', '--clip', help='Clipping threshold in Jy '
        '[default: %default]', type='float', default=700.0)
    opt.add_option('-n', '--npix', help='Number of pixels in image '
        '[default: %default]', type='int', default=2048)
    opt.add_option('-p', '--parmdb', help='Name of parmdb instument file to use '
        '[default: %default]', type='string', default='ion_instrument')
    opt.add_option('-u', '--uvmax', help='UVMax in klambda '
        '[default: %default]', type='float', default=2.0)
    opt.add_option('-s', '--size', help='Cellsize in arcsec'
        '[default: %default]', type='float', default=20)
    opt.add_option('-N', '--noscreen', help='Also make image without screen '
        'applied? [default: %default]', action='store_true', default=False)
    opt.add_option('-a', '--automask', help='Auto clean mask iterations; '
        '0 => no auto masking [default: %default]', type='int', default=0)
    opt.add_option('-m', '--maskfile', help='CASA clean-mask image or sky model '
        'from which a mask will be made [default: %default]', type='string',
        default='')
    opt.add_option('-I', '--iter', help='Number of iterations to do '
        '[default: %default]', type='int', default=100000)
    opt.add_option('-v', '--verbose', help='Set verbose output and interactive '
        'mode [default: %default]', action='store_true', default=False)
    opt.add_option('-c', '--clobber', help='Clobber existing output files? '
        '[default: %default]', action='store_true', default=False)
    (options, args) = opt.parse_args()

    # Get inputs
    if len(args) != 0:
        opt.print_help()
    else:
        ms_list = sorted(glob.glob(options.indir+"/*.MS"))
        if len(ms_list) == 0:
            ms_list = sorted(glob.glob(options.indir+"/*.ms"))
        if len(ms_list) == 0:
            ms_list = sorted(glob.glob(options.indir+"/*.ms.peeled"))
        if len(ms_list) == 0:
            ms_list = sorted(glob.glob(options.indir+"/*.MS.peeled"))
        if len(ms_list) == 0:
            print('No measurement sets found in input directory. They must end '
                'in .MS, .ms, .ms.peeled, or .MS.peeled')
            sys.exit()

        if not os.path.isdir(options.outdir):
            os.mkdir(options.outdir)
        elif options.clobber:
            subprocess.call("rm -rf {0}".format(options.outdir), shell=True)
            os.mkdir(options.outdir)
        else:
            print("The output directory already exists! Please\n"
                "rename/move/delete it, or set the clobber (-c) flag.")
            sys.exit()
        logfilename = options.outdir + '/ion_image.log'
        init_logger(logfilename, debug=options.verbose)
        log = logging.getLogger("Main")
        log.info('Imaging the following MS(es): {0}'.format(ms_list))
        msnames = [options.outdir + "/to_image_screen.ms"]
        if options.noscreen:
            msnames.append(options.outdir + "/to_image_noscreen.ms")

        # Concatenate
        log.info('Copying and concatenating MS(es) (if needed)...')
        concatenate(ms_list, options.outdir, options.parmdb, options.noscreen,
            logfilename=logfilename, cliplevel=options.clip)
        log.info('Final MS(es) to be imaged: {0}'.format(msnames))

        # Define image properties, etc.
        imagedir = options.outdir
        imageroots = ['aprojection']
        use_ions = [True]
        if options.noscreen:
            imageroots.append('original')
            use_ions.append(False)
        UVmax = options.uvmax

        for msname, imageroot, use_ion in zip(msnames, imageroots, use_ions):
            if options.automask > 0:
                from lofar import bdsm
                mask_image = imagedir + '/' + imageroot + '.mask'
                if not os.path.exists(mask_image):
                    # Mask does not exist, need to make it
                    log.info('Generating clean mask "{0}"...'.format(mask_image))
                    for i in range(options.iter):
                        awimager(msname, imageroot, UVmax, options.size, options.npix,
                            options.threshold, clobber=options.clobber, use_ion=use_ion,
                            imagedir=imagedir, logfilename=logfilename, niter=10)
                        img = bdsm.process_image(imagedir+'/'+imageroot+'.restored',
                            blank_limit=1e-4, stop_at='isl', thresh_pix=6,
                            thresh_isl=4)
                        img.export_image(outfile=mask_image, img_type='island_mask',
                            img_format='casa', mask_dilation=2, clobber=True)
                        img.export_image(outfile=mask_image+str(i), img_type='island_mask',
                            img_format='casa', mask_dilation=2, clobber=True)
                log.info('Calling AWimager to make {0} image...'.format(imageroot))
                awimager(msname, imageroot, UVmax, options.size, options.npix,
                    options.threshold, mask_image=mask_image, use_ion=use_ion,
                    imagedir=imagedir, logfilename=logfilename, clobber=True,
                    niter=options.iter)
            elif options.maskfile != '':
                if os.path.isdir(options.maskfile):
                    mask_image = options.maskfile
                elif os.path.exists(options.maskfile):
                    mask_image = imagedir + '/' + imageroot + '.mask'
                    log.info('Generating clean mask "{0}"...'.format(mask_image))
                    mask_image = createMask(msname, options.maskfile, options.npix,
                        options.size, filename=mask_image, logfilename=logfilename)
                else:
                    print('The specified mask file "{0}" was not found.'.format(
                        options.maskfile))
                    sys.exit()

                log.info('Using clean mask "{0}"...'.format(mask_image))
                log.info('Calling AWimager to make {0} image...'.format(imageroot))
                awimager(msname, imageroot, UVmax, options.size, options.npix,
                    options.threshold, mask_image=mask_image, use_ion=use_ion,
                    imagedir=imagedir, logfilename=logfilename, clobber=True,
                    niter=options.iter)
            else:
                log.info('Calling AWimager to make {0} image...'.format(imageroot))
                awimager(msname, imageroot, UVmax, options.size, options.npix,
                    options.threshold, use_ion=use_ion,
                    imagedir=imagedir, logfilename=logfilename, clobber=True,
                    niter=options.iter)

        log.info('Imaging complete.')
