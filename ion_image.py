#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script makes images with and without TEC screens applied

Steps:
  - concatenates MSes for the snapshots
  - images, optionally using PyBDSM to generate clean masks

"""

import pyrap.images
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


def concatenate(msnames, outdir, parmdb):
    """Concatenates all MSes and returns name of concated MS"""

    concat_msname = outdir + "/concatenated.MS"

    for msname in msnames:
        os.system("addImagingColumns.py %s" % msname)

    pyrap.tables.msutil.msconcat(msnames, concat_msname)

    pdb_concat_name = concat_msname + "/ionosphere"
    os.system("rm %s -rf" % pdb_concat_name)
    pdb_concat = lofar.parmdb.parmdb(pdb_concat_name, create=True)

    for msname in msnames:
        pdb = lofar.parmdb.parmdb(msname + "/" + parmdb)
        for parmname in pdb.getNames():
            v = pdb.getValuesGrid(parmname)
            pdb_concat.addValues(v)

    return concat_msname


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
        'threshold=%s wmax=100000 weight=briggs robust=%f '\
        % (msname, imagedir, imageroot, niter, UVmax, cellsize, npix, threshold, robust)
    if logfilename is not None:
        callStr += '>> {0} 2>&1 '.format(logfilename)

    if use_ion:
        callStr += 'applyIonosphere=1 timewindow=10 SpheSupport=45 parmdbname=%s '\
            % (parmdbname)
    if mask_image is not None:
        callStr += 'mask=%s' % mask_image

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
    opt.add_option('-n', '--npix', help='Number of pixels in image '
        '[default: %default]', type='int', default=2048)
    opt.add_option('-p', '--parmdb', help='Name of parmdb instument file to use '
        '[default: %default]', type='string', default='ion_instrument')
    opt.add_option('-u', '--uvmax', help='UVMax in klambda '
        '[default: %default]', type='float', default=2.0)
    opt.add_option('-s', '--size', help='Cellsize in arcsec'
        '[default: %default]', type='float', default=20)
    opt.add_option('-N', '--noscreen', help='Also make image without screen applied? '
        '[default: %default]', action='store_true', default=False)
    opt.add_option('-a', '--automask', help='Auto clean mask iterations; 0 => no auto masking '
        '[default: %default]', type='int', default=0)
    opt.add_option('-m', '--maskfile', help='CASA clean-mask image '
        '[default: %default]', type='string', default='')
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

        # Concatenate data
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
        log.info('Imaging the following data: {0}'.format(ms_list))
        log.info('Concatenating data...')
        msname = concatenate(ms_list, options.outdir, options.parmdb)
        log.info('Concatenated MS is {0}'.format(msname))

        # Define image properties, etc.
        imagedir = options.outdir
        imageroots = ['aprojection']
        use_ions = [True]
        if options.noscreen:
            imageroots.append('original')
            use_ions.append(False)
        UVmax = options.uvmax

        for imageroot, use_ion in zip(imageroots, use_ions):
            log.info('Calling AWimager to make {0} image...'.format(imageroot))
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
                awimager(msname, imageroot, UVmax, options.size, options.npix,
                    options.threshold, mask_image=mask_image, use_ion=use_ion,
                    imagedir=imagedir, logfilename=logfilename, clobber=True,
                    niter=options.iter)
            elif options.maskfile != '':
                mask_image = options.maskfile
                log.info('Using clean mask "{0}"...'.format(mask_image))
                awimager(msname, imageroot, UVmax, options.size, options.npix,
                    options.threshold, mask_image=mask_image, use_ion=use_ion,
                    imagedir=imagedir, logfilename=logfilename, clobber=True,
                    niter=options.iter)
            else:
                awimager(msname, imageroot, UVmax, options.size, options.npix,
                    options.threshold, use_ion=use_ion,
                    imagedir=imagedir, logfilename=logfilename, clobber=True,
                    niter=options.iter)

        log.info('Imaging complete.')
