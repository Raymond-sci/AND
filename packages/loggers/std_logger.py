#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-11-11 15:08:06
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

import sys
import logging

from ..config import CONFIG as cfg

class ColoredFormatter(logging.Formatter):
    """Colored Formatter for logging module
    
    different logging level will used different color when printing
    
    Extends:
        logging.Formatter
    
    Variables:
        BLACK, RED, GREEN, YELLOW, BLUE, MAGEENTA, CYAN, WHITE {[Number]} -- [default colors]
        RESET_SEQ {str} -- [Sequence end flag]
        COLOR_SEQ {str} -- [color sequence start flag]
        BOLD_SEQ {str} -- [bold sequence start flag]
        COLORS {dict} -- [logging level to color dictionary]
    """

    BLACK, RED, GREEN, YELLOW, BLUE, MAGEENTA, CYAN, WHITE = range(8)
    RESET_SEQ = "\033[0m"
    COLOR_SEQ = "\033[%dm"
    BOLD_SEQ = "\033[1m"
    COLORS = {
        'WARNING': YELLOW,
        'DEBUG': GREEN,
        'CRITICAL': BLUE,
        'ERROR': RED
    }

    def __init__(self, *args, **kwargs):
        logging.Formatter.__init__(self, *args, **kwargs)

    def format(self, record):
        msg = logging.Formatter.format(self, record)
        levelname = record.levelname
        msg_color = msg
        if levelname in ColoredFormatter.COLORS:
            msg_color = (ColoredFormatter.COLOR_SEQ %
                (30 + ColoredFormatter.COLORS[levelname]) + msg +
                ColoredFormatter.RESET_SEQ)
        return msg_color

class STDLogger:
    '''
    static class for logging
    call setup() first to set log level and then call info/debug/error/warn
    to print log msg
    '''

    LOGGER = None

    INFO = logging.INFO
    DEBUG = logging.DEBUG
    C_LEVEL = logging.DEBUG

    FMT_GENERAL = logging.Formatter('[%(levelname)s][%(asctime)s]\t%(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')
    FMT_COLOR = ColoredFormatter('[%(levelname)s][%(asctime)s]\t%(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

    @staticmethod
    def require_args():
        
        cfg.add_argument('--log-file', action='store_true',
                            help='store log to file. (default: False)')

    @staticmethod
    def setup(level=None, to_file=None):
        # if level is not provided, then dont change
        level = level if level is not None else STDLogger.C_LEVEL

        if STDLogger.LOGGER is None:
            STDLogger.LOGGER = logging.getLogger(__name__)
            STDLogger.LOGGER.propagate = False
            # declare and set console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(STDLogger.FMT_COLOR)
            STDLogger.LOGGER.addHandler(console_handler)

        if to_file is not None:
            # declare and set file handler
            STDLogger.LOGGER.debug('Log will be stored in %s' % to_file)
            file_handler = logging.FileHandler(to_file)
            file_handler.setFormatter(STDLogger.FMT_GENERAL)
            STDLogger.LOGGER.addHandler(file_handler)

        STDLogger.LOGGER.setLevel(level)
        STDLogger.C_LEVEL = level

    @staticmethod
    def check():
        if not isinstance(STDLogger.LOGGER, logging.Logger):
            STDLogger.setup()
            # raise ValueError('Call logger.setup(level) to initialize')

    @staticmethod
    def info(*args, **kwargs):
        STDLogger.check()
        STDLogger.erase()
        return STDLogger.LOGGER.info(*args, **kwargs)

    @staticmethod
    def debug(*args, **kwargs):
        STDLogger.check()
        STDLogger.erase()
        return STDLogger.LOGGER.debug(*args, **kwargs)

    @staticmethod
    def error(*args, **kwargs):
        STDLogger.check()
        STDLogger.erase()
        return STDLogger.LOGGER.error(*args, **kwargs)

    @staticmethod
    def warn(*args, **kwargs):
        STDLogger.check()
        STDLogger.erase()
        return STDLogger.LOGGER.warn(*args, **kwargs)

    @staticmethod
    def erase_lines(n=1):
        for _ in xrange(n):
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
        sys.stdout.flush()

    @staticmethod
    def go_up():
        sys.stdout.write('\x1b[1A')
        sys.stdout.flush()

    @staticmethod
    def erase():
        sys.stdout.write('\x1b[2K')
        sys.stdout.flush()

    @staticmethod
    def progress(current, total, msg='processing %d/%d item...'):
        STDLogger.erase()
        print msg % (current, total)
        STDLogger.go_up()

from ..register import REGISTER
REGISTER.set_class(REGISTER.get_package_name(__name__), 'STDLogger', STDLogger)