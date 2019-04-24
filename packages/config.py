#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-01-24 22:16:37
# @Author  : Raymond Wong (jiabo.huang@qmul.ac.uk)
# @Link    : github.com/Raymond-sci

import os
import argparse
import yaml
import importlib
import traceback
from prettytable import PrettyTable

from .register import REGISTER

class _META_(type):

    PARSER = argparse.ArgumentParser(formatter_class=
                                    argparse.ArgumentDefaultsHelpFormatter)
    ARGS = dict()

    def require_args(self):

        # args for config file
        _META_.PARSER.add_argument('--cfgs', type=str, nargs='*',
                            help='config files to load')

    def parse(self):

        # collect self args
        self.require_args()

        # load default args from config file
        known_args, _ = _META_.PARSER.parse_known_args()
        self.from_files(known_args.cfgs)

        # collect args for packages
        for package in REGISTER.get_packages():
            m = importlib.import_module(package)
            if hasattr(m, 'require_args'):
                m.require_args()

        # re-update default value for new args
        self.from_files(known_args.cfgs)

        # parse args
        _META_.ARGS = _META_.PARSER.parse_args()

    def from_files(self, files):

        # if no config file is provided, skip
        if files is None or len(files) <= 0:
            return None

        for file in files:
            assert os.path.exists(file), "Config file not found: [%s]" % file
            configs = yaml.load(open(file, 'r'))
            _META_.PARSER.set_defaults(**configs)

    def get(self, attr, default=None):
        if hasattr(_META_.ARGS, attr):
            return getattr(_META_.ARGS, attr)
        return default

    def yaml(self):
        config = {k:v for k,v in sorted(vars(_META_.ARGS).items())}
        return yaml.safe_dump(config, default_flow_style=False)

    def __getattr__(self, attr):
        try:
            return _META_.PARSER.__getattribute__(attr)
        except AttributeError:
            return _META_.ARGS.__getattribute__(attr)
        except:
            traceback.print_exec()
            exit(-1)

    def __str__(self):
        MAX_WIDTH = 20
        table = PrettyTable(["#", "Key", "Value", "Default"])
        table.align = 'l'
        for i, (k, v) in enumerate(sorted(vars(_META_.ARGS).items())):
            v = str(v)
            default = str(_META_.PARSER.get_default(k))
            if default == v:
                default = '--'
            table.add_row([i, k, v[:MAX_WIDTH] + ('...' if len(v) > MAX_WIDTH else ''), default])
        return table.get_string()

class CONFIG(object):
    __metaclass__ = _META_