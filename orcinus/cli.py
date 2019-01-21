# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import argparse
import functools
import logging
import os
import sys
from typing import Sequence

from colorlog import ColoredFormatter
from llvmlite import binding

from orcinus import __version__ as version
from orcinus.codegen import ModuleCodegen
from orcinus.core.diagnostics import Diagnostic, DiagnosticSeverity, DiagnosticManager
from orcinus.server.server import LanguageTCPServer
from orcinus.workspace import Workspace

logger = logging.getLogger('orcinus')

# noinspection PyProtectedMember
LEVELS = list(map(str.lower, logging._nameToLevel.keys()))
DEFAULT_LEVEL = "warning"
KEY_ACTION = '__action__'
KEY_LEVEL = '__level__'
KEY_PDB = '__pdb__'

DIAGNOSTIC_LOGGERS = {
    DiagnosticSeverity.Error: logger.error,
    DiagnosticSeverity.Warning: logger.warning,
    DiagnosticSeverity.Information: logger.info,
    DiagnosticSeverity.Hint: logger.info,
}


def log_diagnostic(diagnostic: Diagnostic):
    DIAGNOSTIC_LOGGERS.get(diagnostic.severity, logger.info)(diagnostic)


def log_diagnostics(diagnostics: DiagnosticManager):
    if diagnostics:
        for diagnostic in diagnostics:  # type: Diagnostic
            log_diagnostic(diagnostic)


def exit_diagnostics(diagnostics: DiagnosticManager):
    if diagnostics.has_error:
        sys.exit(1)
    sys.exit(0)


def initialize_logging():
    """ Prepare rules for loggers """
    # Prepare console logger
    console = logging.StreamHandler()

    # Prepare console formatter
    if sys.stderr.isatty():
        formatter = ColoredFormatter(
            '%(reset)s%(message_log_color)s%(message)s',
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red',
            },
            secondary_log_colors={
                'message': {
                    'ERROR': 'red',
                    'CRITICAL': 'red'
                }
            }
        )
    else:
        formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)

    # Setup logging in console
    logger.addHandler(console)


def process_errors(action):
    @functools.wraps(action)
    def wrapper(*args, **kwargs):
        try:
            return action(*args, **kwargs)
        except Diagnostic as ex:
            log_diagnostic(ex)
            return 1
        except Exception as ex:
            logger.exception(ex)
            return 1

    return wrapper


def process_pdb(action):
    @functools.wraps(action)
    def wrapper(*args, **kwargs):
        try:
            import ipdb as pdb
        except ImportError:
            import pdb

        try:
            return action(*args, **kwargs)
        except Exception as ex:
            logger.fatal(ex)
            pdb.post_mortem()
            raise ex

    return wrapper


def build(filenames: Sequence[str]):
    # initialize llvm targets
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmparser()
    binding.initialize_native_asmprinter()

    # initialize workspace context
    workspace = Workspace(paths=[os.getcwd()])
    for filename in filenames:
        document = workspace.get_or_create_document(filename)

        generator = ModuleCodegen(document.model.context, document.name)
        generator.emit(document.module)
        print(generator)


def start_server(hostname, port):
    server = LanguageTCPServer()
    server.listen(hostname, port)


def main():
    # initialize default logging
    initialize_logging()

    # create arguments parser
    parser = argparse.ArgumentParser('Orcinus')
    parser.add_argument('--pdb', dest=KEY_PDB, action='store_true', help="post-mortem mode")
    parser.add_argument('-l', '--level', dest=KEY_LEVEL, choices=LEVELS, default=DEFAULT_LEVEL)
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {version}')

    # create subparser
    subparsers = parser.add_subparsers()

    # build package
    build_cmd = subparsers.add_parser('build')
    build_cmd.add_argument('filenames', type=str, nargs='+', help="files")
    build_cmd.add_argument('--pdb', dest=KEY_PDB, action='store_true', help="post-mortem mode")
    build_cmd.add_argument('-l', '--level', dest=KEY_LEVEL, choices=LEVELS, default=DEFAULT_LEVEL)
    build_cmd.add_argument(dest=KEY_ACTION, help=argparse.SUPPRESS, action='store_const', const=build)

    # add command: Run LSP server
    server_cmd = subparsers.add_parser('server', help='Run server language server protocol')
    server_cmd.add_argument('--pdb', dest=KEY_PDB, action='store_true', help="post-mortem mode")
    server_cmd.add_argument('-l', '--level', dest=KEY_LEVEL, choices=LEVELS, default=DEFAULT_LEVEL)
    server_cmd.add_argument('--hostname', type=str, default='0.0.0.0')
    server_cmd.add_argument('--port', type=int, default=55290)
    server_cmd.add_argument(dest=KEY_ACTION, help=argparse.SUPPRESS, action='store_const', const=start_server)

    # parse arguments
    kwargs = parser.parse_args().__dict__
    action = kwargs.pop(KEY_ACTION, None)
    is_pdb = kwargs.pop(KEY_PDB, False)

    # change logging level
    logger.setLevel(kwargs.pop(KEY_LEVEL, DEFAULT_LEVEL).upper())

    if action:
        if is_pdb:  # enable pdb if required
            action = process_pdb(action)
        if not sys.gettrace():
            action = process_errors(action)
        return action(**kwargs)
    else:
        parser.print_usage()
        parser.exit(2)
