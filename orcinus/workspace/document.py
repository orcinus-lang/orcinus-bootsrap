# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import io
import os
import urllib.parse
import weakref

from orcinus.core.diagnostics import DiagnosticManager, Diagnostic
from orcinus.language import SyntaxTree, SemanticModel, Module, Parser
from orcinus.language.semantic import SemanticContext
from orcinus.utils import cached_property


class Document:
    def __init__(self, package: Package, uri: str, name: str, source: str = None, version: int = None,
                 diagnostics: DiagnosticManager = None):
        self.__package = weakref.ref(package)
        self.__diagnostics = diagnostics if diagnostics is not None else DiagnosticManager()
        self.__uri = uri
        self.__name = name
        self.__source = source
        self.__version = version
        self.__tree = None
        self.__model = None
        self.__module = None

    @property
    def package(self) -> Package:
        return self.__package()

    @property
    def workspace(self) -> Workspace:
        return self.package.workspace

    @property
    def name(self) -> str:
        """ Return module name """
        return self.__name

    @property
    def uri(self):
        """ Returns uri for source path """
        return self.__uri

    @cached_property
    def path(self) -> str:
        url = urllib.parse.urlparse(self.uri)
        filename = os.path.abspath(url.path)
        return os.path.relpath(filename, self.package.path)

    @property
    def source(self) -> str:
        """ Returns source of document """
        return self.__source

    @source.setter
    def source(self, value: str):
        """ Change source of document """
        self.__source = value
        self.invalidate()

    @property
    def diagnostics(self) -> DiagnosticManager:
        """ Returns diagnostics manager for this document """
        return self.__diagnostics

    @property
    def tree(self) -> SyntaxTree:
        """ Returns syntax tree """
        if not self.__tree:
            parser = Parser(self.uri, io.StringIO(self.source), diagnostics=self.diagnostics)
            self.__tree = parser.parse()
        return self.__tree

    @property
    def model(self) -> SemanticModel:
        """ Returns semantic model """
        if not self.__model:
            try:
                context = SemanticContext(self.workspace, diagnostics=self.diagnostics)
                self.__model = context.open(self)
            except Diagnostic as ex:
                self.diagnostics.add(ex.location, ex.severity, ex.message, ex.source)
            finally:
                self.workspace.on_document_analyze(document=self)
        return self.__model

    @property
    def module(self) -> Module:
        """ Return semantic module for this document """
        if not self.__module:
            self.__module = self.model.module if self.model else None
        return self.__module

    def invalidate(self):
        """ Invalidate document, e.g. detach syntax tree or semantic model from this document """
        self.diagnostics.clear()
        self.__module = None
        self.__model = None
        self.__tree = None

    def __str__(self) -> str:
        return f'{self.package.name}::{self.name} [{self.path}]'

    def __repr__(self):
        class_name = type(self).__name__
        return f'<{class_name}: {self}>'
