# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from orcinus.core.diagnostics import DiagnosticManager
from orcinus.parser import Parser
from orcinus.semantic import SemanticModel, SemanticContext
from orcinus.syntax import ModuleAST


class Document:
    def __init__(self, uri: str, source: str = None, version: int = None, diagnostics: DiagnosticManager = None):
        self.__diagnostics = diagnostics if diagnostics is not None else DiagnosticManager()
        self.__uri = uri
        self.__source = source
        self.__version = version
        self.__tree = None
        self.__model = None

    @property
    def uri(self):
        return self.__uri

    @property
    def source(self) -> str:
        return self.__source

    @source.setter
    def source(self, value: str):
        self.__source = value
        self.__tree = None
        self.__model = None

    @property
    def diagnostics(self) -> DiagnosticManager:
        return self.__diagnostics

    @property
    def tree(self) -> ModuleAST:
        if not self.__tree:
            self.__tree = self.__parse()
        return self.__tree

    @property
    def model(self) -> SemanticModel:
        if not self.__model:
            self.__model = self.__analyze()
        return self.__model

    def __parse(self) -> ModuleAST:
        """ Parse source to syntax tree """
        parser = Parser(self.uri, self.source, diagnostics=self.diagnostics)
        return parser.parse()

    def __analyze(self) -> SemanticModel:
        """ Analyze syntax tree and returns semantic model """
        context = SemanticContext(diagnostics=self.diagnostics)
        return context.load(self.uri, self.source)

    def analyze(self):
        module = self.model
        return module.diagnostics
