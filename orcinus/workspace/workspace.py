# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import logging
import os
import urllib.parse
from typing import Sequence, Optional

from orcinus.exceptions import OrcinusError
from orcinus.signals import Signal
from orcinus.workspace.document import Document
from orcinus.workspace.package import Package
from orcinus.workspace.utils import convert_filename

logger = logging.getLogger('orcinus.workspace')


class Workspace:
    """
    Active representation of collection of projects
    """
    packages: Sequence[Package]

    on_document_create: Signal  # (document: Document) -> void
    on_document_remove: Signal  # (document: Document) -> void
    on_document_analyze: Signal  # (document: Document) -> void

    def __init__(self, paths: Sequence[str] = None):
        paths = list(() or paths)

        # Standard library path
        stdlib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../stdlib'))
        if stdlib_path not in paths:
            paths.insert(0, stdlib_path)

        self.packages = [
            Package(self, os.path.abspath(urllib.parse.urlparse(path).path)) for path in paths
        ]

        # signals
        self.on_document_create = Signal()
        self.on_document_remove = Signal()
        self.on_document_analyze = Signal()

    def get_package_for_document(self, doc_uri: str):
        url = urllib.parse.urlparse(doc_uri)
        fullname = os.path.abspath(url.path)
        for package in self.packages:
            if fullname.startswith(package.path):
                return package

        raise OrcinusError(f"Not found file `{url.path}` in packages")

    def get_or_create_document(self, doc_uri: str) -> Document:
        """
        Return a managed document if-present, else create one pointing at disk.
        """
        package = self.get_package_for_document(doc_uri)
        return package.get_or_create_document(doc_uri)

    def get_document(self, doc_uri: str) -> Optional[Document]:
        """ Returns a managed document if-present, otherwise None """
        package = self.get_package_for_document(doc_uri)
        return package.get_document(doc_uri)

    def create_document(self, doc_uri, source=None, version=None) -> Document:
        """ Create new document """
        package = self.get_package_for_document(doc_uri)
        return package.create_document(doc_uri, source, version)

    def update_document(self, doc_uri: str, source=None, version=None) -> Document:
        """ Update source of document """
        package = self.get_package_for_document(doc_uri)
        return package.update_document(doc_uri, source, version)

    def unload_document(self, doc_uri: str):
        """ Unload document from package """
        package = self.get_package_for_document(doc_uri)
        return package.unload_document(doc_uri)

    def load_document(self, module_name: str) -> Document:
        """
        Load document for module

        :param module_name: Module name
        :return: Document
        """
        for package in self.packages:
            doc_uri = convert_filename(module_name, package.path)
            try:
                return package.get_or_create_document(doc_uri)
            except IOError:
                logger.debug(f"Not found module `{module_name}` in file `{doc_uri}`")
                pass  # Continue

        raise OrcinusError(f'Not found module {module_name}')
