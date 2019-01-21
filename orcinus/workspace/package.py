# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import os
import urllib.parse
import weakref
from typing import Optional, MutableMapping

from orcinus.exceptions import OrcinusError
from orcinus.workspace.document import Document
from orcinus.workspace.utils import convert_module_name


class Package:
    """ Instance of this class is managed single package """

    def __init__(self, workspace: Workspace, path: str):
        self.__workspace = weakref.ref(workspace)
        self.path = path
        self.documents: MutableMapping[str, Document] = {}

    @property
    def workspace(self) -> Workspace:
        return self.__workspace()

    @property
    def name(self) -> str:
        return os.path.basename(self.path)

    def get_module_name(self, filename):
        fullname = os.path.abspath(filename)
        if fullname.startswith(self.path):
            return convert_module_name(fullname, self.path)

        raise OrcinusError(f"Not found file `{filename}` in packages")

    def get_or_create_document(self, doc_uri: str) -> Document:
        """
        Return a managed document if-present, else create one pointing at disk.
        """
        return self.get_document(doc_uri) or self.create_document(doc_uri)

    def get_document(self, doc_uri: str) -> Optional[Document]:
        """ Returns a managed document if-present, otherwise None """
        return self.documents.get(doc_uri)

    def create_document(self, doc_uri, source=None, version=None) -> Document:
        """ Create new document """
        url = urllib.parse.urlparse(doc_uri)
        name = self.get_module_name(url.path)

        if source is None:
            with open(url.path, 'r', encoding='utf-8') as stream:
                source = stream.read()
        return Document(self, doc_uri, name=name, source=source, version=version)

    def update_document(self, doc_uri: str, source=None, version=None) -> Document:
        """ Update source of document """
        document = self.get_document(doc_uri) or self.create_document(doc_uri, source, version)
        document.source = source
        self.documents[doc_uri] = document
        return document

    def unload_document(self, doc_uri: str):
        """ Unload document from package """
        try:
            del self.documents[doc_uri]
        except KeyError:
            pass

    def __str__(self) -> str:
        return f'{self.name} [{self.path}]'

    def __repr__(self):
        class_name = type(self).__name__
        return f'<{class_name}: {self}>'

# from orcinus.workspace.workspace import Workspace
