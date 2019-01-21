import urllib.parse
from typing import Optional

from orcinus.core.diagnostics import DiagnosticManager
from orcinus.server.constants import DOCUMENT_PUBLISH_DIAGNOSTICS
from orcinus.server.converters import to_lsp_diagnostic
from orcinus.server.document import Document


class Workspace:
    directories = None
    documents = None

    def __init__(self, connection, directories):
        self.connection = connection
        self.directories = list(directories)
        self.documents = {}

    def get_or_create_document(self, doc_uri: str) -> Document:
        """
        Return a managed document if-present, else create one pointing at disk.
        """
        return self.get_document(doc_uri) or self.create_document(doc_uri)

    def get_document(self, doc_uri: str) -> Optional[Document]:
        """ Returns a managed document if-present, otherwise None """
        return self.documents.get(doc_uri)

    @staticmethod
    def create_document(doc_uri, source=None, version=None) -> Document:
        """ Create new document """
        if source is None:
            url = urllib.parse.urlparse(doc_uri)
            with open(url.path, 'r', encoding='utf-8') as stream:
                source = stream.read()
        return Document(doc_uri, source=source, version=version)

    def append_document(self, doc_uri: str, source=None, version=None) -> Document:
        """ Create """
        document = self.get_document(doc_uri) or self.create_document(doc_uri, source, version)
        document.source = source
        self.documents[doc_uri] = document
        return document

    def remove_document(self, doc_uri: str):
        try:
            del self.documents[doc_uri]
        except KeyError:
            pass

    def publish_diagnostics(self, doc_uri: str, diagnostics: DiagnosticManager):
        diagnostics = tuple(map(to_lsp_diagnostic, diagnostics))
        self.connection.notify(DOCUMENT_PUBLISH_DIAGNOSTICS, params={'uri': doc_uri, 'diagnostics': diagnostics})

    def analyze(self, doc_uri: str):
        if doc_uri in self.documents:
            document = self.documents[doc_uri]
            diagnostics = document.analyze()
            self.publish_diagnostics(doc_uri, diagnostics)
