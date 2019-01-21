# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import enum
from collections.abc import Sequence

import attr

from orcinus.core.locations import Location
from orcinus.core.source import show_source_lines
from orcinus.exceptions import OrcinusError


@enum.unique
class DiagnosticSeverity(enum.IntEnum):
    """
    Enumeration contains diagnostic severities

    Attributes:
        Error       - Reports an error.
        Warning     - Reports a warning.
        Information - Reports an information.
        Hint        - Reports a hint.
    """
    Error = 1
    Warning = 2
    Information = 3
    Hint = 4


@attr.attrs(frozen=True, slots=True, auto_attribs=True, str=False)
class Diagnostic(OrcinusError):
    """
    The Diagnostic class is represented a diagnostic, such as a compiler error or warning.

    Attributes:
        location - The location at which the message applies
        severity - The diagnostic's severity.
        message  - The diagnostic's message.
        source   - A human-readable string describing the source of this diagnostic, e.g. 'orcinus' or 'doxygen'.
    """
    location: Location
    severity: DiagnosticSeverity
    message: str
    source: str = "orcinus"

    def __str__(self):
        source = show_source_lines(self.location)
        if source:
            return f"[{self.location}] {self.message}:\n{source}"
        return f"[{self.location}] {self.message}"


class DiagnosticManager(Sequence):
    """
    The DiagnosticManager class is represented collection of diagnostics, and used for simple appending new diagnostic
    """

    def __init__(self):
        self.__diagnostics = []
        self.has_error = False
        self.has_warnings = False

    def __getitem__(self, idx: int) -> Diagnostic:
        return self.__diagnostics[idx]

    def __len__(self) -> int:
        return len(self.__diagnostics)

    def add(self, location: Location, severity: DiagnosticSeverity, message: str, source: str = "orcinus"):
        self.has_error |= severity == DiagnosticSeverity.Error
        self.has_warnings |= severity == DiagnosticSeverity.Warning

        self.__diagnostics.append(
            Diagnostic(location, severity, message, source)
        )

    def error(self, location: Location, message: str, source: str = "orcinus"):
        return self.add(location, DiagnosticSeverity.Error, message, source)

    def warning(self, location: Location, message: str, source: str = "orcinus"):
        return self.add(location, DiagnosticSeverity.Warning, message, source)

    def info(self, location: Location, message: str, source: str = "orcinus"):
        return self.add(location, DiagnosticSeverity.Information, message, source)

    def hint(self, location: Location, message: str, source: str = "orcinus"):
        return self.add(location, DiagnosticSeverity.Hint, message, source)
