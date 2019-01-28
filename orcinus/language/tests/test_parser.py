# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from io import StringIO
from typing import Tuple

from orcinus.core.diagnostics import DiagnosticManager
from orcinus.language.parser import Parser, SyntaxTree
from orcinus.language.syntax import SyntaxToken, TokenID, SyntaxSymbol, ImportAST, AliasAST


def parse_string(content) -> Tuple[SyntaxTree, DiagnosticManager]:
    parser = Parser("test", StringIO(content))
    return parser.parse(), parser.diagnostics


def is_token(symbol: SyntaxSymbol, *indices: TokenID) -> bool:
    if isinstance(symbol, SyntaxToken):
        if indices:
            return symbol.id in indices
        return True
    return False


def test_correct_import():
    document, diagnostics = parse_string("import system.io\n")
    assert not diagnostics.has_error
    assert len(document.imports) == 1

    node: ImportAST = document.imports[0]
    assert len(node.aliases) == 1

    alias: AliasAST = node.aliases[0]
    assert alias.name == 'system.io'
    assert len(alias.qualified_name.children) == 3
    assert is_token(alias.qualified_name.children[0], TokenID.Name)
    assert is_token(alias.qualified_name.children[1], TokenID.Dot)
    assert is_token(alias.qualified_name.children[2], TokenID.Name)


def test_malformed_imports():
    document, diagnostics = parse_string("""
import system as
import        as name
    """)

    assert diagnostics.has_error
    assert len(diagnostics) == 2
    assert len(document.imports) == 2

    node: ImportAST = document.imports[0]
    assert len(node.aliases) == 1

    alias: AliasAST = node.aliases[0]
    assert alias.name == 'system'

    node: ImportAST = document.imports[1]
    assert len(node.aliases) == 1

    alias: AliasAST = node.aliases[0]
    assert alias.name == ''
    assert alias.alias == 'name'
