# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import collections
import re
from typing import Iterator

from orcinus.core.diagnostics import DiagnosticSeverity, Diagnostic, DiagnosticManager
from orcinus.core.locations import Location
from orcinus.syntax import SyntaxToken, TokenID


class Scanner:
    # This list contains regex for tokens
    TOKENS = [
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenID.Name),
        (r'[0-9_]+', TokenID.Number),
        (r'\(', TokenID.LeftParenthesis),
        (r'\)', TokenID.RightParenthesis),
        (r'\.\.\.', TokenID.Ellipsis),
        (r'\[', TokenID.LeftSquare),
        (r'\]', TokenID.RightSquare),
        (r'\{', TokenID.LeftCurly),
        (r'\}', TokenID.RightCurly),
        (r'\.', TokenID.Dot),
        (r',', TokenID.Comma),
        (r':', TokenID.Colon),
        (r';', TokenID.Semicolon),
        (r'=', TokenID.Equals),
        (r'\*\*', TokenID.DoubleStar),
        (r'\*', TokenID.Star),
        (r'\+', TokenID.Plus),
        (r'\-\>', TokenID.Then),
        (r'\-', TokenID.Minus),
        (r'\/\/', TokenID.DoubleSlash),
        (r'\/', TokenID.Slash),
        (r'\~', TokenID.Tilde),
        (r'\n', TokenID.NewLine),
        (r'\r\n', TokenID.NewLine),
        (r'#[^\r\n]*', TokenID.Comment),
        (r'[ \t]+', TokenID.Whitespace),
    ]

    # This dictionary contains all keywords
    KEYWORDS = {
        'def': TokenID.Def,
        'pass': TokenID.Pass,
        'import': TokenID.Import,
        'from': TokenID.From,
        'as': TokenID.As,
        'return': TokenID.Return,
        'if': TokenID.If,
        'elif': TokenID.Elif,
        'else': TokenID.Else,
        'while': TokenID.While,
        'struct': TokenID.Struct,
        'class': TokenID.Class,
    }

    # Final tuple contains all patterns
    TOKENS = tuple([(re.escape(keyword), token_id) for keyword, token_id in KEYWORDS.items()] + TOKENS)

    # This tuple contains all trivia tokens.
    TRIVIA_TOKENS = (TokenID.Whitespace, TokenID.Comment)
    OPEN_BRACKETS = (TokenID.LeftParenthesis,)
    CLOSE_BRACKETS = (TokenID.RightParenthesis,)

    def __init__(self, filename, stream, *, diagnostics: DiagnosticManager = None):
        self.diagnostics = diagnostics if diagnostics is not None else DiagnosticManager()
        self.index = 0
        self.buffer = stream.read()
        self.length = len(self.buffer)
        self.location = Location(filename)

    # noinspection PyMethodParameters
    def __make_regex(patterns):
        regex_parts = []
        groups = {}

        for idx, (regex, token_id) in enumerate(patterns):
            group_name = f'GROUP{idx}'
            regex_parts.append(f'(?P<{group_name}>{regex})')
            groups[group_name] = token_id

        return re.compile('|'.join(regex_parts)), groups

    # noinspection PyArgumentList
    regex_pattern, regex_groups = __make_regex(TOKENS)

    def tokenize(self) -> Iterator[SyntaxToken]:
        indentions = collections.deque([0])
        is_new = True  # new line
        is_empty = True  # empty line
        whitespace = None
        level = 0  # disable indentation

        for token in self.tokenize_all():
            # new line
            if token.id == TokenID.NewLine:
                if level:
                    continue

                if not is_empty:
                    yield token

                is_new = True
                is_empty = True
                continue

            elif token.id == TokenID.Whitespace:
                if is_new:
                    whitespace = token
                continue

            elif token.id == TokenID.EndFile:
                location = Location(token.location.filename, token.location.end, token.location.end)
                while indentions[-1] > 0:
                    yield SyntaxToken(TokenID.Undent, '', location)
                    indentions.pop()

                yield token
                continue

            elif token.id in self.TRIVIA_TOKENS:
                continue

            if is_new:
                if whitespace:
                    indent = len(whitespace.value)
                    location = whitespace.location
                    whitespace = None
                else:
                    indent = 0
                    location = Location(token.location.filename, token.location.begin, token.location.begin)

                if indentions[-1] < indent:
                    yield SyntaxToken(TokenID.Indent, '', location)
                    indentions.append(indent)

                while indentions[-1] > indent:
                    yield SyntaxToken(TokenID.Undent, '', location)
                    indentions.pop()

            is_new = False
            is_empty = False

            if token.id in self.OPEN_BRACKETS:
                level += 1
            elif token.id in self.CLOSE_BRACKETS:
                level -= 1

            yield token

    def tokenize_all(self) -> Iterator[SyntaxToken]:
        while self.index < self.length:
            yield self.__match()
        yield SyntaxToken(TokenID.EndFile, "", self.location)

    def __match(self):
        self.location.columns(1)
        self.location = self.location.step()

        # other tokens
        match = self.regex_pattern.match(self.buffer, self.index)
        if not match:
            raise Diagnostic(self.location, DiagnosticSeverity.Error, "Unknown symbol")

        group_name = match.lastgroup
        symbol_id = self.regex_groups[group_name]
        value = match.group(group_name)
        self.index += len(value)
        location = self.__consume_location(value)
        return SyntaxToken(symbol_id, value, location)

    def __consume_location(self, value):
        for c in value[:-1]:
            if c == '\n':
                self.location = self.location.lines(1)
            elif len(value) > 1:
                self.location = self.location.columns(1)
        location = self.location
        if value[-1] == '\n':
            self.location = self.location.lines(1)
        else:
            self.location = self.location.columns(1)
        return location

    def __iter__(self):
        return self.tokenize()
