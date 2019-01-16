#!/usr/bin/env python
# Copyright (C) 2018 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import abc
import argparse
import collections
import dataclasses
import enum
import functools
import heapq
import io
import itertools
import logging
import os
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence, Iterator, Optional, cast, Mapping, Tuple

from colorlog import ColoredFormatter
from llvmlite import binding
from llvmlite import ir
from multidict import MultiDict
from multimethod import multimethod

APPLICATION_NAME = 'bootstrap'
ANSI_COLOR_RED = "\033[31m" if sys.stderr.isatty() else ""
ANSI_COLOR_GREEN = "\x1b[32m" if sys.stderr.isatty() else ""
ANSI_COLOR_YELLOW = "\x1b[33m" if sys.stderr.isatty() else ""
ANSI_COLOR_BLUE = "\x1b[34m" if sys.stderr.isatty() else ""
ANSI_COLOR_MAGENTA = "\x1b[35m" if sys.stderr.isatty() else ""
ANSI_COLOR_CYAN = "\x1b[36m" if sys.stderr.isatty() else ""
ANSI_COLOR_RESET = "\x1b[0m" if sys.stderr.isatty() else ""

logger = logging.getLogger(APPLICATION_NAME)


# noinspection PyPep8Naming
class cached_property(object):
    def __init__(self, func):
        self.func = func

    def __get__(self, instance, cls=None):
        if instance:
            result = instance.__dict__[self.func.__name__] = self.func(instance)
            return result
        return None  # ABC


class BootstrapError(Exception):
    pass


@enum.unique
class TokenID(enum.IntEnum):
    Name = enum.auto()
    Number = enum.auto()
    LeftParenthesis = enum.auto()
    RightParenthesis = enum.auto()
    LeftSquare = enum.auto()
    RightSquare = enum.auto()
    LeftCurly = enum.auto()
    RightCurly = enum.auto()
    Dot = enum.auto()
    Comma = enum.auto()
    Colon = enum.auto()
    Semicolon = enum.auto()
    Comment = enum.auto()
    Whitespace = enum.auto()
    NewLine = enum.auto()
    EndFile = enum.auto()
    Indent = enum.auto()
    Undent = enum.auto()
    Def = enum.auto()
    Pass = enum.auto()
    Import = enum.auto()
    From = enum.auto()
    Return = enum.auto()
    As = enum.auto()
    Then = enum.auto()
    Ellipsis = enum.auto()
    If = enum.auto()
    Elif = enum.auto()
    Else = enum.auto()
    While = enum.auto()
    Struct = enum.auto()
    Class = enum.auto()
    Equals = enum.auto()
    Star = enum.auto()
    DoubleStar = enum.auto()
    Plus = enum.auto()
    Minus = enum.auto()
    Slash = enum.auto()
    DoubleSlash = enum.auto()
    Tilde = enum.auto()
    String = enum.auto()


@dataclass
class Token:
    id: TokenID
    value: str
    location: Location

    def __str__(self) -> str:
        value = self.value.strip()
        if value:
            return f'{self.id.name} [{value}]'
        return self.id.name

    def __repr__(self):
        return str(self)


@dataclass(order=True, unsafe_hash=True, frozen=True)
class Position:
    # Line position in a document (one-based).
    line: int = 1

    # Character offset on a line in a document (one-based).
    column: int = 1

    @staticmethod
    def __add(lhs: int, rhs: int, min_v: int) -> int:
        """Compute max(min_v, lhs+rhs) (provided min_v <= lhs)."""
        return rhs + lhs if 0 < rhs or -rhs < lhs else min_v

    def lines(self, count: int = 1) -> Position:
        """(line related) Advance to the COUNT next lines."""
        if count:
            line = self.__add(self.line, count, 1)
            return Position(line, 1)
        return self

    def columns(self, count: int = 1) -> Position:
        """(column related) Advance to the COUNT next columns."""
        column = self.__add(self.column, count, 1)
        return Position(self.line, column)

    def __str__(self):
        return f"{self.line}:{self.column}"


@dataclass(unsafe_hash=True, frozen=True)
class Location:
    # The location's filename
    filename: str

    # The location's begin position.
    begin: Position = Position()

    # The end's begin position.
    end: Position = Position()

    def step(self) -> Location:
        """Reset initial location to final location."""
        return Location(self.filename, self.end, self.end)

    def columns(self, count: int = 1) -> Location:
        """Extend the current location to the COUNT next columns."""
        end = self.end.columns(count)
        return Location(self.filename, self.begin, end)

    def lines(self, count: int = 1) -> Location:
        """Extend the current location to the COUNT next lines."""
        end = self.end.lines(count)
        return Location(self.filename, self.begin, end)

    def __add__(self, other: Location) -> Location:
        return Location(self.filename, self.begin, other.end)

    def __str__(self):
        if self.begin == self.end:
            return f"{self.filename}:{self.begin}"
        elif self.begin.line == self.end.line:
            return f"{self.filename}:{self.begin}-{self.end.column}"
        else:
            return f"{self.filename}:{self.begin}-{self.end}"


@enum.unique
class DiagnosticSeverity(enum.IntEnum):
    """ Enumeration contains diagnostic severities """
    # Reports an error.
    Error = 1

    # Reports a warning.
    Warning = 2

    # Reports an information.
    Information = 3

    # Reports a hint.
    Hint = 4


# The Diagnostic class is represented a diagnostic, such as a compiler error or warning.
@dataclass(frozen=True)
class Diagnostic(BootstrapError):
    # The location at which the message applies
    location: Location

    # The diagnostic's severity.
    severity: DiagnosticSeverity

    # The diagnostic's message.
    message: str

    # A human-readable string describing the source of this diagnostic, e.g. 'orcinus' or 'doxygen'.
    source: str = APPLICATION_NAME

    def __str__(self):
        source = show_source_lines(self.location)
        if source:
            return f"[{self.location}] {self.message}:\n{source}"
        return f"[{self.location}] {self.message}"


class DiagnosticManager(collections.abc.Sequence):
    def __init__(self):
        self.__diagnostics = []
        self.has_error = False
        self.has_warnings = False

    def __getitem__(self, idx: int) -> Diagnostic:
        return self.__diagnostics[idx]

    def __len__(self) -> int:
        return len(self.__diagnostics)

    def add(self, location: Location, severity: DiagnosticSeverity, message: str, source: str = APPLICATION_NAME):
        self.has_error |= severity == DiagnosticSeverity.Error
        self.has_warnings |= severity == DiagnosticSeverity.Warning

        self.__diagnostics.append(
            Diagnostic(location, severity, message, source)
        )

    def error(self, location: Location, message: str, source: str = APPLICATION_NAME):
        return self.add(location, DiagnosticSeverity.Error, message, source)

    def warning(self, location: Location, message: str, source: str = APPLICATION_NAME):
        return self.add(location, DiagnosticSeverity.Warning, message, source)

    def info(self, location: Location, message: str, source: str = APPLICATION_NAME):
        return self.add(location, DiagnosticSeverity.Information, message, source)

    def hint(self, location: Location, message: str, source: str = APPLICATION_NAME):
        return self.add(location, DiagnosticSeverity.Hint, message, source)


class Scanner:
    # This list contains regex for tokens
    TOKENS = [
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenID.Name),
        (r'[0-9_]+', TokenID.Number),

        (r'\(', TokenID.LeftParenthesis),
        (r'\)', TokenID.RightParenthesis),
        (r'\.\.\.', TokenID.Ellipsis),
        ('\[', TokenID.LeftSquare),
        ('\]', TokenID.RightSquare),
        ('\{', TokenID.LeftCurly),
        ('\}', TokenID.RightCurly),
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

    def __init__(self, filename, stream):
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

    def tokenize(self) -> Iterator[Token]:
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
                    yield Token(TokenID.Undent, '', location)
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
                    yield Token(TokenID.Indent, '', location)
                    indentions.append(indent)

                while indentions[-1] > indent:
                    yield Token(TokenID.Undent, '', location)
                    indentions.pop()

            is_new = False
            is_empty = False

            if token.id in self.OPEN_BRACKETS:
                level += 1
            elif token.id in self.CLOSE_BRACKETS:
                level -= 1

            yield token

    def tokenize_all(self) -> Iterator[Token]:
        while self.index < self.length:
            yield self.__match()
        yield Token(TokenID.EndFile, "", self.location)

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
        return Token(symbol_id, value, location)

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


class Parser:
    MEMBERS_STARTS = (TokenID.Pass, TokenID.Def, TokenID.Class, TokenID.Struct, TokenID.Name)
    EXPRESSION_STARTS = (
        TokenID.Number, TokenID.Name, TokenID.LeftParenthesis, TokenID.Plus, TokenID.Minus, TokenID.Tilde
    )
    STATEMENT_STARTS = EXPRESSION_STARTS + (TokenID.Pass, TokenID.Return, TokenID.While, TokenID.If)

    def __init__(self, filename, stream):
        self.tokens = list(Scanner(filename, stream))
        self.index = 0

    @property
    def current_token(self) -> Token:
        return self.tokens[self.index]

    def match(self, *indexes: TokenID) -> bool:
        """
        Match current token

        :param indexes:     Token identifiers
        :return: True, if current token is matched passed identifiers
        """
        return self.current_token.id in indexes

    def consume(self, *indexes: TokenID) -> Token:
        """
        Consume current token

        :param indexes:     Token identifiers
        :return: Return consumed token
        :raise Diagnostic if current token is not matched passed identifiers
        """
        if not indexes or self.match(*indexes):
            token = self.current_token
            self.index += 1
            return token

        # generate exception message
        existed_name = self.current_token.id.name
        if len(indexes) > 1:
            required_names = ', '.join(f'`{x.name}`' for x in indexes)
            message = f"Required one of {required_names}, but got `{existed_name}`"
        else:
            required_name = indexes[0].name
            message = f"Required `{required_name}`, but got `{existed_name}`"
        raise Diagnostic(self.current_token.location, DiagnosticSeverity.Error, message)

    def try_consume(self, *indexes: TokenID) -> bool:
        """
        Try consume current token

        :param indexes:     Token identifiers
        :return: True, if current token is matched passed identifiers
        """
        if not self.match(*indexes):
            return False

        self.consume(*indexes)
        return True

    def parse(self) -> ModuleAST:
        """
        Parse module from source

        module:
            members EndFile
        """
        location = self.tokens[0].location + self.tokens[-1].location
        members = self.parse_members()
        self.consume(TokenID.EndFile)

        # noinspection PyArgumentList
        return ModuleAST(members=members, location=location)

    def parse_type(self) -> TypeAST:
        """
        type:
            atom_type { generic_arguments }
        """
        result_type = self.parse_atom_type()
        while self.match(TokenID.LeftSquare):
            arguments = self.parse_generic_arguments()

            # noinspection PyArgumentList
            result_type = ParameterizedTypeAST(type=result_type, arguments=arguments, location=result_type.location)
        return result_type

    def parse_atom_type(self) -> TypeAST:
        """
        atom_type:
            Name
        """
        tok_name = self.consume(TokenID.Name)

        # noinspection PyArgumentList
        return NamedTypeAST(name=tok_name.value, location=tok_name.location)

    def parse_generic_parameters(self) -> Sequence[GenericParameterAST]:
        """
        generic_parameters
            : [ '[' generic_parameter { ',' generic_parameter } ] ']' ]
        """
        if not self.match(TokenID.LeftSquare):
            return tuple()

        self.consume(TokenID.LeftSquare),
        parameters = [self.parse_generic_parameter()]
        while self.match(TokenID.Comma):
            self.consume(TokenID.Comma)
            parameters.append(self.parse_generic_parameter())
        self.consume(TokenID.RightSquare)

        return tuple(parameters)

    def parse_generic_parameter(self) -> GenericParameterAST:
        """
        generic_parameter
            Name
        """
        tok_name = self.consume(TokenID.Name)

        # noinspection PyArgumentList
        return GenericParameterAST(name=tok_name.value, location=tok_name.location)

    def parse_generic_arguments(self) -> Sequence[TypeAST]:
        """
        generic_arguments:
            '[' type { ',' type} ']'
        """
        self.consume(TokenID.LeftSquare)
        arguments = [self.parse_type()]
        while self.match(TokenID.Comma):
            self.consume(TokenID.Comma)
            arguments.append(self.parse_type())
        self.consume(TokenID.RightSquare)
        return tuple(arguments)

    def parse_members(self) -> Sequence[MemberAST]:
        """
        members:
            { member }
        """
        members = []
        while self.match(*self.MEMBERS_STARTS):
            members.append(self.parse_member())
        return tuple(members)

    def parse_member(self) -> MemberAST:
        """
        member:
            function
            class
            struct
            pass_member
            named_member
        """
        if self.match(TokenID.Def):
            return self.parse_function()
        elif self.match(TokenID.Class):
            return self.parse_class()
        elif self.match(TokenID.Struct):
            return self.parse_struct()
        elif self.match(TokenID.Pass):
            return self.parse_pass_member()
        elif self.match(TokenID.Name):
            return self.parse_named_member()

        self.match(*self.MEMBERS_STARTS)

    def parse_class(self) -> ClassAST:
        """
        class:
            'class' Name generic_parameters ':' type_members
        """
        self.consume(TokenID.Class)
        tok_name = self.consume(TokenID.Name)
        generic_parameters = self.parse_generic_parameters()
        members = self.parse_type_members()

        # noinspection PyArgumentList
        return ClassAST(
            name=tok_name.value,
            generic_parameters=generic_parameters,
            members=members,
            location=tok_name.location
        )

    def parse_struct(self) -> StructAST:
        """
        struct:
            'struct' Name generic_parameters ':' type_members
        """
        self.consume(TokenID.Struct)
        tok_name = self.consume(TokenID.Name)
        members = self.parse_type_members()
        generic_parameters = self.parse_generic_parameters()

        # noinspection PyArgumentList
        return StructAST(
            name=tok_name.value,
            generic_parameters=generic_parameters,
            members=members,
            location=tok_name.location
        )

    def parse_type_members(self) -> Sequence[MemberAST]:
        """
        type_members:
            ':' '...' '\n'
            ':' '\n' Indent members Undent
        """
        self.consume(TokenID.Colon)

        if self.try_consume(TokenID.Ellipsis):
            self.consume(TokenID.NewLine)
            return tuple()

        self.consume(TokenID.NewLine)
        self.consume(TokenID.Indent)
        members = self.parse_members()
        self.consume(TokenID.Undent)
        return members

    def parse_pass_member(self) -> PassMemberAST:
        """ pass_member: pass """
        tok_pass = self.consume(TokenID.Pass)
        self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return PassMemberAST(location=tok_pass.location)

    def parse_named_member(self) -> FieldAST:
        """
        named_member:
            Name ':' type
        """
        tok_name = self.consume(TokenID.Name)
        self.consume(TokenID.Colon)
        field_type = self.parse_type()
        self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return FieldAST(name=tok_name.value, type=field_type, location=tok_name.location)

    def parse_function(self) -> FunctionAST:
        """
        function:
            'def' Name generic_parameters '(' parameters ')' [ '->' type ] ':' NewLine function_statement
        """
        self.consume(TokenID.Def)
        tok_name = self.consume(TokenID.Name)
        generic_parameters = self.parse_generic_parameters()
        self.consume(TokenID.LeftParenthesis)
        parameters = self.parse_parameters()
        self.consume(TokenID.RightParenthesis)
        if self.try_consume(TokenID.Then):
            return_type = self.parse_type()
        else:
            # noinspection PyArgumentList
            return_type = NamedTypeAST(name='void', location=tok_name.location)
        self.consume(TokenID.Colon)
        statement = self.parse_function_statement()

        # noinspection PyArgumentList
        return FunctionAST(
            name=tok_name.value,
            generic_parameters=generic_parameters,
            parameters=parameters,
            return_type=return_type,
            statement=statement,
            location=tok_name.location
        )

    def parse_parameters(self) -> Sequence[ParameterAST]:
        """
        parameters:
            [ parameter { ',' parameter } ]
        """
        parameters = []
        if self.match(TokenID.Name):
            parameters.append(self.parse_parameter())
            while self.try_consume(TokenID.Comma):
                parameters.append(self.parse_parameter())

        return tuple(parameters)

    def parse_parameter(self) -> ParameterAST:
        """
        parameter:
            Name [ ':' type ]
        """
        tok_name = self.consume(TokenID.Name)
        if self.match(TokenID.Colon):
            self.consume(TokenID.Colon)
            param_type = self.parse_type()
        else:
            # noinspection PyArgumentList
            param_type = AutoTypeAST(location=tok_name.location)

        # noinspection PyArgumentList
        return ParameterAST(name=tok_name.value, type=param_type, location=tok_name.location)

    def parse_function_statement(self) -> Optional[StatementAST]:
        """
        function_statement:
            '...' EndFile
            NewLine block_statement
        """
        if self.try_consume(TokenID.Ellipsis):
            self.consume(TokenID.NewLine)
            return None

        self.consume(TokenID.NewLine)
        return self.parse_block_statement()

    def parse_block_statement(self) -> StatementAST:
        """
        block_statement:
            Indent statement { statement } Undent
        """
        self.consume(TokenID.Indent)
        statements = [self.parse_statement()]
        while self.match(*self.STATEMENT_STARTS):
            statements.append(self.parse_statement())
        self.consume(TokenID.Undent)
        location = statements[0].location + statements[-1].location

        # noinspection PyArgumentList
        return BlockStatementAST(statements=tuple(statements), location=location)

    def parse_statement(self) -> StatementAST:
        """
        statement:
            pass_statement
            return_statement
            expression_statement
            condition_statement
            while_statement
        """
        if self.match(TokenID.Pass):
            return self.parse_pass_statement()
        elif self.match(TokenID.Return):
            return self.parse_return_statement()
        elif self.match(TokenID.If):
            return self.parse_condition_statement()
        elif self.match(TokenID.While):
            return self.parse_while_statement()
        elif self.match(*self.EXPRESSION_STARTS):
            return self.parse_expression_statement()

        self.consume(*self.STATEMENT_STARTS)

    def parse_pass_statement(self) -> StatementAST:
        """ pass_statement: pass """
        tok_pass = self.consume(TokenID.Pass)
        self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return PassStatementAST(location=tok_pass.location)

    def parse_return_statement(self) -> StatementAST:
        """
        return_statement
            'return' [ expression ]
        """
        tok_return = self.consume(TokenID.Return)
        value = self.parse_expression() if self.match(*self.EXPRESSION_STARTS) else None
        self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return ReturnStatementAST(value=value, location=tok_return.location)

    def parse_else_statement(self):
        """
        else_statement:
            'else' ':' '\n' block_statement
        """
        self.consume(TokenID.Else)
        self.consume(TokenID.Colon)
        self.consume(TokenID.NewLine)
        return self.parse_block_statement()

    def parse_condition_statement(self, token_id: TokenID = TokenID.If) -> StatementAST:
        """
        condition_statement:
            'if' expression ':' '\n' block_statement            ⏎
                { 'elif' expression ':' '\n' block_statement }  ⏎
                [ else_statement ]
        """
        tok_if = self.consume(token_id)
        condition = self.parse_expression()
        self.consume(TokenID.Colon)
        self.consume(TokenID.NewLine)
        then_statement = self.parse_block_statement()

        else_statement = None
        if self.match(TokenID.Else):
            else_statement = self.parse_else_statement()
        elif self.match(TokenID.Elif):
            else_statement = self.parse_condition_statement(TokenID.Elif)

        # noinspection PyArgumentList
        return ConditionStatementAST(
            condition=condition,
            then_statement=then_statement,
            else_statement=else_statement,
            location=tok_if.location
        )

    def parse_while_statement(self) -> StatementAST:
        """
        while_statement:
            'while' expression ':' '\n' block_statement     ⏎
                [ 'else' ':' '\n' block_statement ]
        """
        tok_while = self.consume(TokenID.While)
        condition = self.parse_expression()
        self.consume(TokenID.Colon)
        self.consume(TokenID.NewLine)
        then_statement = self.parse_block_statement()
        else_statement = self.parse_else_statement() if self.match(TokenID.Else) else None

        # noinspection PyArgumentList
        return WhileStatementAST(
            condition=condition,
            then_statement=then_statement,
            else_statement=else_statement,
            location=tok_while.location
        )

    def parse_expression_statement(self) -> StatementAST:
        """
        expression_statement
            expression
            assign_expression
        """
        expression = self.parse_expression()
        statement = None

        if self.match(TokenID.Equals):
            statement = self.parse_assign_statement(expression)

        self.consume(TokenID.NewLine)
        if not statement:
            # noinspection PyArgumentList
            statement = ExpressionStatementAST(value=expression, location=expression.location)
        return statement

    def parse_assign_statement(self, target: ExpressionAST):
        """
        assign_expression
            target '=' expression
        """
        self.consume(TokenID.Equals)
        source = self.parse_expression()
        location = target.location + source.location

        # noinspection PyArgumentList
        return AssignStatementAST(
            target=target,
            source=source,
            location=location
        )

    def parse_arguments(self) -> Sequence[ExpressionAST]:
        """
        arguments:
            [ expression { ',' expression } [','] ]
        """
        if not self.match(*self.EXPRESSION_STARTS):
            return tuple()

        arguments = [self.parse_expression()]
        while self.match(TokenID.Comma):
            self.consume(TokenID.Comma)
            if self.match(*self.EXPRESSION_STARTS):
                arguments.append(self.parse_expression())
            else:
                break

        return tuple(arguments)

    def parse_expression(self) -> ExpressionAST:
        """
        expression:
            atom
        """
        return self.parse_addition_expression()

    def parse_addition_expression(self) -> ExpressionAST:
        """
        addition_expression:
            multiplication_expression
            addition_expression '+' multiplication_expression
            addition_expression '-' multiplication_expression
        """
        expression = self.parse_multiplication_expression()
        while self.match(TokenID.Plus, TokenID.Minus):
            if self.match(TokenID.Plus):
                tok_operator = self.consume(TokenID.Plus)
                right_operand = self.parse_unary_expression()

                # noinspection PyArgumentList
                expression = BinaryExpressionAST(
                    operator=BinaryID.Add,
                    left_operand=expression,
                    right_operand=right_operand,
                    location=tok_operator.location
                )
            elif self.match(TokenID.Minus):
                tok_operator = self.consume(TokenID.Minus)
                right_operand = self.parse_unary_expression()

                # noinspection PyArgumentList
                expression = BinaryExpressionAST(
                    operator=BinaryID.Sub,
                    left_operand=expression,
                    right_operand=right_operand,
                    location=tok_operator.location,
                )
        return expression

    def parse_multiplication_expression(self) -> ExpressionAST:
        """
        multiplication_expression:
            unary_expression
            multiplication_expression '*' unary_expression
            # multiplication_expression '@' multiplication_expression
            multiplication_expression '//' unary_expression
            multiplication_expression '/' unary_expression
            # multiplication_expression '%' unary_expression
        """
        expression = self.parse_unary_expression()
        while self.match(TokenID.Star, TokenID.Slash, TokenID.DoubleSlash):
            if self.match(TokenID.Star):
                tok_operator = self.consume(TokenID.Star)
                right_operand = self.parse_unary_expression()

                # noinspection PyArgumentList
                expression = BinaryExpressionAST(
                    operator=BinaryID.Mul,
                    left_operand=expression,
                    right_operand=right_operand,
                    location=tok_operator.location,
                )

            elif self.match(TokenID.Slash):
                tok_operator = self.consume(TokenID.Slash)
                right_operand = self.parse_unary_expression()

                # noinspection PyArgumentList
                expression = BinaryExpressionAST(
                    operator=BinaryID.Div,
                    left_operand=expression,
                    right_operand=right_operand,
                    location=tok_operator.location,
                )

            elif self.match(TokenID.DoubleSlash):
                tok_operator = self.consume(TokenID.DoubleSlash)
                right_operand = self.parse_unary_expression()

                # noinspection PyArgumentList
                expression = BinaryExpressionAST(
                    operator=BinaryID.DoubleDiv,
                    left_operand=expression,
                    right_operand=right_operand,
                    location=tok_operator.location,
                )

        return expression

    def parse_unary_expression(self) -> ExpressionAST:
        """
        u_expr:
            power
            "-" u_expr
            "+" u_expr
            "~" u_expr
        """
        if self.match(TokenID.Minus):
            tok_operator = self.consume(TokenID.Minus)
            operand = self.parse_unary_expression()

            # noinspection PyArgumentList
            return UnaryExpressionAST(operator=UnaryID.Neg, operand=operand, location=tok_operator.location)

        elif self.match(TokenID.Plus):
            tok_operator = self.consume(TokenID.Plus)
            operand = self.parse_unary_expression()

            # noinspection PyArgumentList
            return UnaryExpressionAST(operator=UnaryID.Pos, operand=operand, location=tok_operator.location)

        elif self.match(TokenID.Tilde):
            tok_operator = self.consume(TokenID.Tilde)
            operand = self.parse_unary_expression()

            # noinspection PyArgumentList
            return UnaryExpressionAST(operator=UnaryID.Inv, operand=operand, location=tok_operator.location)

        return self.parse_power_expression()

    def parse_power_expression(self) -> ExpressionAST:
        """
        power:
            primary ["**" u_expr]
        """
        expression = self.parse_primary_expression()
        if self.match(TokenID.DoubleStar):
            tok_operator = self.consume(TokenID.DoubleStar)
            unary_expression = self.parse_unary_expression()

            # noinspection PyArgumentList
            expression = BinaryExpressionAST(
                operator=BinaryID.Pow,
                left_operand=expression,
                right_operand=unary_expression,
                location=tok_operator.location
            )
        return expression

    def parse_primary_expression(self) -> ExpressionAST:
        """
        primary:
             number_expression
             name_expression
             call_expression
             parenthesis_expression
             subscribe_expression
             attribute_expression
        """
        if self.match(TokenID.Number):
            expression = self.parse_number_expression()
        elif self.match(TokenID.Name):
            expression = self.parse_name_expression()
        elif self.match(TokenID.LeftParenthesis):
            expression = self.parse_parenthesis_expression()
        else:
            raise self.match(*self.EXPRESSION_STARTS)  # Make linter happy

        while self.match(TokenID.LeftParenthesis, TokenID.LeftSquare, TokenID.Dot):
            if self.match(TokenID.LeftParenthesis):
                expression = self.parse_call_expression(expression)
            elif self.match(TokenID.LeftSquare):
                expression = self.parse_subscribe_expression(expression)
            elif self.match(TokenID.Dot):
                expression = self.parse_attribute_expression(expression)
        return expression

    def parse_number_expression(self) -> ExpressionAST:
        """
        number:
            Number
        """
        tok_number = self.consume(TokenID.Number)

        # noinspection PyArgumentList
        return IntegerExpressionAST(value=int(tok_number.value), location=tok_number.location)

    def parse_name_expression(self) -> ExpressionAST:
        """
        name:
            Name
        """
        tok_name = self.consume(TokenID.Name)

        # noinspection PyArgumentList
        return NamedExpressionAST(name=tok_name.value, location=tok_name.location)

    def parse_call_expression(self, value: ExpressionAST) -> ExpressionAST:
        """
        call_expression
            atom '(' arguments ')'
        """
        self.consume(TokenID.LeftParenthesis)
        arguments = self.parse_arguments()
        tok_close = self.consume(TokenID.RightParenthesis)
        location = value.location + tok_close.location

        # noinspection PyArgumentList
        return CallExpressionAST(value=value, arguments=arguments, location=location)

    def parse_subscribe_expression(self, value: ExpressionAST) -> ExpressionAST:
        """
        subscribe_expression
            atom '[' arguments ']'
        """
        self.consume(TokenID.LeftSquare)
        arguments = self.parse_arguments()
        tok_close = self.consume(TokenID.RightSquare)
        location = value.location + tok_close.location

        # noinspection PyArgumentList
        return SubscribeExpressionAST(value=value, arguments=arguments, location=location)

    def parse_attribute_expression(self, value: ExpressionAST) -> ExpressionAST:
        """
        attribute_expression:
            atom '.' Name
        """
        self.consume(TokenID.Dot)
        tok_name = self.consume(TokenID.Name)
        location = value.location + tok_name.location

        # noinspection PyArgumentList
        return AttributeAST(value=value, name=tok_name.value, location=location)

    def parse_parenthesis_expression(self) -> ExpressionAST:
        """
        parenthesis_expression:
            '(' expression ')'
        """
        self.consume(TokenID.LeftParenthesis)
        expression = self.parse_expression()
        self.consume(TokenID.RightParenthesis)
        return expression


@dataclass(unsafe_hash=True, frozen=True)
class NodeAST:
    location: Location

    def __make_iter(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if isinstance(value, list):
                class_name = type(self).__class__
                raise RuntimeError(f"Found not hashable field `{class_name}`.`{field.name}`")
            elif isinstance(value, tuple):
                yield from (child for child in value if isinstance(child, NodeAST))
            elif isinstance(value, NodeAST):
                yield value

    def __iter__(self):
        return self.__make_iter()


@dataclass(unsafe_hash=True, frozen=True)
class ModuleAST(NodeAST):
    members: Sequence[MemberAST]


@dataclass(unsafe_hash=True, frozen=True)
class TypeAST(NodeAST):
    pass


@dataclass(unsafe_hash=True, frozen=True)
class ParameterizedTypeAST(TypeAST):
    type: TypeAST
    arguments: Sequence[TypeAST]


@dataclass(unsafe_hash=True, frozen=True)
class GenericParameterAST(NodeAST):
    name: str


@dataclass(unsafe_hash=True, frozen=True)
class NamedTypeAST(TypeAST):
    name: str


@dataclass(unsafe_hash=True, frozen=True)
class AutoTypeAST(TypeAST):
    pass


@dataclass(unsafe_hash=True, frozen=True)
class MemberAST(NodeAST):
    pass


@dataclass(unsafe_hash=True, frozen=True)
class PassMemberAST(MemberAST):
    pass


@dataclass(unsafe_hash=True, frozen=True)
class TypeDeclarationAST(MemberAST):
    name: str
    members: Sequence[MemberAST]


@dataclass(unsafe_hash=True, frozen=True)
class StructAST(TypeDeclarationAST):
    generic_parameters: Sequence[GenericParameterAST]


@dataclass(unsafe_hash=True, frozen=True)
class ClassAST(TypeDeclarationAST):
    generic_parameters: Sequence[GenericParameterAST]


@dataclass(unsafe_hash=True, frozen=True)
class FieldAST(MemberAST):
    name: str
    type: TypeAST


@dataclass(unsafe_hash=True, frozen=True)
class ParameterAST(NodeAST):
    name: str
    type: TypeAST


@dataclass(unsafe_hash=True, frozen=True)
class FunctionAST(MemberAST):
    name: str
    generic_parameters: Sequence[GenericParameterAST]
    parameters: Sequence[ParameterAST]
    return_type: TypeAST
    statement: Optional[StatementAST]


@dataclass(unsafe_hash=True, frozen=True)
class StatementAST(NodeAST):
    pass


@dataclass(unsafe_hash=True, frozen=True)
class BlockStatementAST(StatementAST):
    statements: Sequence[StatementAST]


@dataclass(unsafe_hash=True, frozen=True)
class PassStatementAST(StatementAST):
    pass


@dataclass(unsafe_hash=True, frozen=True)
class ReturnStatementAST(StatementAST):
    value: Optional[ExpressionAST] = None


@dataclass(unsafe_hash=True, frozen=True)
class ConditionStatementAST(StatementAST):
    condition: ExpressionAST
    then_statement: StatementAST
    else_statement: Optional[StatementAST]


@dataclass(unsafe_hash=True, frozen=True)
class WhileStatementAST(StatementAST):
    condition: ExpressionAST
    then_statement: StatementAST
    else_statement: Optional[StatementAST]


@dataclass(unsafe_hash=True, frozen=True)
class ExpressionStatementAST(StatementAST):
    value: ExpressionAST


@dataclass(unsafe_hash=True, frozen=True)
class AssignStatementAST(StatementAST):
    target: ExpressionAST
    source: ExpressionAST


@dataclass(unsafe_hash=True, frozen=True)
class ExpressionAST(NodeAST):
    pass


@dataclass(unsafe_hash=True, frozen=True)
class IntegerExpressionAST(ExpressionAST):
    value: int


@dataclass(unsafe_hash=True, frozen=True)
class NamedExpressionAST(ExpressionAST):
    name: str


@enum.unique
class UnaryID(enum.IntEnum):
    Not = enum.auto()
    Pos = enum.auto()
    Neg = enum.auto()
    Inv = enum.auto()


@dataclass(unsafe_hash=True, frozen=True)
class UnaryExpressionAST(ExpressionAST):
    operator: UnaryID
    operand: ExpressionAST


@enum.unique
class BinaryID(enum.IntEnum):
    Add = enum.auto()
    Sub = enum.auto()
    Mul = enum.auto()
    Div = enum.auto()
    DoubleDiv = enum.auto()
    Pow = enum.auto()


@dataclass(unsafe_hash=True, frozen=True)
class BinaryExpressionAST(ExpressionAST):
    operator: BinaryID
    left_operand: ExpressionAST
    right_operand: ExpressionAST


@dataclass(unsafe_hash=True, frozen=True)
class CallExpressionAST(ExpressionAST):
    value: ExpressionAST
    arguments: Sequence[ExpressionAST]


@dataclass(unsafe_hash=True, frozen=True)
class SubscribeExpressionAST(ExpressionAST):
    value: ExpressionAST
    arguments: Sequence[ExpressionAST]


@dataclass(unsafe_hash=True, frozen=True)
class AttributeAST(ExpressionAST):
    value: ExpressionAST
    name: str


class LexicalScope:
    def __init__(self, parent: LexicalScope = None):
        self.__parent = parent
        self.__defined = dict()  # Defined symbols
        self.__resolved = dict()  # Resolved symbols

    @property
    def parent(self) -> LexicalScope:
        return self.__parent

    def resolve(self, name: str) -> Optional[NamedSymbol]:
        """
        Resolve symbol by name in current scope.

        If symbol is defined in current scope and:

            - has type `Overload` it must extended with functions from parent scope

        If symbol is not defined in current scope, check if it can be resolved in parent scope.
        """

        # If symbol already resolved then returns it.
        if name in self.__resolved:
            return self.__resolved[name]

        # Resolve symbol in current scope
        symbol = self.__defined.get(name)
        if symbol:
            if self.parent and isinstance(symbol, Overload):
                parent_symbol = self.parent.resolve(name)
                if isinstance(parent_symbol, Overload):
                    symbol.extend(parent_symbol)

        # Resolve symbol in parent scope
        elif self.parent:
            symbol = self.parent.resolve(name)

        # Return None, if symbol is not found in current and ascendant scopes
        if not symbol:
            return None

        # Clone overload
        if isinstance(symbol, Overload):
            overload = Overload(name, symbol.functions[0])
            overload.extend(overload)
            symbol = overload

        # Save resolved symbol
        self.__resolved[name] = symbol
        return symbol

    def append(self, symbol: NamedSymbol, name: str = None) -> None:
        name = name or symbol.name
        try:
            existed_symbol = self.__defined[name]
        except KeyError:
            self.__defined[name] = Overload(name, symbol) if isinstance(symbol, Function) else symbol
        else:
            if not isinstance(existed_symbol, Overload) or not isinstance(symbol, Function):
                raise Diagnostic(symbol.location, DiagnosticSeverity.Error, f"Already defined symbol with name {name}")
            existed_symbol.append(symbol)


class SemanticContext:
    def __init__(self, paths=None):
        if not paths:
            paths = [
                os.path.join(os.path.dirname(__file__), 'stdlib'),
                os.getcwd()
            ]
        self.paths = paths
        self.modules = {}

    @staticmethod
    def convert_module_name(filename, path):
        fullname = os.path.abspath(filename)
        if not fullname.startswith(path):
            raise BootstrapError(f"Not found file `{filename}` in library path `{path}`")

        module_name = fullname[len(path):]
        module_name, _ = os.path.splitext(module_name)
        module_name = module_name.strip(os.path.sep)
        module_name = module_name.replace(os.path.sep, '.')
        return module_name

    @staticmethod
    def convert_filename(module_name, path):
        filename = module_name.replace('.', os.path.sep) + '.orx'
        return os.path.join(path, filename)

    @cached_property
    def builtins_module(self) -> Module:
        return self.load('__builtins__')

    @cached_property
    def boolean_type(self) -> BooleanType:
        return cast(BooleanType, self.builtins_module.scope.resolve('bool'))

    @cached_property
    def integer_type(self) -> IntegerType:
        return cast(IntegerType, self.builtins_module.scope.resolve('int'))

    @cached_property
    def void_type(self) -> VoidType:
        return cast(VoidType, self.builtins_module.scope.resolve('void'))

    def get_module_name(self, filename):
        fullname = os.path.abspath(filename)
        for path in self.paths:
            if fullname.startswith(path):
                return self.convert_module_name(fullname, path)

        raise BootstrapError(f"Not found file `{filename}` in library paths")

    def open(self, filename) -> Module:
        """ Open module from file """
        module_name = self.get_module_name(filename)

        with open(filename, 'r', encoding='utf8') as stream:
            return self.__open_source(filename, module_name, stream)

    def load(self, module_name) -> Module:
        for path in self.paths:
            filename = self.convert_filename(module_name, path)
            try:
                with open(filename, 'r', encoding='utf8') as stream:
                    return self.__open_source(filename, module_name, stream)
            except IOError:
                logger.info(f"Not found module `{module_name}` in file `{filename}`")
                pass  # Continue

        raise BootstrapError(f'Not found module {module_name}')

    def __open_source(self, filename, module_name, stream) -> Module:
        logger.info(f"Open `{module_name}` from file `{filename}`")

        if module_name in self.modules:
            model = self.modules[module_name]
        else:
            parser = Parser(filename, stream)
            tree = parser.parse()

            model = SemanticModel(self, module_name, tree)
            self.modules[module_name] = model
            model.analyze()
        return model.module


class SemanticModel:
    def __init__(self, context: SemanticContext, module_name: str, tree: ModuleAST):
        self.context = context
        self.module_name = module_name
        self.tree = tree
        self.symbols = {}
        self.scopes = {}

        self.__functions = collections.deque()

    @property
    def module(self) -> Module:
        return self.symbols[self.tree]

    @contextmanager
    def with_function(self, func: Function):
        self.__functions.append(func)
        yield func
        self.__functions.pop()

    @property
    def current_function(self) -> Function:
        return self.__functions[0]

    def analyze(self):
        self.annotate_recursive_scope(self.tree)
        self.declare_symbol(self.tree, None)
        self.emit_functions(self.tree)

    def annotate_recursive_scope(self, node: NodeAST, parent=None):
        scope = self.scopes.get(node) or self.annotate_scope(node, parent)
        self.scopes[node] = scope
        for child in node:
            self.annotate_recursive_scope(child, scope)

    @multimethod
    def annotate_scope(self, _: NodeAST, parent: LexicalScope) -> LexicalScope:
        return parent

    @multimethod
    def annotate_scope(self, _1: ModuleAST, _2=None) -> LexicalScope:
        return LexicalScope()

    @multimethod
    def annotate_scope(self, _: FunctionAST, parent: LexicalScope) -> LexicalScope:
        return LexicalScope(parent)

    @multimethod
    def annotate_scope(self, _: BlockStatementAST, parent: LexicalScope) -> LexicalScope:
        return LexicalScope(parent)

    @multimethod
    def annotate_scope(self, _: ClassAST, parent: LexicalScope) -> LexicalScope:
        return LexicalScope(parent)

    @multimethod
    def annotate_scope(self, _: StructAST, parent: LexicalScope) -> LexicalScope:
        return LexicalScope(parent)

    def declare_symbol(self, node: NodeAST, scope: LexicalScope = None, parent: ContainerSymbol = None):
        symbol = self.annotate_symbol(node, parent)
        if not symbol:
            return None

        # Declare symbol in parent scope
        self.symbols[node] = symbol
        if scope is not None and isinstance(symbol, NamedSymbol):
            scope.append(symbol)
        if parent:
            parent.add_member(symbol)

        types = []
        functions = []
        others = []

        # Collect types
        if hasattr(node, 'members'):
            for child in node.members:
                if isinstance(child, TypeDeclarationAST):
                    types.append(child)
                elif isinstance(child, FunctionAST):
                    functions.append(child)
                else:
                    others.append(child)

        child_scope = self.scopes[node]
        for child in itertools.chain(types, functions, others):
            self.declare_symbol(child, child_scope, symbol)

        return symbol

    @multimethod
    def resolve_type(self, node: TypeAST) -> Type:
        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented type resolving")

    @multimethod
    def resolve_type(self, node: NamedTypeAST) -> Type:
        if node.name == 'void':
            return self.context.void_type
        elif node.name == 'bool':
            return self.context.boolean_type
        elif node.name == 'int':
            return self.context.integer_type

        symbol = self.scopes[node].resolve(node.name)
        if isinstance(symbol, Type):
            return symbol

        raise Diagnostic(
            node.location, DiagnosticSeverity.Error, f"Not found symbol `{node.name} in current scope`")

    @multimethod
    def resolve_type(self, node: ParameterizedTypeAST) -> Type:
        instance_type = self.resolve_type(node.type)
        arguments = [self.resolve_type(arg) for arg in node.arguments]
        return instance_type.instantiate(self.module, arguments, node.location)

    def annotate_generics(self, scope: LexicalScope, generic_parameters: Sequence[GenericParameterAST]):
        parameters = []
        for generic_node in generic_parameters:
            generic = GenericType(self.module, generic_node.name, generic_node.location)

            scope.append(generic)
            parameters.append(generic)
        return parameters

    @multimethod
    def annotate_symbol(self, node: NodeAST, parent: ContainerSymbol) -> Symbol:
        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented member declaration")

    # noinspection PyUnusedLocal
    @multimethod
    def annotate_symbol(self, node: ModuleAST, parent=None) -> Module:
        return Module(self.context, self.module_name, Location(node.location.filename))

    @multimethod
    def annotate_symbol(self, node: PassMemberAST, parent: ContainerSymbol) -> Optional[Symbol]:
        return None

    @multimethod
    def annotate_symbol(self, node: FunctionAST, parent: ContainerSymbol) -> Function:
        scope = self.scopes[node]
        generic_parameters = self.annotate_generics(scope, node.generic_parameters)
        if isinstance(parent, Type) and node.parameters and isinstance(node.parameters[0].type, AutoTypeAST):
            parameters = [parent]
            parameters.extend(self.resolve_type(param.type) for param in node.parameters[1:])
        else:
            parameters = [self.resolve_type(param.type) for param in node.parameters]
        return_type = self.resolve_type(node.return_type)
        func_type = FunctionType(self.module, parameters, return_type, node.location)
        func = Function(parent, node.name, func_type, node.location, generic_parameters=generic_parameters)

        for node_param, func_param in zip(node.parameters, func.parameters):
            func_param.name = node_param.name
            func_param.location = node_param.location

            self.symbols[node_param] = func_param
            scope.append(func_param)

        return func

    @multimethod
    def annotate_symbol(self, node: StructAST, parent: ContainerSymbol) -> Type:
        if self.module == self.context.builtins_module:
            if node.name == "int":
                return IntegerType(parent, node.location)
            elif node.name == "bool":
                return BooleanType(parent, node.location)
            elif node.name == "void":
                return VoidType(parent, node.location)

        generic_parameters = self.annotate_generics(self.scopes[node], node.generic_parameters)
        return StructType(parent, node.name, node.location, generic_parameters=generic_parameters)

    @multimethod
    def annotate_symbol(self, node: ClassAST, parent: ContainerSymbol) -> Type:
        generic_parameters = self.annotate_generics(self.scopes[node], node.generic_parameters)
        return ClassType(parent, node.name, node.location, generic_parameters=generic_parameters)

    @multimethod
    def annotate_symbol(self, node: GenericParameterAST, parent: ContainerSymbol) -> Symbol:
        return GenericType(node.name, node.location)

    @multimethod
    def annotate_symbol(self, node: FieldAST, parent: ContainerSymbol) -> Field:
        if not isinstance(parent, Type):
            raise Diagnostic(node.location, DiagnosticSeverity.Error, "Field member must be declared in type")

        field_type = self.resolve_type(node.type)
        return Field(cast(Type, parent), node.name, field_type, node.location)

    def emit_functions(self, module: ModuleAST):
        for member in module.members:
            if isinstance(member, FunctionAST):
                self.emit_function(member)

    def emit_function(self, node: FunctionAST):
        func = self.symbols[node]
        if node.statement:
            with self.with_function(func):
                func.statement = self.emit_statement(node.statement)

    def get_functions(self, scope: LexicalScope, name: str, self_type: Type = None) -> Sequence[Function]:
        functions = []

        # scope function
        symbol = scope.resolve(name)
        if isinstance(symbol, Overload):
            functions.extend(symbol.functions)

        # type function
        symbol = self_type.scope.resolve(name) if self_type else None
        if isinstance(symbol, Overload):
            functions.extend(symbol.functions)

        return functions

    @staticmethod
    def check_naive_function(func: Function, arguments: Sequence[Value]) -> Tuple[Optional[int], Function]:
        """
        Returns:

            - None              - if function can not be called with arguments
            - Sequence[int]     - if function can be called with arguments. Returns priority

        :param func:
        :param arguments:
        :return:
        """
        if len(func.parameters) != len(arguments):
            return None, func

        priority = 0
        for param, arg in zip(func.parameters, arguments):
            if arg.type != param.type:
                return None, func
            priority += 2
        return priority, func

    def check_generic_function(self, func: Function, arguments: Sequence[Value], location: Location) \
            -> Tuple[Optional[int], Function]:
        if len(func.parameters) != len(arguments):
            return None, func

        context = InferenceContext()
        instance_types = [context.add_generic_parameter(parameter) for parameter in func.generic_parameters]
        parameter_types = [context.add_type(parameter.type) for parameter in func.parameters]
        argument_types = [context.add_type(arg.type) for arg in arguments]
        for param_type, arg_type in zip(parameter_types, argument_types):
            context.unify(param_type, arg_type)

        generic_arguments = [var_type.instantiate(self.module) for var_type in instance_types]
        instance = func.instantiate(self.module, generic_arguments, location)
        return -1, instance

    def check_function(self, func: Function, arguments: Sequence[Value], location: Location) \
            -> Optional[Tuple[int, Function]]:
        if func.is_generic:
            return self.check_generic_function(func, arguments, location)
        return self.check_naive_function(func, arguments)

    def find_function(self, scope: LexicalScope, name: str, arguments: Sequence[Value], location: Location) \
            -> Optional[Function]:
        # find candidates
        functions = self.get_functions(scope, name, arguments[0].type if arguments else None)

        # check candidates
        counter = itertools.count()
        candidates = []
        for func in functions:
            priority, instance = self.check_function(func, arguments, location)
            if priority is not None:
                heapq.heappush(candidates, (priority, next(counter), instance))

        # pop all function with minimal priority
        functions = []
        current_priority = None
        while candidates:
            priority, _, func = heapq.heappop(candidates)
            if current_priority is not None and current_priority != priority:
                break

            current_priority = priority
            functions.append(func)

        if functions:
            return functions[0]
        return None

    def resolve_function(self, scope: LexicalScope, name: str, arguments: Sequence[Value], location: Location) \
            -> Function:
        func = self.find_function(scope, name, arguments, location)
        if not func:
            arguments = ', '.join(str(arg.type) for arg in arguments)
            message = f'Not found function ‘{name}({arguments})’ in current scope'
            raise Diagnostic(location, DiagnosticSeverity.Error, message)
        return func

    @multimethod
    def emit_statement(self, node: StatementAST) -> Statement:
        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented statement emitting")

    @multimethod
    def emit_statement(self, node: BlockStatementAST) -> Statement:
        statements = [self.emit_statement(statement) for statement in node.statements]
        return BlockStatement(statements, node.location)

    @multimethod
    def emit_statement(self, node: PassStatementAST) -> Statement:
        return PassStatement(node.location)

    @multimethod
    def emit_statement(self, node: ReturnStatementAST) -> Statement:
        value = self.emit_value(node.value) if node.value else None
        return_type = self.current_function.return_type
        void_type = self.context.void_type

        if value and value.type != return_type:
            message = f"Return statement value must have ‘{return_type}’ type, got ‘{value.type}’"
            raise Diagnostic(node.location, DiagnosticSeverity.Error, message)
        elif not value and void_type != return_type:
            message = f"Return statement value must have ‘{return_type}’ type, got ‘{void_type}’"
            raise Diagnostic(node.location, DiagnosticSeverity.Error, message)
        return ReturnStatement(value, node.location)

    @multimethod
    def emit_statement(self, node: ExpressionStatementAST) -> Statement:
        value = self.emit_value(node.value)
        return ExpressionStatement(value)

    @multimethod
    def emit_statement(self, node: ConditionStatementAST) -> Statement:
        condition = self.emit_value(node.condition)
        then_statement = self.emit_statement(node.then_statement)
        else_statement = self.emit_statement(node.else_statement) if node.else_statement else None

        c_type = condition.type
        if c_type != self.context.boolean_type:
            message = f"Condition expression for statement must have ‘bool’ type, got ‘{c_type}’"
            raise Diagnostic(node.condition.location, DiagnosticSeverity.Error, message)

        return ConditionStatement(condition, then_statement, else_statement, node.location)

    @multimethod
    def emit_statement(self, node: WhileStatementAST) -> Statement:
        condition = self.emit_value(node.condition)
        then_statement = self.emit_statement(node.then_statement)
        else_statement = self.emit_statement(node.else_statement) if node.else_statement else None

        c_type = condition.type
        if c_type != self.context.boolean_type:
            message = f"Condition expression for statement must have ‘bool’ type, got ‘{c_type}’"
            raise Diagnostic(node.condition.location, DiagnosticSeverity.Error, message)

        return WhileStatement(condition, then_statement, else_statement, node.location)

    @multimethod
    def emit_statement(self, node: AssignStatementAST) -> Statement:
        value = self.emit_value(node.source)
        return self.emit_assignment(node.target, value, node.location)

    @multimethod
    def emit_assignment(self, node: ExpressionAST, value: Value, location: Location) -> Statement:
        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented target assignement emitting")

    @multimethod
    def emit_assignment(self, node: NamedExpressionAST, value: Value, location: Location) -> Statement:
        symbol = self.emit_symbol(node, False)
        if not isinstance(symbol, TargetValue):
            symbol = self.current_function.add_variables(node.name, value.type, location)

            scope: LexicalScope = self.scopes[node]
            scope.append(symbol)

        if symbol.type != value.type:
            message = f"Can not cast  from type ‘{value.type}’ type, got ‘{symbol.type}’"
            raise Diagnostic(node.location, DiagnosticSeverity.Error, message)

        return AssignStatement(symbol, value, node.location)

    @multimethod
    def emit_assignment(self, node: AttributeAST, value: Value, location: Location) -> Statement:
        symbol = self.emit_symbol(node, True)
        if not isinstance(symbol, TargetValue):
            message = f"Can not assign value to target"
            raise Diagnostic(node.location, DiagnosticSeverity.Error, message)

        if symbol.type != value.type:
            message = f"Can not cast  from type ‘{value.type}’ type, got ‘{symbol.type}’"
            raise Diagnostic(node.location, DiagnosticSeverity.Error, message)

        return AssignStatement(symbol, value, node.location)

    @multimethod
    def emit_value(self, node: ExpressionAST) -> Value:
        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented value emitting")

    @multimethod
    def emit_value(self, node: IntegerExpressionAST) -> Value:
        return cast(Value, self.emit_symbol(node, True))

    @multimethod
    def emit_value(self, node: NamedExpressionAST) -> Value:
        value = self.emit_symbol(node, True)
        if isinstance(value, Value):
            return value

        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Required value, but got another object")

    @multimethod
    def emit_value(self, node: AttributeAST) -> Value:
        value = self.emit_symbol(node, True)
        if isinstance(value, Value):
            return value

        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Required value, but got another object")

    @multimethod
    def emit_value(self, node: CallExpressionAST) -> Value:
        arguments = [self.emit_value(arg) for arg in node.arguments]

        symbol = self.emit_symbol(node.value, False)

        # function call
        if isinstance(symbol, Overload):
            func = self.resolve_function(self.scopes[node], symbol.name, arguments, node.location)
            return CallInstruction(func, arguments, node.location)

        # type instance instantiate
        elif isinstance(symbol, Type):
            return NewInstruction(symbol, arguments, node.location)

        # instance function call (uniform calls)
        elif not symbol and isinstance(node.value, NamedExpressionAST):
            func = self.resolve_function(self.scopes[node], node.value.name, arguments, node.location)
            return CallInstruction(func, arguments, node.location)

        raise Diagnostic(node.location, DiagnosticSeverity.Error, f'Not found function for call')

    @multimethod
    def emit_value(self, node: UnaryExpressionAST) -> Value:
        arguments = [self.emit_value(node.operand)]

        if node.operator == UnaryID.Pos:
            func = self.resolve_function(self.scopes[node], '__pos__', arguments, node.location)
        elif node.operator == UnaryID.Neg:
            func = self.resolve_function(self.scopes[node], '__neg__', arguments, node.location)
        elif node.operator == UnaryID.Not:
            func = self.resolve_function(self.scopes[node], '__not__', arguments, node.location)
        else:
            raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented unary operator")

        return CallInstruction(func, arguments, node.location)

    @multimethod
    def emit_value(self, node: BinaryExpressionAST) -> Value:
        arguments = [self.emit_value(node.left_operand), self.emit_value(node.right_operand)]

        if node.operator == BinaryID.Add:
            func = self.resolve_function(self.scopes[node], '__add__', arguments, node.location)
        elif node.operator == BinaryID.Sub:
            func = self.resolve_function(self.scopes[node], '__sub__', arguments, node.location)
        elif node.operator == BinaryID.Mul:
            func = self.resolve_function(self.scopes[node], '__mul__', arguments, node.location)
        elif node.operator == BinaryID.Div:
            func = self.resolve_function(self.scopes[node], '__div__', arguments, node.location)
        else:
            raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented binary operator")

        return CallInstruction(func, arguments, node.location)

    @multimethod
    def emit_symbol(self, node: ExpressionAST, is_exists: bool) -> Symbol:
        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented symbol emitting")

    @multimethod
    def emit_symbol(self, node: IntegerExpressionAST, is_exists: bool) -> Symbol:
        return IntegerConstant(self.context.integer_type, node.value, node.location)

    @multimethod
    def emit_symbol(self, node: NamedExpressionAST, is_exists: bool) -> Symbol:
        if node.name in ['True', 'False']:
            return BooleanConstant(self.context.boolean_type, node.name == 'True', node.location)
        elif node.name == 'void':
            return self.context.void_type
        elif node.name == 'bool':
            return self.context.boolean_type
        elif node.name == 'int':
            return self.context.integer_type

        scope = self.scopes[node]
        symbol = scope.resolve(node.name)
        if is_exists and not symbol:
            raise Diagnostic(
                node.location, DiagnosticSeverity.Error, f"Not found symbol `{node.name} in current scope`")
        return symbol

    @multimethod
    def emit_symbol(self, node: AttributeAST, is_exists: bool) -> Symbol:
        instance = self.emit_symbol(node.value, True)
        if isinstance(instance, Value):
            value_type = instance.type
            symbol = value_type.scope.resolve(node.name)

            if isinstance(symbol, Field):
                return BoundedField(instance, symbol, node.location)
            # elif isinstance(symbol, Overload):
            #     return BoundedOverload(instance, symbol)
            elif is_exists and not symbol:
                raise Diagnostic(
                    node.location, DiagnosticSeverity.Error, f"Not found symbol `{node.name}` in type {value_type}`")

        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented symbol emitting")

    @multimethod
    def emit_symbol(self, node: SubscribeExpressionAST, is_exists: bool) -> Symbol:
        symbol = self.emit_symbol(node.value, True)
        arguments = [self.emit_symbol(arg, True) for arg in node.arguments]

        if isinstance(symbol, Type):
            return symbol.instantiate(self.module, arguments, node.location)

        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented symbol emitting")


class InstantiateContext:
    def __init__(self, module: Module):
        self.module = module
        self.__mapping = {}

    def aggregate(self, generic_parameters, generic_arguments):
        for param, arg in zip(generic_parameters, generic_arguments):
            self.register(param, arg)

    def register(self, param, arg):
        self.__mapping[param] = arg

    @multimethod
    def instantiate(self, generic: Type, location: Location):
        if generic in self.__mapping:
            return self.__mapping[generic]

        if isinstance(generic, FunctionType):  # TODO: Make generic for function type!
            parameters = [self.instantiate(param, location) for param in generic.parameters]
            return_type = self.instantiate(generic.return_type, location)
            result_type = FunctionType(self.module, parameters, return_type, generic.location)

        elif generic.generic_parameters:
            generic_arguments = [self.instantiate(arg, location) for arg in generic.generic_parameters]
            result_type = generic.instantiate(self.module, generic_arguments, location)

        elif generic.generic_arguments:
            generic_arguments = [self.instantiate(arg, location) for arg in generic.generic_arguments]
            result_type = generic.instantiate(self.module, generic_arguments, location)
        else:
            result_type = generic

        self.register(generic, result_type)
        return result_type

    @multimethod
    def instantiate(self, field: Field, location: Location) -> Field:
        if field in self.__mapping:
            return self.__mapping[field]

        new_owner = self.instantiate(field.owner, location)
        new_type = self.instantiate(field.type, location)
        new_field = Field(new_owner, field.name, new_type, field.location)

        self.register(field, new_field)
        return new_field

    @multimethod
    def instantiate(self, statement: Statement, location: Location):
        raise Diagnostic(statement.location, DiagnosticSeverity.Error, "Not implemented statement instantiation")

    @multimethod
    def instantiate(self, statement: BlockStatement, location: Location):
        return BlockStatement(
            [self.instantiate(child, location) for child in statement.statements],
            statement.location
        )

    @multimethod
    def instantiate(self, statement: ReturnStatement, location: Location):
        return ReturnStatement(
            self.instantiate(statement.value, location) if statement.value else None,
            statement.location
        )

    @multimethod
    def instantiate(self, value: Value, location: Location):
        raise Diagnostic(value.location, DiagnosticSeverity.Error, "Not implemented value instantiation")

    @multimethod
    def instantiate(self, value: IntegerConstant, location: Location):
        return value

    @multimethod
    def instantiate(self, value: BooleanConstant, location: Location):
        return value

    @multimethod
    def instantiate(self, value: Parameter, location: Location):
        return self.__mapping[value]


class InferenceType(abc.ABC):
    def __init__(self, location: Location):
        self.location = location

    @abc.abstractmethod
    def prune(self) -> InferenceType:
        raise NotImplementedError

    @abc.abstractmethod
    def instantiate(self, module: Module) -> Type:
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError


class InferenceVariable(InferenceType):
    def __init__(self, name: str, location: Location):
        super(InferenceVariable, self).__init__(location)

        self.__name = name
        self.__instance = None

    @property
    def name(self):
        return self.__name

    @property
    def instance(self) -> InferenceType:
        return self.__instance

    @instance.setter
    def instance(self, value: InferenceType):
        self.__instance = value

    def prune(self) -> InferenceType:
        if self.instance:
            self.instance = self.instance.prune()
            return self.instance
        return self

    def instantiate(self, module: Module) -> Type:
        if self.instance:
            return self.instance.instantiate(module)

        raise Diagnostic(self.location, DiagnosticSeverity.Error, "Can not instantiate type variable")

    def __str__(self):
        if self.instance:
            return str(self.instance)
        return self.__name


class InferenceConstructor(InferenceType):
    def __init__(self, constructor: Type, arguments: Sequence[InferenceType], location: Location):
        super(InferenceConstructor, self).__init__(location)

        self.constructor = constructor
        self.arguments = tuple(arguments)

    def prune(self) -> InferenceType:
        arguments = [arg.prune() for arg in self.arguments]
        if self.arguments != arguments:
            return InferenceConstructor(self.constructor, arguments, self.location)
        return self

    def instantiate(self, module: Module) -> Type:
        if not self.arguments:
            return self.constructor

        arguments = [arg.instantiate(module) for arg in self.arguments]
        return self.constructor.instantiate(module, arguments)

    def __str__(self):
        if self.arguments:
            arguments = ', '.join(str(arg) for arg in self.arguments)
            return f'{self.constructor.name}[{arguments}]'
        return self.constructor.name


class InferenceError(BootstrapError):
    pass


class InferenceContext:
    def __init__(self):
        self.__types = {}

    def add_generic_parameter(self, param: GenericParameter) -> InferenceVariable:
        self.__types[param] = InferenceVariable(param.name, param.location)
        return self.__types[param]

    def add_type(self, param_type: Type) -> InferenceType:
        if param_type in self.__types:
            return self.__types[param_type]

        if param_type.generic_arguments:
            arguments = [self.add_type(generic_argument) for generic_argument in param_type.generic_arguments]
            constructor = InferenceConstructor(param_type.definition, arguments, param_type.location)
        elif param_type.generic_parameters:
            arguments = [self.add_type(generic_parameter) for generic_parameter in param_type.generic_arguments]
            constructor = InferenceConstructor(param_type.definition, arguments, param_type.location)
        else:
            constructor = InferenceConstructor(param_type, [], param_type.location)

        self.__types[param_type] = constructor
        return constructor

    @classmethod
    def is_generic(cls, v: InferenceType, non_generic):
        """Checks whether a given variable occurs in a list of non-generic variables

        Note that a variables in such a list may be instantiated to a type term,
        in which case the variables contained in the type term are considered
        non-generic.

        Note: Must be called with v pre-pruned

        Args:
            v: The TypeVariable to be tested for genericity
            non_generic: A set of non-generic TypeVariables

        Returns:
            True if v is a generic variable, otherwise False
        """
        return not cls.occurs_in(v, non_generic)

    @classmethod
    def occurs_in_type(cls, v: InferenceType, type2: InferenceType):
        """Checks whether a type variable occurs in a type expression.

        Note: Must be called with v pre-pruned

        Args:
            v:  The TypeVariable to be tested for
            type2: The type in which to search

        Returns:
            True if v occurs in type2, otherwise False
        """
        pruned_type2 = type2.prune()
        if pruned_type2 == v:
            return True
        elif isinstance(pruned_type2, InferenceConstructor):
            return cls.occurs_in(v, pruned_type2.arguments)
        return False

    @classmethod
    def occurs_in(cls, t: InferenceType, types: Sequence[InferenceType]):
        """Checks whether a types variable occurs in any other types.

        Args:
            t:  The TypeVariable to be tested for
            types: The sequence of types in which to search

        Returns:
            True if t occurs in any of types, otherwise False
        """
        return any(cls.occurs_in_type(t, t2) for t2 in types)

    @classmethod
    def unify(cls, t1: InferenceType, t2: InferenceType):
        """
        Makes the types t1 and t2 the same.

        :param t1:  The first type to be made equivalent
        :param t2:  The second type to be be equivalent
        :return: None
        :raises InferenceError - Raised if the types cannot be unified.
        """

        t1 = t1.prune()
        t2 = t2.prune()
        if isinstance(t1, InferenceVariable):
            if t1 != t2:
                if cls.occurs_in_type(t1, t2):
                    raise InferenceError("recursive unification")
                t1.instance = t2
        elif isinstance(t1, InferenceConstructor) and isinstance(t2, InferenceVariable):
            cls.unify(t2, t1)
        elif isinstance(t1, InferenceConstructor) and isinstance(t2, InferenceConstructor):
            if t1.constructor != t2.constructor or len(t1.arguments) != len(t2.arguments):
                raise InferenceError("Type mismatch: {0} != {1}".format(str(t1), str(t2)))
            for p, q in zip(t1.arguments, t2.arguments):
                cls.unify(p, q)
        else:
            assert 0, "Not unified"


class MangledContext:
    def __init__(self):
        self.parts = []

    @multimethod
    def append(self, name: str):
        self.parts.append(name)
        self.append(len(name))

    @multimethod
    def append(self, value: int):
        self.parts.append(str(value))

    @multimethod
    def append(self, module: Module):
        self.append(module.name)
        self.append("M")

    def append_generic(self, generics):
        for generic in reversed(generics):
            self.append(generic)
        self.append(len(generics))
        self.append("G")

    def construct(self):
        return ''.join(reversed(self.parts))

    @multimethod
    def append(self, type: Type):
        self.parts.append(str(type))

    @multimethod
    def mangle(self, symbol: MangledSymbol):
        raise Diagnostic(symbol.location, DiagnosticSeverity.Error, "Can not mangle symbol name")

    @multimethod
    def mangle(self, func: Function):
        definition = func.definition if func.definition else func

        self.append(func.return_type)
        self.parts.append("R")
        for param in reversed(func.parameters):
            self.append(param.type)
            self.parts.append("P")
        self.append(len(func.parameters))
        self.append('A')

        if func.generic_arguments:
            self.append_generic(func.generic_arguments)
        elif definition.generic_parameters:
            self.append_generic(definition.generic_parameters)
        self.append(func.name)
        self.append(len(func.name))
        self.append("F")

        self.append('::')
        self.append(definition.owner)
        self.append('ORX_FUNC_')

        return self.construct()

    @multimethod
    def mangle(self, symbol: IntegerType):
        return "i32"

    @multimethod
    def mangle(self, symbol: BooleanType):
        return "b"

    @multimethod
    def mangle(self, symbol: VoidType):
        return "v"

    @multimethod
    def mangle(self, type_symbol: Type):
        definition = type_symbol.definition if type_symbol.definition else type_symbol

        if type_symbol.generic_arguments:
            self.append_generic(type_symbol.generic_arguments)
        elif definition.generic_parameters:
            self.append_generic(definition.generic_parameters)
        self.append(type_symbol.name)
        self.append(len(type_symbol.name))
        self.append("T")

        self.append('::')
        self.append(definition.owner)
        self.append('ORX_TYPE_')

        return self.construct()


class Symbol(abc.ABC):
    """ Abstract base for all symbols """

    @property
    @abc.abstractmethod
    def location(self) -> Location:
        raise NotImplementedError

    @abc.abstractmethod
    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        class_name = type(self).__name__
        return f'<{class_name}: {self}>'


class NamedSymbol(Symbol, abc.ABC):
    """ Abstract base for all named symbols """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    def __str__(self):
        return self.name


class OwnedSymbol(NamedSymbol, abc.ABC):
    """ Abstract base for all owned symbols """

    @property
    @abc.abstractmethod
    def owner(self) -> ContainerSymbol:
        raise NotImplementedError

    @property
    def module(self) -> Module:
        if isinstance(self.owner, Module):
            return cast(Module, self.owner)
        return cast(OwnedSymbol, self.owner).module


class MangledSymbol(OwnedSymbol, abc.ABC):
    @property
    @abc.abstractmethod
    def mangled_name(self) -> str:
        raise NotImplementedError


class ContainerSymbol(Symbol, abc.ABC):
    """ Abstract base for container symbols """

    def __init__(self):
        self.__members = []
        self.__scope = LexicalScope()

    @property
    def scope(self) -> LexicalScope:
        return self.__scope

    @property
    def members(self) -> Sequence[OwnedSymbol]:
        return self.__members

    def add_member(self, symbol: OwnedSymbol):
        self.__members.append(symbol)
        self.__scope.append(symbol)


class GenericSymbol(NamedSymbol, abc.ABC):
    @property
    def is_generic(self) -> bool:
        if self.generic_parameters:
            return True
        return any(arg.is_generic for arg in self.generic_arguments)

    @property
    @abc.abstractmethod
    def definition(self) -> GenericSymbol:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def generic_parameters(self) -> Sequence[GenericParameter]:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def generic_arguments(self) -> Sequence[Type]:
        raise NotImplementedError

    @abc.abstractmethod
    def instantiate(self, module: Module, generic_arguments: Sequence[Type], location: Location):
        raise NotImplementedError

    def __str__(self):
        arguments = None
        if self.generic_arguments:
            arguments = ', '.join(str(arg) for arg in self.generic_arguments)
        if self.generic_parameters:
            arguments = ', '.join(str(arg) for arg in self.generic_parameters)

        if arguments:
            return f'{self.name}[{arguments}]'
        return super(GenericSymbol, self).__str__()


class GenericParameter(NamedSymbol, abc.ABC):
    pass


class Value(Symbol, abc.ABC):
    """ Abstract base for all values """

    def __init__(self, value_type: Type, location: Location):
        self.__location = location
        self.__type = value_type

    @property
    def type(self) -> Type:
        return self.__type

    @property
    def location(self) -> Location:
        return self.__location

    @location.setter
    def location(self, value: Location):
        self.__location = value


class Module(NamedSymbol, ContainerSymbol):
    def __init__(self, context: SemanticContext, name, location: Location):
        super(Module, self).__init__()

        self.__name = name
        self.__location = location
        self.__instances = {}  # Map of all generic instances
        self.__functions = []
        self.__types = []

    @property
    def name(self) -> str:
        return self.__name

    @property
    def location(self) -> Location:
        return self.__location

    @property
    def functions(self) -> Sequence[Function]:
        return self.__functions

    def add_function(self, func: Function):
        self.__functions.append(func)

    def find_instance(self, generic: GenericSymbol, generic_arguments):
        key = (generic, tuple(generic_arguments))
        return self.__instances.get(key)

    def register_instance(self, generic: GenericSymbol, generic_arguments, instance: GenericSymbol):
        key = (generic, tuple(generic_arguments))
        self.__instances[key] = instance


class Type(MangledSymbol, GenericSymbol, OwnedSymbol, ContainerSymbol, abc.ABC):
    """ Abstract base for all types """

    def __init__(self, owner: ContainerSymbol, name: str, location: Location, *,
                 generic_parameters=None, generic_arguments=None, definition=None):
        super(Type, self).__init__()

        self.__owner = owner
        self.__name = name
        self.__location = location
        self.__generic_parameters = tuple(generic_parameters or [])
        self.__generic_arguments = tuple(generic_arguments or [])
        self.__definition = definition

    @property
    def owner(self) -> ContainerSymbol:
        return self.__owner

    @property
    def name(self) -> str:
        return self.__name

    @cached_property
    def mangled_name(self) -> str:
        context = MangledContext()
        return context.mangle(self)

    @property
    def definition(self) -> Type:
        return self.__definition

    @property
    def generic_parameters(self) -> Sequence[GenericParameter]:
        return self.__generic_parameters

    @property
    def generic_arguments(self) -> Sequence[Type]:
        return self.__generic_arguments

    @property
    def location(self) -> Location:
        return self.__location

    @property
    def is_pointer(self) -> True:
        return False

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return not (self == other)

    def instantiate(self, module: Module, generic_arguments: Sequence[Type], location: Location):
        raise Diagnostic(location, DiagnosticSeverity.Error, "Can not instantiate non generic type")


class VoidType(Type):
    def __init__(self, owner: ContainerSymbol, location: Location):
        super(VoidType, self).__init__(owner, 'void', location)


class BooleanType(Type):
    def __init__(self, owner: ContainerSymbol, location: Location):
        super(BooleanType, self).__init__(owner, 'bool', location)


class IntegerType(Type):
    def __init__(self, owner: ContainerSymbol, location: Location):
        super(IntegerType, self).__init__(owner, 'int', location)


class ClassType(Type):
    @property
    def is_pointer(self) -> bool:
        return True

    @property
    def fields(self) -> Sequence[Field]:
        return tuple(field for field in self.members if isinstance(field, Field))

    def instantiate(self, module: Module, generic_arguments: Sequence[Type], location: Location):
        instance = module.find_instance(self.definition or self, generic_arguments)
        if not instance:
            context = InstantiateContext(module)
            context.aggregate(self.generic_parameters, generic_arguments)
            instance = ClassType(module, self.name, self.location, generic_arguments=generic_arguments, definition=self)
            context.register(self, instance)

            for member in self.members:
                new_member = context.instantiate(member, location)
                instance.add_member(new_member)

        module.register_instance(self.definition or self, generic_arguments, instance)
        return instance


class StructType(Type):
    def instantiate(self, module: Module, generic_arguments: Sequence[Type], location: Location):
        instance = module.find_instance(self.definition or self, generic_arguments)
        if not instance:
            context = InstantiateContext(module)
            context.aggregate(self.generic_parameters, generic_arguments)
            return StructType(module, self.name, self.location, generic_arguments=generic_arguments, definition=self)
        module.register_instance(self.definition or self, generic_arguments, instance)
        return instance


class FunctionType(Type):
    def __init__(self, owner: ContainerSymbol, parameters: Sequence[Type], return_type: Type, location: Location):
        super(FunctionType, self).__init__(owner, "Function", location)

        assert return_type is not None
        assert all(param_type is not None for param_type in parameters)

        self.__return_type = return_type
        self.__parameters = parameters

    @property
    def return_type(self) -> Type:
        return self.__return_type

    @property
    def parameters(self) -> Sequence[Type]:
        return self.__parameters

    def __eq__(self, other):
        if not isinstance(other, FunctionType):
            return False
        is_return_equal = self.return_type == other.return_type
        return is_return_equal and all(
            param == other_param for param, other_param in zip(self.parameters, other.parameters))

    def __hash__(self):
        return id(self)

    def __str__(self):
        parameters = ', '.join(str(param_type) for param_type in self.parameters)
        return f"({parameters}) -> {self.return_type}"


class GenericType(GenericParameter, Type):
    pass


class TargetValue(Value, abc.ABC):
    pass


class Parameter(OwnedSymbol, TargetValue):
    def __init__(self, owner: Function, name: str, param_type: Type):
        super(Parameter, self).__init__(param_type, owner.location)

        self.__owner = owner
        self.__name = name

    @property
    def owner(self) -> Function:
        return self.__owner

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, value: str):
        self.__name = value

    def __str__(self):
        return f'{self.name}: {self.type}'


class Variable(NamedSymbol, TargetValue):
    def __init__(self, name: str, type: Type, location: Location):
        super(Variable, self).__init__(type, location)

        self.__name = name

    @property
    def name(self) -> str:
        return self.__name

    def __str__(self):
        return self.name


class Function(MangledSymbol, GenericSymbol, OwnedSymbol, Value):
    def __init__(self, owner: ContainerSymbol, name: str, func_type: FunctionType, location: Location, *,
                 generic_parameters=None, generic_arguments=None, definition=None):
        super(Function, self).__init__(func_type, location)
        self.__owner = owner
        self.__name = name
        self.__parameters = [
            Parameter(self, f'arg{idx}', param_type) for idx, param_type in enumerate(func_type.parameters)
        ]
        self.__statement = None
        self.__generic_parameters = tuple(generic_parameters or [])
        self.__generic_arguments = tuple(generic_arguments or [])
        self.__definition = definition
        self.__variables = []

        self.module.add_function(self)

    @property
    def owner(self) -> ContainerSymbol:
        return self.__owner

    @property
    def name(self) -> str:
        return self.__name

    @cached_property
    def mangled_name(self) -> str:
        context = MangledContext()
        return context.mangle(self)

    @property
    def definition(self) -> Function:
        return self.__definition

    @property
    def generic_parameters(self) -> Sequence[GenericParameter]:
        return self.__generic_parameters

    @property
    def generic_arguments(self) -> Sequence[Type]:
        return self.__generic_arguments

    @property
    def function_type(self) -> FunctionType:
        return cast(FunctionType, self.type)

    @property
    def parameters(self) -> Sequence[Parameter]:
        return self.__parameters

    @property
    def return_type(self) -> Type:
        return self.function_type.return_type

    @property
    def variables(self) -> Sequence[Variable]:
        return self.__variables

    @property
    def statement(self) -> Optional[Statement]:
        return self.__statement

    @statement.setter
    def statement(self, statement: Optional[Statement]):
        self.__statement = statement

    def __str__(self):
        parameters = ', '.join(str(param) for param in self.parameters)
        return f'{self.name}({parameters}) -> {self.return_type}'

    def instantiate(self, module: Module, generic_arguments: Sequence[Type], location: Location):
        instance = module.find_instance(self.definition or self, generic_arguments)
        if not instance:
            context = InstantiateContext(module)
            context.aggregate(self.generic_parameters, generic_arguments)
            function_type = context.instantiate(self.function_type, location)
            instance = Function(
                module, self.name, function_type, self.location, generic_arguments=generic_arguments, definition=self)
            context.register(self, instance)

            for original_param, instance_param in zip(self.parameters, instance.parameters):
                context.register(original_param, instance_param)

            for original_var in self.variables:
                new_type = context.instantiate(original_var.type, location)
                new_var = instance.add_variables(original_var.name, new_type, original_var.location)
                context.register(original_var, new_var)

            instance.statement = context.instantiate(self.statement, location)

        module.register_instance(self.definition or self, generic_arguments, instance)
        return instance

    def add_variables(self, name: str, type: Type, location: Location):
        var = Variable(name, type, location)
        self.__variables.append(var)
        return var


class Overload(NamedSymbol):
    def __init__(self, name: str, function: Function):
        self.__name = name
        self.__functions = [function]

    @property
    def functions(self) -> Sequence[Function]:
        return self.__functions

    @property
    def name(self) -> str:
        return self.__name

    @property
    def location(self) -> Location:
        return self.functions[0].location

    def append(self, func: Function):
        if func not in self.__functions:
            self.__functions.append(func)

    def extend(self, overload: Overload):
        for function in overload.functions:
            self.append(function)


class Field(OwnedSymbol):
    def __init__(self, owner: Type, name: str, field_type: Type, location: Location):
        self.__owner = owner
        self.__name = name
        self.__type = field_type
        self.__location = location

    @property
    def owner(self) -> Type:
        return self.__owner

    @property
    def name(self) -> str:
        return self.__name

    @property
    def type(self) -> Type:
        return self.__type

    @property
    def location(self) -> Location:
        return self.__location


class IntegerConstant(Value):
    def __init__(self, value_type: IntegerType, value: int, location: Location):
        super(IntegerConstant, self).__init__(value_type, location)

        self.value = value

    def __str__(self):
        return str(self.value)


class BooleanConstant(Value):
    def __init__(self, value_type: BooleanType, value: bool, location: Location):
        super(BooleanConstant, self).__init__(value_type, location)

        self.value = value

    def __str__(self):
        return "True" if self.value else "False"


class CallInstruction(Value):
    def __init__(self, func: Function, arguments: Sequence[Value], location: Location):
        super(CallInstruction, self).__init__(func.return_type, location)

        self.function = func
        self.arguments = arguments

    def __str__(self):
        arguments = ', '.join(str(arg) for arg in self.arguments)
        return f'{self.function.name}({arguments})'


class NewInstruction(Value):
    def __init__(self, return_type: Type, arguments: Sequence[Value], location: Location):
        super(NewInstruction, self).__init__(return_type, location)

        self.arguments = arguments

    def __str__(self):
        arguments = ', '.join(str(arg) for arg in self.arguments)
        return f'{self.type}({arguments})'


class BoundedValue(Value, abc.ABC):
    def __init__(self, instance: Value, value_type: Type, location: Location):
        super(BoundedValue, self).__init__(value_type, location)

        self.instance = instance


class BoundedField(BoundedValue, TargetValue):
    def __init__(self, instance: Value, field: Field, location: Location):
        super(BoundedField, self).__init__(instance, field.type, location)

        self.field = field

    def __str__(self):
        return f'{self.instance}.{self.field.name}'


class Statement:
    def __init__(self, location: Location):
        self.location = location


class BlockStatement(Statement):
    def __init__(self, statements: Sequence[Statement], location: Location):
        super(BlockStatement, self).__init__(location)

        self.statements = statements


class PassStatement(Statement):
    pass


class ReturnStatement(Statement):
    def __init__(self, value: Optional[Value], location=None):
        super(ReturnStatement, self).__init__(location)

        self.value = value


class ExpressionStatement(Statement):
    def __init__(self, value: Value):
        super(ExpressionStatement, self).__init__(value.location)

        self.value = value


class ConditionStatement(Statement):
    def __init__(self, condition: Value, then_statement: Statement, else_statement: Optional[Statement], location):
        super(ConditionStatement, self).__init__(location)

        self.condition = condition
        self.then_statement = then_statement
        self.else_statement = else_statement


class WhileStatement(Statement):
    def __init__(self, condition: Value, then_statement: Statement, else_statement: Optional[Statement], location):
        super(WhileStatement, self).__init__(location)

        self.condition = condition
        self.then_statement = then_statement
        self.else_statement = else_statement


class AssignStatement(Statement):
    def __init__(self, target: TargetValue, source: Value, location: Location):
        super(AssignStatement, self).__init__(location)

        self.target = target
        self.source = source


class LazyDict(dict):
    def __init__(self, seq=None, *, builder, initializer=None, **kwargs):
        super().__init__(seq or (), **kwargs)
        self.__builder = builder
        self.__initializer = initializer or (lambda x: None)

    def __getitem__(self, item):
        try:
            return super().__getitem__(item)
        except KeyError:
            value = self.__builder(item)
            if value is None:
                raise KeyError(item)
            self[item] = value
            self.__initializer(item)
            return value

    def __contains__(self, item):
        try:
            self[item]
        except KeyError:
            return False
        return True


class ModuleCodegen:
    def __init__(self, context: SemanticContext, name='<stdin>'):
        self.llvm_module = ir.Module(name)
        self.llvm_module.triple = binding.Target.from_default_triple().triple

        # names to symbol
        self.types = {}
        self.functions = MultiDict()

        # symbol to llvm
        self.llvm_types = LazyDict(builder=self.declare_type, initializer=self.initialize_type)
        self.llvm_functions = LazyDict(builder=self.declare_function)

        # builtins functions
        self.context = context
        self.builtins = BuiltinsCodegen(self)

    def __str__(self):
        return str(self.llvm_module)

    @property
    def llvm_context(self) -> ir.Context:
        return self.llvm_module.context

    @cached_property
    def llvm_size(self) -> ir.IntType:
        return ir.IntType(32)

    @cached_property
    def llvm_void(self) -> ir.VoidType:
        return ir.VoidType()

    @cached_property
    def llvm_opaque(self):
        return ir.IntType(8).as_pointer()

    @cached_property
    def llvm_malloc(self):
        llvm_type = ir.FunctionType(self.llvm_opaque, [self.llvm_size])
        return ir.Function(self.llvm_module, llvm_type, 'malloc')

    @multimethod
    def declare_type(self, type_symbol: Type):
        raise Diagnostic(type_symbol.location, DiagnosticSeverity.Error, "Not implemented type conversion to LLVM")

    @multimethod
    def declare_type(self, _: VoidType):
        return ir.VoidType()

    @multimethod
    def declare_type(self, _: BooleanType):
        return ir.IntType(1)

    @multimethod
    def declare_type(self, _: IntegerType):
        return ir.IntType(64)

    @multimethod
    def declare_type(self, type_symbol: ClassType):
        """ class = pointer to struct { fields... } """
        llvm_struct = self.llvm_context.get_identified_type(type_symbol.mangled_name)
        return llvm_struct.as_pointer()

    def declare_function(self, func: Function):
        llvm_return = self.llvm_types[func.return_type]
        llvm_params = [self.llvm_types[param.type] for param in func.parameters]
        llvm_type = ir.FunctionType(llvm_return, llvm_params)
        llvm_func = ir.Function(self.llvm_module, llvm_type, func.mangled_name)
        llvm_func.linkage = 'internal'

        for llvm_arg, param in zip(llvm_func.args, func.parameters):
            llvm_arg.name = param.name

        return llvm_func

    @multimethod
    def initialize_type(self, _: Type):
        pass

    @multimethod
    def initialize_type(self, type_symbol: ClassType):
        llvm_struct = self.llvm_context.get_identified_type(type_symbol.mangled_name)
        llvm_fields = (self.llvm_types[field.type] for field in type_symbol.fields)
        llvm_struct.set_body(*llvm_fields)

    def emit(self, module: Module):
        for func in module.functions:
            if not func.is_generic:
                self.emit_function(func)

    def emit_function(self, func: Function):
        llvm_func = self.llvm_functions[func]
        if func.statement:
            builder = FunctionCodegen(self, func, llvm_func)
            builder.emit_statement(func.statement)
        if func.name == 'main':
            self.emit_main(func)
        return llvm_func

    def emit_main(self, func: Function):
        # main prototype
        llvm_type = ir.FunctionType(ir.IntType(32), [
            ir.IntType(32),
            ir.IntType(8).as_pointer().as_pointer()
        ])
        llvm_func = ir.Function(self.llvm_module, llvm_type, name="main")

        argc, argv = llvm_func.args
        argc.name, argv.name = 'argc', 'argv'

        llvm_entry = llvm_func.append_basic_block('entry')
        llvm_builder = ir.IRBuilder(llvm_entry)

        if func.parameters:
            raise Diagnostic(func.location, DiagnosticSeverity.Error, f"Main function must have zero arguments")
        else:
            arguments = []
        llvm_result = llvm_builder.call(self.llvm_functions[func], arguments)

        if not isinstance(func.return_type, (VoidType, IntegerType)):
            raise Diagnostic(func.location, DiagnosticSeverity.Error,
                             f"Return type of main function must be ‘int’ or ‘void’")
        elif isinstance(func.return_type, VoidType):
            llvm_result = ir.Constant(ir.IntType(32), 0)

        if llvm_result.type.width > 32:
            llvm_result = llvm_builder.trunc(llvm_result, ir.IntType(32))
        elif llvm_result.type.width < 32:
            llvm_result = llvm_builder.sext(llvm_result, ir.IntType(32))
        llvm_builder.ret(llvm_result)


class FunctionCodegen:
    def __init__(self, parent: ModuleCodegen, func: Function, llvm_func: ir.Function):
        self.parent = parent
        self.function = func
        self.llvm_function = llvm_func
        self.llvm_variables = {}

        llvm_entry = llvm_func.append_basic_block('entry')
        self.llvm_builder = ir.IRBuilder(llvm_entry)

        for param, llvm_arg in zip(func.parameters, llvm_func.args):
            llvm_alloca = self.llvm_builder.alloca(llvm_arg.type)
            self.llvm_variables[param] = llvm_alloca
            self.llvm_builder.store(llvm_arg, llvm_alloca)

        for var in func.variables:
            llvm_alloca = self.llvm_builder.alloca(self.llvm_types[var.type])
            self.llvm_variables[var] = llvm_alloca

    @property
    def llvm_module(self) -> ir.Module:
        return self.parent.llvm_module

    @property
    def llvm_types(self) -> Mapping[Type, ir.Type]:
        return self.parent.llvm_types

    @property
    def llvm_functions(self) -> Mapping[Function, ir.Function]:
        return self.parent.llvm_functions

    def emit_sizeof(self, type: Type):
        llvm_type = self.parent.llvm_types[type]
        llvm_pointer = llvm_type.as_pointer() if not type.is_pointer else llvm_type
        llvm_size = self.llvm_builder.gep(ir.Constant(llvm_pointer, None), [ir.Constant(ir.IntType(32), 0)])
        return self.llvm_builder.ptrtoint(llvm_size, self.parent.llvm_size)

    @multimethod
    def emit_statement(self, statement: Statement) -> bool:
        """

        :param statement:
        :return: True, if statement is terminated
        """
        raise Diagnostic(statement.location, DiagnosticSeverity.Error, "Not implemented statement conversion to LLVM")

    @multimethod
    def emit_statement(self, statement: BlockStatement) -> bool:
        for child in statement.statements:
            if self.emit_statement(child):
                return True

    @multimethod
    def emit_statement(self, statement: PassStatement) -> bool:
        return False  # :D

    @multimethod
    def emit_statement(self, statement: ReturnStatement) -> bool:
        if statement.value:
            llvm_value = self.emit_value(statement.value)
            self.llvm_builder.ret(llvm_value)
        else:
            self.llvm_builder.ret_void()
        return True

    @multimethod
    def emit_statement(self, statement: ExpressionStatement) -> bool:
        self.emit_value(statement.value)
        return False

    @multimethod
    def emit_statement(self, statement: ConditionStatement) -> bool:
        llvm_cond = self.emit_value(statement.condition)

        with self.llvm_builder.if_else(llvm_cond) as (then, otherwise):
            # emit instructions for when the predicate is true
            with then:
                is_terminated = self.emit_statement(statement.then_statement)

            # emit instructions for when the predicate is false
            with otherwise:
                if statement.else_statement:
                    is_terminated = self.emit_statement(statement.else_statement) and is_terminated
                else:
                    is_terminated = False

        if is_terminated:
            self.llvm_function.blocks.remove(self.llvm_builder.basic_block)
        return is_terminated

    @multimethod
    def emit_statement(self, statement: WhileStatement) -> bool:
        # condition block
        llvm_cond_block = self.llvm_builder.append_basic_block('while.cond')
        self.llvm_builder.branch(llvm_cond_block)
        self.llvm_builder.position_at_end(llvm_cond_block)
        llvm_cond = self.emit_value(statement.condition)
        if statement.else_statement:
            with self.llvm_builder.goto_entry_block():
                llvm_flag = self.llvm_builder.alloca(ir.IntType(1))
            self.llvm_builder.store(ir.Constant(ir.IntType(1), False), llvm_flag)
        else:
            llvm_flag = None

        # then block
        llvm_then_block = self.llvm_builder.append_basic_block('while.then')
        self.llvm_builder.position_at_end(llvm_then_block)
        if statement.else_statement:
            self.llvm_builder.store(ir.Constant(ir.IntType(1), True), llvm_flag)
        self.emit_statement(statement.then_statement)
        if not self.llvm_builder.block.is_terminated:
            self.llvm_builder.branch(llvm_cond_block)

        # continue block
        llvm_continue_block = self.llvm_builder.append_basic_block('while.continue')

        # condition -> then | continue
        self.llvm_builder.position_at_end(llvm_cond_block)
        self.llvm_builder.cbranch(llvm_cond, llvm_then_block, llvm_continue_block)

        if statement.else_statement:
            llvm_else_block = self.llvm_builder.append_basic_block('while.else')

            self.llvm_builder.position_at_end(llvm_else_block)
            self.emit_statement(statement.else_statement)

            llvm_next_block = self.llvm_builder.append_basic_block('while.next')
            if not self.llvm_builder.block.is_terminated:
                self.llvm_builder.branch(llvm_next_block)

            self.llvm_builder.position_at_end(llvm_continue_block)
            llvm_flag = self.llvm_builder.load(llvm_flag)
            self.llvm_builder.cbranch(llvm_flag, llvm_else_block, llvm_next_block)

            self.llvm_builder.position_at_end(llvm_next_block)
        else:
            self.llvm_builder.position_at_end(llvm_continue_block)

        return False

    @multimethod
    def emit_statement(self, statement: AssignStatement) -> bool:
        llvm_source = self.emit_value(statement.source)
        llvm_target = self.emit_target(statement.target)
        self.llvm_builder.store(llvm_source, llvm_target)
        return False

    @multimethod
    def emit_value(self, value: Value):
        raise Diagnostic(value.location, DiagnosticSeverity.Error, "Not implemented value conversion to LLVM")

    @multimethod
    def emit_value(self, value: Parameter):
        llvm_alloca = self.llvm_variables[value]
        return self.llvm_builder.load(llvm_alloca)

    @multimethod
    def emit_value(self, value: Variable):
        llvm_alloca = self.llvm_variables[value]
        return self.llvm_builder.load(llvm_alloca)

    @multimethod
    def emit_value(self, value: BoundedField):
        llvm_offset = self.emit_offset(value)
        return self.llvm_builder.load(llvm_offset)

    @multimethod
    def emit_value(self, value: IntegerConstant):
        llvm_type = self.llvm_types[value.type]
        return ir.Constant(llvm_type, value.value)

    @multimethod
    def emit_value(self, value: BooleanConstant):
        llvm_type = self.llvm_types[value.type]
        return ir.Constant(llvm_type, value.value)

    @multimethod
    def emit_value(self, value: CallInstruction):
        emitter = self.parent.builtins.emitters.get(value.function)
        if emitter:
            return emitter(self, value.function, value.arguments, value.location)

        llvm_args = [self.emit_value(arg) for arg in value.arguments]
        llvm_func = self.llvm_functions[value.function]
        return self.llvm_builder.call(llvm_func, llvm_args)

    @multimethod
    def emit_value(self, value: NewInstruction):
        if value.arguments:
            raise Diagnostic(value.location, DiagnosticSeverity.Error, "Not implemented constructors")

        llvm_type = self.llvm_types[value.type]
        if value.type.is_pointer:
            # allocate memory for type from heap (GC in future)
            llvm_size = self.emit_sizeof(value.type)
            llvm_instance = self.llvm_builder.call(self.parent.llvm_malloc, [llvm_size])
            llvm_instance = self.llvm_builder.bitcast(llvm_instance, self.parent.llvm_types[value.type])
            return llvm_instance
        else:
            return ir.Constant(llvm_type, None)

    @multimethod
    def emit_target(self, value: TargetValue):
        raise Diagnostic(value.location, DiagnosticSeverity.Error, "Not implemented target conversion to LLVM")

    @multimethod
    def emit_target(self, value: Parameter):
        return self.llvm_variables[value]

    @multimethod
    def emit_target(self, value: Variable):
        return self.llvm_variables[value]

    @multimethod
    def emit_target(self, value: BoundedField):
        return self.emit_offset(value)

    def emit_offset(self, value: BoundedField):
        index = cast(ClassType, value.instance.type).fields.index(value.field)
        llvm_instance = self.emit_value(value.instance)

        return self.llvm_builder.gep(llvm_instance, [
            ir.Constant(ir.IntType(32), 0),
            ir.Constant(ir.IntType(32), index),
        ])


class BuiltinsCodegen:
    def __init__(self, parent: ModuleCodegen):
        builtins_module = parent.context.builtins_module
        integer_type = parent.context.integer_type

        self.emitters = {
            integer_type.scope.resolve('__pos__').functions[0]: self.int_pos,
            integer_type.scope.resolve('__neg__').functions[0]: self.int_neg,
            integer_type.scope.resolve('__add__').functions[0]: self.int_add,
            integer_type.scope.resolve('__sub__').functions[0]: self.int_sub,
            integer_type.scope.resolve('__mul__').functions[0]: self.int_mul,
        }

    @staticmethod
    def int_pos(self: FunctionCodegen, func: Function, arguments: Sequence[Value], location: Location):
        return self.emit_value(arguments[0])

    @staticmethod
    def int_neg(self: FunctionCodegen, func: Function, arguments: Sequence[Value], location: Location):
        return self.llvm_builder.neg(self.emit_value(arguments[0]))

    @staticmethod
    def int_add(self: FunctionCodegen, func: Function, arguments: Sequence[Value], location: Location):
        llvm_args = (self.emit_value(arg) for arg in arguments)
        return self.llvm_builder.add(*llvm_args)

    @staticmethod
    def int_sub(self: FunctionCodegen, func: Function, arguments: Sequence[Value], location: Location):
        llvm_args = (self.emit_value(arg) for arg in arguments)
        return self.llvm_builder.sub(*llvm_args)

    @staticmethod
    def int_mul(self: FunctionCodegen, func: Function, arguments: Sequence[Value], location: Location):
        llvm_args = (self.emit_value(arg) for arg in arguments)
        return self.llvm_builder.mul(*llvm_args)


def load_source_content(location: Location, before=2, after=2):
    """ Load selected line and it's neighborhood lines """
    try:
        with open(location.filename, 'r', encoding='utf-8') as stream:
            at_before = max(0, location.begin.line - before)
            at_after = location.end.line + after

            idx = 0
            results = []
            for idx, line in itertools.islice(enumerate(stream), at_before, at_after):
                results.append((idx + 1, line.rstrip("\n")))
    except IOError:
        return []
    else:
        results.append([idx + 2, ""])
        return results


def show_source_lines(location: Location, before=2, after=2):
    """
    Convert selected lines to error message, e.g.:

    ```
        1 : from module import system =
          : --------------------------^
    ```
    """
    stream = io.StringIO()

    strings = load_source_content(location, before, after)
    if not strings:
        return

    width = max(max(len(str(idx)) for idx, _ in strings), 5)
    for line, string in strings:
        s_line = str(line).rjust(width)

        stream.write(ANSI_COLOR_CYAN)
        stream.write(s_line)
        stream.write(" : ")
        stream.write(ANSI_COLOR_BLUE)
        for column, char in enumerate(string):
            column += 1
            is_error = False
            if location.begin.line == line:
                is_error = column >= location.begin.column
            if location.end.line == line:
                is_error = is_error and column <= location.end.column

            if is_error:
                stream.write(ANSI_COLOR_RED)
            else:
                stream.write(ANSI_COLOR_GREEN)
            stream.write(char)

        stream.write(ANSI_COLOR_RESET)
        stream.write("\n")

        # write error line
        if location.begin.line <= line <= location.end.line:
            stream.write("·" * width)
            stream.write(" : ")

            for column, char in itertools.chain(enumerate(string), ((len(string), None),)):
                column += 1

                is_error = False
                if location.begin.line == line:
                    is_error = column >= location.begin.column
                if location.end.line == line:
                    is_error = is_error and column <= location.end.column

                if is_error:
                    stream.write(ANSI_COLOR_RED)
                    stream.write("^")
                    stream.write(ANSI_COLOR_RESET)
                elif char is not None:
                    stream.write("·")
            stream.write("\n")

    return stream.getvalue()


def build(filenames: Sequence[str]):
    # initialize llvm targets
    binding.initialize()
    binding.initialize_native_target()
    binding.initialize_native_asmparser()
    binding.initialize_native_asmprinter()

    # initialize semantic context
    context = SemanticContext()
    for filename in filenames:
        module = context.open(filename)
        generator = ModuleCodegen(context, module.name)
        generator.emit(module)
        print(generator)


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
            return 2

    return wrapper


def process_errors(action):
    @functools.wraps(action)
    def wrapper(*args, **kwargs):
        try:
            return action(*args, **kwargs)
        except BootstrapError as ex:
            logger.error(str(ex))
            return 1
        except Exception as ex:
            logger.exception(ex)
            return 1

    return wrapper


def initialize_logging():
    """ Prepare rules for loggers """
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
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)


def main():
    initialize_logging()

    key_level = '__level__'
    key_pdb = '__pdb__'
    # noinspection PyProtectedMember
    logger_levels = list(map(str.lower, logging._nameToLevel.keys()))
    logger_default = "warning"

    parser = argparse.ArgumentParser(prog=APPLICATION_NAME)
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s 0.1')
    parser.add_argument('--pdb', dest=key_pdb, action='store_true', help="post-mortem mode")
    parser.add_argument('-l', '--level', dest=key_level, choices=logger_levels, default=logger_default)
    parser.add_argument('filenames', type=str, nargs='+', help="input files")

    # parse arguments
    kwargs = parser.parse_args().__dict__
    is_pdb = kwargs.pop(key_pdb, False)
    logger.setLevel(kwargs.pop(key_level, logger_default).upper())

    action = build
    if is_pdb:  # enable pdb if required
        action = process_pdb(action)
    if not sys.gettrace():
        action = process_errors(action)
    sys.exit(action(**kwargs) or 0)


if __name__ == '__main__':
    main()
