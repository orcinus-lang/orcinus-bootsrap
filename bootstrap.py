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
import io
import itertools
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Sequence, Iterator, Optional, cast, Mapping

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
        result = instance.__dict__[self.func.__name__] = self.func(instance)
        return result


class BootstrapError(Exception):
    pass


@enum.unique
class TokenID(enum.IntEnum):
    Name = enum.auto()
    Number = enum.auto()
    LeftParenthesis = enum.auto()
    RightParenthesis = enum.auto()
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
        (r'\.', TokenID.Dot),
        (r',', TokenID.Comma),
        (r':', TokenID.Colon),
        (r';', TokenID.Semicolon),
        (r'\-\>', TokenID.Then),

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
    MEMBERS_STARTS = (TokenID.Pass, TokenID.Def, TokenID.Class, TokenID.Struct)
    EXPRESSION_STARTS = (TokenID.Number, TokenID.Name, TokenID.LeftParenthesis)
    STATEMENT_STARTS = (TokenID.Pass, TokenID.Return, TokenID.While, TokenID.If) + EXPRESSION_STARTS

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
            name
        """
        tok_name = self.consume(TokenID.Name)

        # noinspection PyArgumentList
        return NamedTypeAST(name=tok_name.value, location=tok_name.location)

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
        """
        if self.match(TokenID.Def):
            return self.parse_function()
        elif self.match(TokenID.Class):
            return self.parse_class()
        elif self.match(TokenID.Struct):
            return self.parse_struct()
        elif self.match(TokenID.Pass):
            return self.parse_pass_member()

        self.match(*self.MEMBERS_STARTS)

    def parse_class(self) -> ClassAST:
        """
        class:
            'class' Name ':' type_members
        """
        self.consume(TokenID.Class)
        tok_name = self.consume(TokenID.Name)
        members = self.parse_type_members()

        # noinspection PyArgumentList
        return ClassAST(
            name=tok_name.value,
            members=members,
            location=tok_name.location
        )

    def parse_struct(self) -> StructAST:
        """
        struct:
            'struct' Name ':' type_members
        """
        self.consume(TokenID.Struct)
        tok_name = self.consume(TokenID.Name)
        members = self.parse_type_members()

        # noinspection PyArgumentList
        return StructAST(
            name=tok_name.value,
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

        if self.match(TokenID.Ellipsis):
            self.consume(TokenID.Ellipsis)
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

    def parse_function(self) -> FunctionAST:
        """
        function:
            'def' Name '(' parameters ')' [ '->' type ] ':' NewLine function_statement
        """
        self.consume(TokenID.Def)
        tok_name = self.consume(TokenID.Name)
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
            Name ':' type
        """
        tok_name = self.consume(TokenID.Name)
        self.consume(TokenID.Colon)
        param_type = self.parse_type()

        # noinspection PyArgumentList
        return ParameterAST(name=tok_name.value, type=param_type, location=tok_name.location)

    def parse_function_statement(self) -> Optional[StatementAST]:
        """
        function_statement:
            '...' EndFile
            NewLine block_statement
        """
        if self.try_consume(TokenID.Ellipsis):
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
        value = self.parse_expression()
        self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return ExpressionStatementAST(value=value, location=value.location)

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
        return self.parse_atom_expression()

    def parse_atom_expression(self) -> ExpressionAST:
        """
        atom:
             number_expression
             name_expression
             call_expression
             parenthesis_expression
        """
        if self.match(TokenID.Number):
            atom = self.parse_number_expression()
        elif self.match(TokenID.Name):
            atom = self.parse_name_expression()
        elif self.match(TokenID.LeftParenthesis):
            atom = self.parse_parenthesis_expression()
        else:
            self.match(*self.EXPRESSION_STARTS)
            raise NotImplementedError  # Make linter happy

        if self.match(TokenID.LeftParenthesis):
            return self.parse_call_expression(atom)
        return atom

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
            expression '(' arguments ')'
        """
        self.consume(TokenID.LeftParenthesis)
        arguments = self.parse_arguments()
        tok_close = self.consume(TokenID.RightParenthesis)
        location = value.location + tok_close.location

        # noinspection PyArgumentList
        return CallExpressionAST(value=value, arguments=arguments, location=location)

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
class NamedTypeAST(TypeAST):
    name: str


@dataclass(unsafe_hash=True, frozen=True)
class MemberAST(NodeAST):
    pass


@dataclass(unsafe_hash=True, frozen=True)
class PassMemberAST(MemberAST):
    pass


@dataclass(unsafe_hash=True, frozen=True)
class StructAST(MemberAST):
    name: str
    members: Sequence[MemberAST]


@dataclass(unsafe_hash=True, frozen=True)
class ClassAST(MemberAST):
    name: str
    members: Sequence[MemberAST]


@dataclass(unsafe_hash=True, frozen=True)
class ParameterAST(NodeAST):
    name: str
    type: TypeAST


@dataclass(unsafe_hash=True, frozen=True)
class FunctionAST(MemberAST):
    name: str
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
class ExpressionAST(NodeAST):
    pass


@dataclass(unsafe_hash=True, frozen=True)
class IntegerExpressionAST(ExpressionAST):
    value: int


@dataclass(unsafe_hash=True, frozen=True)
class NamedExpressionAST(ExpressionAST):
    name: str


@dataclass(unsafe_hash=True, frozen=True)
class CallExpressionAST(ExpressionAST):
    value: ExpressionAST
    arguments: Sequence[ExpressionAST]


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
        self.filenames = {}

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
    def builtins_model(self) -> SemanticModel:
        return self.load('__builtins__')

    @cached_property
    def builtins_module(self) -> Module:
        return self.builtins_model.module

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

    def open(self, filename) -> SemanticModel:
        """ Open module from file """
        module_name = self.get_module_name(filename)

        with open(filename, 'r', encoding='utf8') as stream:
            return self.__open_source(filename, module_name, stream)

    def load(self, module_name) -> SemanticModel:
        for path in self.paths:
            filename = self.convert_filename(module_name, path)
            try:
                with open(filename, 'r', encoding='utf8') as stream:
                    return self.__open_source(filename, module_name, stream)
            except IOError:
                logger.info(f"Not found module `{module_name}` in file `{filename}`")
                pass  # Continue

        raise BootstrapError(f'Not found module {module_name}')

    def __open_source(self, filename, module_name, stream):
        logger.info(f"Open `{module_name}` from file `{filename}`")

        if module_name in self.modules:
            return self.modules[module_name]

        parser = Parser(filename, stream)
        tree = parser.parse()

        model = SemanticModel(self, module_name, tree)
        self.modules[module_name] = self.filenames[filename] = model
        model.analyze()
        return model


class SemanticModel:
    def __init__(self, context: SemanticContext, module_name: str, tree: ModuleAST):
        self.context = context
        self.module_name = module_name
        self.tree = tree
        self.symbols = {}
        self.scopes = {}
        self.types = []
        self.functions = []

    @property
    def module(self) -> Module:
        return self.symbols[self.tree]

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

    def declare_symbol(self, node: NodeAST, scope: LexicalScope = None, parent: ContainerSymbol = None):
        symbol = self.annotate_symbol(node, parent)
        self.symbols[node] = symbol

        # Collect types and functions from model
        if isinstance(symbol, Type):
            self.types.append(symbol)
        elif isinstance(symbol, Function):
            self.functions.append(symbol)

        # Declare symbol in parent scope
        if scope is not None and isinstance(symbol, NamedSymbol):
            scope.append(symbol)

        # Add members
        if hasattr(node, 'members'):
            child_scope = self.scopes[node]
            for child in node.members:
                child_symbol = self.declare_symbol(child, child_scope, symbol)
                if child_symbol:
                    symbol.add_member(child_symbol)

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

        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented type resolving")

    @multimethod
    def annotate_symbol(self, node: NodeAST, parent: ContainerSymbol) -> Symbol:
        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented member declaration")

    # noinspection PyUnusedLocal
    @multimethod
    def annotate_symbol(self, node: ModuleAST, parent=None) -> Module:
        return Module(self.module_name, Location(node.location.filename))

    @multimethod
    def annotate_symbol(self, node: FunctionAST, parent: ContainerSymbol) -> Function:
        parameters = [self.resolve_type(param.type) for param in node.parameters]
        return_type = self.resolve_type(node.return_type)
        func_type = FunctionType(self.module, parameters, return_type, node.location)
        func = Function(parent, node.name, func_type, node.location)
        scope = self.scopes[node]

        for node_param, func_param in zip(node.parameters, func.parameters):
            func_param.name = node_param.name
            func_param.location = node_param.location

            self.symbols[node_param] = func_param
            scope.append(func_param)

        return func

    @multimethod
    def annotate_symbol(self, node: StructAST, parent: ContainerSymbol) -> Symbol:
        if self.module == self.context.builtins_module:
            if node.name == "int":
                return IntegerType(parent, node.location)
            elif node.name == "bool":
                return BooleanType(parent, node.location)
            elif node.name == "void":
                return VoidType(parent, node.location)

        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented struct declaration")

    def emit_functions(self, module: ModuleAST):
        for member in module.members:
            if isinstance(member, FunctionAST):
                self.emit_function(member)

    def emit_function(self, node: FunctionAST):
        func = self.symbols[node]
        if node.statement:
            func.statement = self.emit_statement(node.statement)

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

        return ConditionStatement(condition, then_statement, else_statement, node.location)

    @multimethod
    def emit_statement(self, node: WhileStatementAST) -> Statement:
        condition = self.emit_value(node.condition)
        then_statement = self.emit_statement(node.then_statement)
        else_statement = self.emit_statement(node.else_statement) if node.else_statement else None

        return WhileStatement(condition, then_statement, else_statement, node.location)

    @multimethod
    def emit_value(self, node: ExpressionAST) -> Value:
        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented value emitting")

    @multimethod
    def emit_value(self, node: IntegerExpressionAST) -> Value:
        return IntegerConstant(self.context.integer_type, node.value, node.location)

    @multimethod
    def emit_value(self, node: NamedExpressionAST) -> Value:
        if node.name in ['True', 'False']:
            return BooleanConstant(self.context.boolean_type, node.name == 'True', node.location)

        scope = self.scopes[node]
        symbol = scope.resolve(node.name)
        if not symbol:
            raise Diagnostic(
                node.location, DiagnosticSeverity.Error, f"Not found symbol `{node.name} in current scope`")
        elif isinstance(symbol, Parameter):
            return symbol

        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented value emitting")

    @multimethod
    def emit_value(self, node: CallExpressionAST) -> Value:
        arguments = [self.emit_value(arg) for arg in node.arguments]
        if not isinstance(node.value, NamedExpressionAST):
            raise Diagnostic(
                node.location, DiagnosticSeverity.Error, 'Not implemented object call')

        name = node.value.name
        scope: LexicalScope = self.scopes[node]
        symbol = scope.resolve(name)
        if not symbol:
            raise Diagnostic(
                node.location, DiagnosticSeverity.Error, f'Not found function `{name}` in current scope')

        if not isinstance(symbol, Overload):
            raise Diagnostic(
                node.location, DiagnosticSeverity.Error, 'Not implemented object call')

        if len(symbol.functions) != 1:
            raise Diagnostic(
                node.location, DiagnosticSeverity.Error, 'Not implemented function overloading')

        func = symbol.functions[0]
        return Call(func, arguments, node.location)


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
    def __init__(self, name, location: Location):
        super(Module, self).__init__()

        self.__name = name
        self.__location = location

    @property
    def name(self) -> str:
        return self.__name

    @property
    def location(self) -> Location:
        return self.__location


class Type(OwnedSymbol, ContainerSymbol, abc.ABC):
    """ Abstract base for all types """

    def __init__(self, owner: ContainerSymbol, name: str, location: Location):
        super(Type, self).__init__()

        self.__owner = owner
        self.__name = name
        self.__location = location

    @property
    def owner(self) -> ContainerSymbol:
        return self.__owner

    @property
    def name(self) -> str:
        return self.__name

    @property
    def location(self) -> Location:
        return self.__location

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return not (self == other)


class VoidType(Type):
    def __init__(self, owner: ContainerSymbol, location: Location):
        super(VoidType, self).__init__(owner, 'void', location)


class BooleanType(Type):
    def __init__(self, owner: ContainerSymbol, location: Location):
        super(BooleanType, self).__init__(owner, 'bool', location)


class IntegerType(Type):
    def __init__(self, owner: ContainerSymbol, location: Location):
        super(IntegerType, self).__init__(owner, 'int', location)


class FunctionType(Type):
    def __init__(self, owner: ContainerSymbol, parameters: Sequence[Type], return_type: Type, location: Location):
        super(FunctionType, self).__init__(owner, "Function", location)

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

    def __str__(self):
        parameters = ', '.join(str(param_type) for param_type in self.parameters)
        return f"({parameters}) -> {self.return_type}"


class Parameter(Value, OwnedSymbol):
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


class Function(Value, OwnedSymbol):
    def __init__(self, owner: ContainerSymbol, name: str, func_type: FunctionType, location: Location):
        super(Function, self).__init__(func_type, location)
        self.__owner = owner
        self.__name = name
        self.__parameters = [
            Parameter(self, f'arg{idx}', param_type) for idx, param_type in enumerate(func_type.parameters)
        ]
        self.__statement = None

    @property
    def owner(self) -> ContainerSymbol:
        return self.__owner

    @property
    def name(self) -> str:
        return self.__name

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
    def statement(self) -> Optional[Statement]:
        return self.__statement

    @statement.setter
    def statement(self, statement: Optional[Statement]):
        self.__statement = statement

    def __str__(self):
        parameters = ', '.join(str(param) for param in self.parameters)
        return f'{self.name}({parameters}) -> {self.return_type}'


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


class Call(Value):
    def __init__(self, func: Function, arguments: Sequence[Value], location: Location):
        super(Call, self).__init__(func.return_type, location)

        self.function = func
        self.arguments = arguments

    def __str__(self):
        arguments = ', '.join(str(arg) for arg in self.arguments)
        return f'{self.function.name}({arguments})'


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
    def __init__(self, name='<stdin>'):
        self.llvm_module = ir.Module(name)
        self.llvm_module.triple = binding.Target.from_default_triple().triple

        # names to symbol
        self.types = {}
        self.functions = MultiDict()

        # symbol to llvm
        self.llvm_types = LazyDict(builder=self.declare_type)
        self.llvm_functions = LazyDict(builder=self.declare_function)

    def __str__(self):
        return str(self.llvm_module)

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

    def declare_function(self, func: Function):
        llvm_return = self.llvm_types[func.return_type]
        llvm_params = [self.llvm_types[param.type] for param in func.parameters]
        llvm_type = ir.FunctionType(llvm_return, llvm_params)
        llvm_func = ir.Function(self.llvm_module, llvm_type, f'ORX_{func.name}')
        llvm_func.linkage = 'internal'

        for llvm_arg, param in zip(llvm_func.args, func.parameters):
            llvm_arg.name = param.name

        return llvm_func

    def emit(self, model: SemanticModel):
        for func in model.functions:
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
        self.llvm_parameters = {param: llvm_arg for param, llvm_arg in zip(func.parameters, llvm_func.args)}

        llvm_entry = llvm_func.append_basic_block('entry')
        self.llvm_builder = ir.IRBuilder(llvm_entry)

    @property
    def llvm_module(self) -> ir.Module:
        return self.parent.llvm_module

    @property
    def llvm_types(self) -> Mapping[Type, ir.Type]:
        return self.parent.llvm_types

    @property
    def llvm_functions(self) -> Mapping[Function, ir.Function]:
        return self.parent.llvm_functions

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

    @multimethod
    def emit_value(self, value: Value):
        raise Diagnostic(value.location, DiagnosticSeverity.Error, "Not implemented value conversion to LLVM")

    @multimethod
    def emit_value(self, value: Parameter):
        return self.llvm_parameters[value]

    @multimethod
    def emit_value(self, value: IntegerConstant):
        llvm_type = self.llvm_types[value.type]
        return ir.Constant(llvm_type, value.value)

    @multimethod
    def emit_value(self, value: BooleanConstant):
        llvm_type = self.llvm_types[value.type]
        return ir.Constant(llvm_type, value.value)

    @multimethod
    def emit_value(self, value: Call):
        llvm_args = [self.emit_value(arg) for arg in value.arguments]
        llvm_func = self.llvm_functions[value.function]
        return self.llvm_builder.call(llvm_func, llvm_args)


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
        model = context.open(filename)

        generator = ModuleCodegen(model.module.name)
        generator.emit(model)
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
    action = process_errors(action)
    sys.exit(action(**kwargs) or 0)


if __name__ == '__main__':
    main()
