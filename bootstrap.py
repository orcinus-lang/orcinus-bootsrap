#!/usr/bin/env python
# Copyright (C) 2018 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import abc
import argparse
import collections
import enum
import functools
import io
import itertools
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Sequence, Iterator, Optional, cast

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
    As = enum.auto()
    Then = enum.auto()
    Ellipsis = enum.auto()


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
    def __add(lhs: int, rhs: int, min: int) -> int:
        """Compute max(min, lhs+rhs) (provided min <= lhs)."""
        return rhs + lhs if 0 < rhs or -rhs < lhs else min

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
        'as': TokenID.As
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
            if token not in self.TRIVIA_TOKENS:
                yield token

            if token.id in self.OPEN_BRACKETS:
                level += 1
            elif token.id in self.CLOSE_BRACKETS:
                level -= 1

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
        while self.match(TokenID.Def):
            members.append(self.parse_member())
        return tuple(members)

    def parse_member(self) -> MemberAST:
        """
        member:
            function
        """
        if self.match(TokenID.Def):
            return self.parse_function()

        raise NotImplementedError

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
        type = self.parse_type()

        # noinspection PyArgumentList
        return ParameterAST(name=tok_name.value, type=type, location=tok_name.location)

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
        while self.match(TokenID.Pass):
            statements.append(self.parse_statement())
        self.consume(TokenID.Undent)
        location = statements[0].location + statements[-1].location

        # noinspection PyArgumentList
        return BlockStatementAST(statements=tuple(statements), location=location)

    def parse_statement(self) -> StatementAST:
        """
        statement:
            'pass'
        """
        tok_pass = self.consume(TokenID.Pass)
        self.consume(TokenID.NewLine)

        # noinspection PyArgumentList
        return PassStatementAST(location=tok_pass.location)


@dataclass(unsafe_hash=True, frozen=True)
class NodeAST:
    location: Location


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

    def get_module_name(self, filename):
        fullname = os.path.abspath(filename)
        for path in self.paths:
            if fullname.startswith(path):
                return self.convert_module_name(fullname, path)

        raise BootstrapError(f"Not found file `{filename}` in library paths")

    def open(self, filename):
        """ Open module from file """
        module_name = self.get_module_name(filename)

        with open(filename, 'r', encoding='utf8') as stream:
            return self.__open_source(filename, module_name, stream)

    def load(self, module_name):
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
        self.types = []
        self.functions = []

    @property
    def module(self) -> Module:
        return self.symbols[self.tree]

    def analyze(self):
        self.declare_symbol(self.tree)

    def declare_symbol(self, node: NodeAST, parent: ContainerSymbol = None):
        symbol = self.annotate_symbol(node, parent)
        self.symbols[node] = symbol

        if isinstance(symbol, Type):
            self.types.append(symbol)
        elif isinstance(symbol, Function):
            self.functions.append(symbol)

        if hasattr(node, 'members'):
            for child in node.members:
                child_symbol = self.declare_symbol(child, symbol)
                symbol.add_member(child_symbol)

        return symbol

    @multimethod
    def resolve_type(self, node: TypeAST) -> Type:
        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented type resolving")

    @multimethod
    def resolve_type(self, node: NamedTypeAST) -> Type:
        if node.name == 'void':
            return VoidType(self.module, node.location)
        elif node.name == 'bool':
            return BooleanType(self.module, node.location)
        elif node.name == 'int':
            return IntegerType(self.module, node.location)

        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented type resolving")

    @multimethod
    def annotate_symbol(self, node: NodeAST, parent: ContainerSymbol) -> Symbol:
        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented member declaration")

    @multimethod
    def annotate_symbol(self, node: ModuleAST, parent=None) -> Module:
        return Module(self.module_name, Location(node.location.filename))

    @multimethod
    def annotate_symbol(self, node: FunctionAST, parent: ContainerSymbol) -> Function:
        parameters = [self.resolve_type(param.type) for param in node.parameters]
        return_type = self.resolve_type(node.return_type)
        func_type = FunctionType(self.module, parameters, return_type, node.location)
        func = Function(parent, node.name, func_type, node.location)

        for node_param, func_param in zip(node.parameters, func.parameters):
            func_param.name = node_param.name
            func_param.location = node_param.location

            self.symbols[node_param] = func_param

        return func


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

    @property
    def members(self) -> Sequence[OwnedSymbol]:
        return self.__members

    def add_member(self, symbol: OwnedSymbol):
        self.__members.append(symbol)


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
        self.__location = locals()


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

    def __str__(self):
        parameters = ', '.join(str(param) for param in self.parameters)
        return f'{self.name}({parameters}) -> {self.return_type}'


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

        for llvm_arg, param in zip(llvm_func.args, func.parameters):
            llvm_arg.name = param.name

        return llvm_func

    def emit(self, model: SemanticModel):
        for func in model.functions:
            self.emit_function(func)

    def emit_function(self, func: Function):
        llvm_func = self.llvm_functions[func]
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


def show_source_lines(location: Location, before=2, after=2, columns=None):
    """
    Convert selected lines to error message, e.g.:

    ```
        1 : from module import system =
          : --------------------------^
    ```
    """
    stream = io.StringIO()
    columns = columns or 80

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


def process_pdb(parser: argparse.ArgumentParser, action):
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
        action = process_pdb(parser, action)
    action = process_errors(action)
    sys.exit(action(**kwargs) or 0)


if __name__ == '__main__':
    main()
