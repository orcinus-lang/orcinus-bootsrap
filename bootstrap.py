#!/usr/bin/env python3.7
from __future__ import annotations

import argparse
import collections
import enum
import functools
import io
import itertools
import logging
import re
import sys
from dataclasses import dataclass
from typing import Sequence, Iterator, Optional

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


@dataclass(order=True)
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


@dataclass
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


@dataclass
class NodeAST:
    location: Location


@dataclass
class ModuleAST(NodeAST):
    members: Sequence[MemberAST]


@dataclass
class TypeAST(NodeAST):
    pass


@dataclass
class NamedTypeAST(TypeAST):
    name: str


@dataclass
class MemberAST(TypeAST):
    pass


@dataclass
class ParameterAST(NodeAST):
    name: str
    type: TypeAST


@dataclass
class FunctionAST(MemberAST):
    name: str
    parameters: Sequence[ParameterAST]
    return_type: TypeAST
    statement: Optional[StatementAST]


@dataclass
class StatementAST(NodeAST):
    pass


@dataclass
class BlockStatementAST(StatementAST):
    statements: Sequence[StatementAST]


@dataclass
class PassStatementAST(StatementAST):
    pass


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
    for filename in filenames:
        with open(filename, 'r', encoding='utf8') as stream:
            parser = Parser(filename, stream)
            module = parser.parse()
        print(module)


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


def main():
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
