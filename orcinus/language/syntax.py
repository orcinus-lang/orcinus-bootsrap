# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import abc
import collections
import enum
import itertools
import weakref
from dataclasses import dataclass
from typing import Sequence, Optional, Iterator, cast

from orcinus.core.locations import Location
from orcinus.core.locations import Position
from orcinus.utils import cached_property


class SyntaxSymbol(abc.ABC):
    @property
    @abc.abstractmethod
    def location(self) -> Location:
        raise NotImplementedError

    @property
    def begin_location(self) -> Location:
        """ Begin location in source, include all leading trivia and tokens """
        return self.location

    @property
    def end_location(self) -> Location:
        """ End location in source, include all trailing trivia and tokens """
        return self.location

    @abc.abstractmethod
    def contains(self, position: Position) -> bool:
        raise NotImplemented

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self):
        class_name = type(self).__name__
        return f'<{class_name}: {self}>'


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


@enum.unique
class TriviaID(enum.IntEnum):
    NewLine = enum.auto()
    Whitespace = enum.auto()
    Comment = enum.auto()


class SyntaxTrivia(SyntaxSymbol):
    def __init__(self, trivia_id: TriviaID, value: str, location: Location):
        self.__id = trivia_id
        self.__value = value
        self.__location = location
        self.__parent = None

    @property
    def id(self) -> TriviaID:
        return self.__id

    @property
    def value(self) -> str:
        return self.__value

    @property
    def location(self) -> Location:
        return self.__location

    @property
    def parent(self) -> Optional[SyntaxToken]:
        return self.__parent() if self.__parent else None

    @parent.setter
    def parent(self, value: SyntaxNode):
        self.__parent = weakref.ref(value) if value else None

    def contains(self, position: Position) -> bool:
        return self.begin_location.begin <= position <= self.end_location.end

    def __str__(self) -> str:
        value = self.value.strip()
        if value:
            return f'[{self.location}] {self.id.name}: `{value}`'
        return f'[{self.location}] {self.id.name}'


class SyntaxToken(SyntaxSymbol):
    def __init__(self, token_id: TokenID, value: str, location: Location, *,
                 leading_trivia: Sequence[SyntaxTrivia] = None, trailing_trivia: Sequence[SyntaxTrivia] = None):
        self.__id = token_id
        self.__value = value
        self.__location = location
        self.__leading_trivia = tuple(leading_trivia or [])
        self.__trailing_trivia = tuple(trailing_trivia or [])
        self.__parent = None

    @property
    def id(self) -> TokenID:
        return self.__id

    @property
    def value(self) -> str:
        return self.__value

    @property
    def parent(self) -> Optional[SyntaxNode]:
        return self.__parent() if self.__parent else None

    @parent.setter
    def parent(self, value: SyntaxNode):
        self.__parent = weakref.ref(value) if value else None

    @property
    def leading_trivia(self) -> Sequence[SyntaxTrivia]:
        return self.__leading_trivia

    @property
    def trailing_trivia(self) -> Sequence[SyntaxTrivia]:
        return self.__trailing_trivia

    @property
    def location(self) -> Location:
        return self.__location

    @property
    def begin_location(self) -> Location:
        """ Begin location in source, include leading trivia """
        if self.leading_trivia:
            return self.leading_trivia[0].location
        return self.location

    @property
    def end_location(self) -> Location:
        """ End location in source, include trailing trivia """
        if self.trailing_trivia:
            return self.trailing_trivia[-1].location
        return self.location

    def contains(self, position: Position) -> bool:
        return self.location.begin <= position <= self.location.end

    def __str__(self) -> str:
        value = self.value.strip()
        if value:
            return f'[{self.location}] {self.id.name}: `{value}`'
        return f'[{self.location}] {self.id.name}'

    def _propagate_parents(self):
        for child in itertools.chain(self.leading_trivia, self.trailing_trivia):
            child.parent = self


class SyntaxNode(SyntaxSymbol):
    __parent = None

    @property
    def parent(self) -> Optional[SyntaxNode]:
        return self.__parent() if self.__parent else None

    @parent.setter
    def parent(self, value: SyntaxNode):
        self.__parent = weakref.ref(value) if value else None

    @property
    @abc.abstractmethod
    def children(self) -> Sequence[SyntaxSymbol]:
        raise NotImplementedError

    @property
    def nodes(self) -> Sequence[SyntaxNode]:
        """ Returns children syntax nodes """
        return cast(Sequence[SyntaxNode], list(filter(lambda s: isinstance(s, SyntaxNode), self.children)))

    @property
    def tokens(self) -> Sequence[SyntaxToken]:
        """ Returns children syntax tokens """
        return cast(Sequence[SyntaxToken], list(filter(lambda s: isinstance(s, SyntaxToken), self.children)))

    @property
    def begin_location(self) -> Location:
        """ Begin location in source, include leading tokens and trivia """
        if self.children:
            return self.children[0].begin_location
        return self.location

    @property
    def end_location(self) -> Location:
        """ End location in source, include leading tokens and trivia """
        if self.children:
            return self.children[-1].end_location
        return self.location

    def contains(self, position: Position) -> bool:
        return self.begin_location.begin <= position <= self.end_location.end

    def find_position(self, position: Position) -> Optional[SyntaxNode]:
        # Find children
        for child in self.children:
            if isinstance(child, SyntaxNode):
                found = child.find_position(position)
                if found:
                    return found

        # Find location
        if self.contains(position):
            return self
        return None

    def propagate_parents(self):
        for child in self.children:
            child.parent = self
            child.propagate_parents()

    def __iter__(self) -> Iterator[SyntaxSymbol]:
        return iter(self.nodes)

    def __str__(self) -> str:
        return type(self).__name__

    def _cleanup(self, *symbols: SyntaxSymbol) -> Sequence[SyntaxSymbol]:
        return [n for n in symbols if isinstance(n, SyntaxSymbol)]


class SyntaxCollection(SyntaxNode, collections.abc.Sequence):
    def __init__(self, children: Sequence[SyntaxSymbol] = None, location: Location = None):
        if not children and not location:
            raise ValueError(u'Require children or location')

        self.__children = tuple(children or ())
        self.__location = location

    @property
    def location(self) -> Location:
        if self.__location:
            return self.__location
        return self.__children[0].begin_location + self.__children[-1].end_location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return self.__children

    def __getitem__(self, i: int):
        return self.nodes[i]

    def __len__(self) -> int:
        return len(self.nodes)

    def __contains__(self, symbol: SyntaxSymbol) -> bool:
        return symbol in self.__children


@dataclass(unsafe_hash=True, frozen=True)
class ModuleAST(SyntaxNode):
    imports: Sequence[ImportAST]
    members: Sequence[MemberAST]
    tok_eof: SyntaxToken

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.members, self.tok_eof]

    @property
    def location(self) -> Location:
        begin = cast(SyntaxCollection, self.members).begin_location
        return begin + self.tok_eof.end_location


@dataclass(unsafe_hash=True, frozen=True)
class QualifiedNameAST(SyntaxNode):
    names: Sequence[SyntaxToken]

    @cached_property
    def full_name(self):
        return ''.join(token.value for token in self.tokens)

    @property
    def location(self) -> Location:
        return cast(SyntaxCollection, self.names).location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return cast(SyntaxCollection, self.names).children


@dataclass(unsafe_hash=True, frozen=True)
class AliasAST(SyntaxNode):
    qualified_name: QualifiedNameAST
    tok_as: Optional[SyntaxToken]
    tok_alias: SyntaxToken

    @property
    def name(self) -> str:
        return self.qualified_name.full_name

    @property
    def alias(self) -> Optional[str]:
        return self.tok_alias.value if self.tok_alias else None

    @property
    def location(self) -> Location:
        return self.qualified_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return self._cleanup(self.qualified_name, self.tok_as, self.tok_alias)


@dataclass(unsafe_hash=True, frozen=True)
class ImportAST(SyntaxNode):
    tok_import: SyntaxToken
    aliases: Sequence[AliasAST]
    tok_newline: SyntaxToken

    @property
    def location(self) -> Location:
        return self.tok_import.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.tok_import, self.aliases, self.tok_newline]


@dataclass(unsafe_hash=True, frozen=True)
class ImportFromAST(ImportAST):
    tok_from: SyntaxToken
    qualified_name: QualifiedNameAST

    @property
    def module(self):
        return self.qualified_name.full_name

    @property
    def location(self) -> Location:
        return self.tok_from.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.tok_from, self.qualified_name, self.tok_import, self.aliases]


@dataclass(unsafe_hash=True, frozen=True)
class TypeAST(SyntaxNode):
    pass


@dataclass(unsafe_hash=True, frozen=True)
class ParameterizedTypeAST(TypeAST):
    type: TypeAST
    arguments: Sequence[TypeAST]


@dataclass(unsafe_hash=True, frozen=True)
class GenericParameterAST(SyntaxNode):
    tok_name: SyntaxToken

    @property
    def name(self) -> str:
        return self.tok_name.value

    @property
    def location(self) -> Location:
        return self.tok_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.tok_name]


@dataclass(unsafe_hash=True, frozen=True)
class NamedTypeAST(TypeAST):
    tok_name: SyntaxToken

    @property
    def location(self) -> Location:
        return self.tok_name.location

    @property
    def name(self) -> str:
        return self.tok_name.value

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.tok_name]


class AutoTypeAST(TypeAST):
    __location: Location

    def __init__(self, location: Location):
        super(AutoTypeAST, self).__init__()

        self.__location = location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return []

    @property
    def location(self) -> Location:
        return self.__location


@dataclass(unsafe_hash=True, frozen=True)
class MemberAST(SyntaxNode):
    pass


@dataclass(unsafe_hash=True, frozen=True)
class PassMemberAST(MemberAST):
    tok_pass: SyntaxToken
    tok_newline: SyntaxToken

    @property
    def location(self) -> Location:
        return self.tok_pass.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.tok_pass, self.tok_newline]


@dataclass(unsafe_hash=True, frozen=True)
class TypeDeclarationAST(MemberAST):
    tok_name: SyntaxToken
    members: Sequence[MemberAST]

    @property
    def name(self) -> str:
        return self.tok_name.value

    @property
    def location(self) -> Location:
        return self.tok_name.location


@dataclass(unsafe_hash=True, frozen=True)
class StructAST(TypeDeclarationAST):
    tok_struct: SyntaxToken
    generic_parameters: Sequence[GenericParameterAST]

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return self._cleanup(self.tok_struct, self.tok_name, self.generic_parameters, self.members)


@dataclass(unsafe_hash=True, frozen=True)
class ClassAST(TypeDeclarationAST):
    tok_class: SyntaxToken
    generic_parameters: Sequence[GenericParameterAST]

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return self._cleanup(self.tok_class, self.tok_name, self.generic_parameters, self.members)


@dataclass(unsafe_hash=True, frozen=True)
class FieldAST(MemberAST):
    tok_name: SyntaxToken
    tok_colon: SyntaxToken
    type: TypeAST
    tok_newline: SyntaxToken

    @property
    def name(self) -> str:
        return self.tok_name.value

    @property
    def location(self) -> Location:
        return self.tok_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.tok_name, self.tok_colon, self.type, self.tok_newline]


@dataclass(unsafe_hash=True, frozen=True)
class ParameterAST(SyntaxNode):
    tok_name: SyntaxToken
    tok_colon: SyntaxToken
    type: TypeAST

    @property
    def name(self) -> str:
        return self.tok_name.value

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return self._cleanup(self.tok_name, self.tok_colon, self.type)

    @property
    def location(self) -> Location:
        return self.tok_name.location


@dataclass(unsafe_hash=True, frozen=True)
class FunctionAST(MemberAST):
    tok_def: SyntaxToken
    tok_name: SyntaxToken
    generic_parameters: Sequence[GenericParameterAST]
    tok_open: SyntaxToken
    parameters: Sequence[ParameterAST]
    tok_close: SyntaxToken
    tok_then: SyntaxToken
    return_type: TypeAST
    tok_colon: SyntaxToken
    statement: Optional[StatementAST]

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return self._cleanup(
            self.tok_def,
            self.tok_name,
            self.generic_parameters,
            self.tok_open,
            self.parameters,
            self.tok_close,
            self.tok_then,
            self.return_type,
            self.tok_colon,
            self.statement
        )

    @property
    def name(self) -> str:
        return self.tok_name.value

    @property
    def location(self) -> Location:
        return self.tok_name.location


@dataclass(unsafe_hash=True, frozen=True)
class StatementAST(SyntaxNode):
    pass


@dataclass(unsafe_hash=True, frozen=True)
class BlockStatementAST(StatementAST):
    statements: Sequence[StatementAST]

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return self.statements

    @property
    def location(self) -> Location:
        return cast(SyntaxCollection, self.statements).location


@dataclass(unsafe_hash=True, frozen=True)
class EllipsisStatementAST(StatementAST):
    tok_ellipsis: SyntaxToken
    tok_newline: SyntaxToken

    @property
    def location(self) -> Location:
        return self.tok_ellipsis.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.tok_ellipsis, self.tok_newline]


@dataclass(unsafe_hash=True, frozen=True)
class ElseStatementAST(StatementAST):
    tok_else: SyntaxToken
    tok_colon: SyntaxToken
    tok_newline: SyntaxToken
    statement: StatementAST

    @property
    def location(self) -> Location:
        return self.tok_else.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.tok_else, self.tok_colon, self.tok_newline, self.statement]


@dataclass(unsafe_hash=True, frozen=True)
class PassStatementAST(StatementAST):
    tok_pass: SyntaxToken
    tok_newline: SyntaxToken

    @property
    def location(self) -> Location:
        return self.tok_pass.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.tok_pass, self.tok_newline]


@dataclass(unsafe_hash=True, frozen=True)
class ReturnStatementAST(StatementAST):
    tok_return: SyntaxToken
    value: Optional[ExpressionAST] = None

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return self._cleanup(self.tok_return, self.value)

    @property
    def location(self) -> Location:
        return self.tok_return.location


@dataclass(unsafe_hash=True, frozen=True)
class ConditionStatementAST(StatementAST):
    tok_if: SyntaxToken
    condition: ExpressionAST
    tok_colon: SyntaxToken
    tok_newline: SyntaxToken
    then_statement: StatementAST
    else_statement: Optional[StatementAST]

    @property
    def location(self) -> Location:
        return self.tok_if.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return self._cleanup(self.tok_if,
                             self.condition,
                             self.tok_colon,
                             self.tok_newline,
                             self.then_statement,
                             self.else_statement)


@dataclass(unsafe_hash=True, frozen=True)
class WhileStatementAST(StatementAST):
    tok_while: SyntaxToken
    condition: ExpressionAST
    tok_colon: SyntaxToken
    tok_newline: SyntaxToken
    then_statement: StatementAST
    else_statement: Optional[StatementAST]

    @property
    def location(self) -> Location:
        return self.tok_while.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return self._cleanup(self.tok_while,
                             self.condition,
                             self.tok_colon,
                             self.tok_newline,
                             self.then_statement,
                             self.else_statement)


@dataclass(unsafe_hash=True, frozen=True)
class ExpressionStatementAST(StatementAST):
    value: ExpressionAST
    tok_newline: SyntaxToken

    @property
    def location(self) -> Location:
        return self.value.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.value, self.tok_newline]


@dataclass(unsafe_hash=True, frozen=True)
class AssignStatementAST(StatementAST):
    target: ExpressionAST
    tok_equals: SyntaxToken
    source: ExpressionAST

    @property
    def location(self) -> Location:
        return self.target.location + self.source.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.target, self.tok_equals, self.source]


@dataclass(unsafe_hash=True, frozen=True)
class ExpressionAST(SyntaxNode):
    pass


@dataclass(unsafe_hash=True, frozen=True)
class IntegerExpressionAST(ExpressionAST):
    tok_number: SyntaxToken

    @property
    def value(self) -> int:
        return int(self.tok_number.value)

    @property
    def location(self) -> Location:
        return self.tok_number.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.tok_number]


@dataclass(unsafe_hash=True, frozen=True)
class NamedExpressionAST(ExpressionAST):
    tok_name: SyntaxToken

    @property
    def name(self) -> str:
        return self.tok_name.value

    @property
    def location(self) -> Location:
        return self.tok_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.tok_name]


@enum.unique
class UnaryID(enum.IntEnum):
    Not = enum.auto()
    Pos = enum.auto()
    Neg = enum.auto()
    Inv = enum.auto()


@dataclass(unsafe_hash=True, frozen=True)
class UnaryExpressionAST(ExpressionAST):
    operator: UnaryID
    tok_operator: SyntaxToken
    operand: ExpressionAST

    @property
    def location(self) -> Location:
        return self.tok_operator.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.tok_operator, self.operand]


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
    tok_operator: SyntaxToken
    left_operand: ExpressionAST
    right_operand: ExpressionAST

    @property
    def location(self) -> Location:
        return self.tok_operator.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.left_operand, self.tok_operator, self.right_operand]


@dataclass(unsafe_hash=True, frozen=True)
class CallExpressionAST(ExpressionAST):
    value: ExpressionAST
    tok_open: SyntaxToken
    arguments: Sequence[ExpressionAST]
    tok_close: SyntaxToken

    @property
    def location(self) -> Location:
        return self.value.location + self.tok_close.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.value, self.tok_open, self.arguments, self.tok_close]


@dataclass(unsafe_hash=True, frozen=True)
class SubscribeExpressionAST(ExpressionAST):
    value: ExpressionAST
    tok_open: SyntaxToken
    arguments: Sequence[ExpressionAST]
    tok_close: SyntaxToken

    @property
    def location(self) -> Location:
        return self.value.location + self.tok_close.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.value, self.tok_open, self.arguments, self.tok_close]


@dataclass(unsafe_hash=True, frozen=True)
class AttributeAST(ExpressionAST):
    value: ExpressionAST
    tok_dot: SyntaxToken
    tok_name: SyntaxToken

    @property
    def name(self) -> str:
        return self.tok_name.value

    @property
    def location(self) -> Location:
        return self.value.location + self.tok_name.location

    @property
    def children(self) -> Sequence[SyntaxSymbol]:
        return [self.value, self.tok_dot, self.tok_name]
