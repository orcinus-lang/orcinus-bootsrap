# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import dataclasses
import enum
from dataclasses import dataclass
from typing import Sequence, Optional


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
