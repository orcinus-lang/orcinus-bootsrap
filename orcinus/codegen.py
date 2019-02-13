# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from io import StringIO
from typing import TextIO

from orcinus.language.semantic import *


class ModuleCodegen:
    def __init__(self, context: SemanticContext):
        # builtins functions
        self.context = context

    def __str__(self):
        return str(self.llvm_module)

    def emit_header(self, module: Module, stream: TextIO):
        writer = HeaderWriter(Stream(stream))
        writer.write_module(module)

    def emit_source(self, module: Module, stream: TextIO):
        writer = SourceWriter(Stream(stream))
        writer.write_module(module)


class Stream:
    def __init__(self, stream: TextIO):
        self.stream = stream

    def write(self, value: str = None):
        if value:
            self.stream.write(value)

    def write_line(self, value: str = None):
        self.write(value)
        self.stream.write('\n')


class Writer:
    def __init__(self, stream: Stream):
        self.stream = stream

    def write_pragma(self, pragma):
        self.stream.write_line(f'#pragma {pragma}')

    def write_include(self, filename: str, is_system=False):
        if is_system:
            self.stream.write_line(f'#include <{filename}>')
        else:
            self.stream.write_line(f'#include "{filename}"')

    def get_module_name(self, module: Module) -> str:
        return module.name.replace('.', '::')

    def get_type_name(self, vtype: Type) -> str:
        if isinstance(vtype, StringType):
            return 'std::string'

        elif isinstance(vtype, IntegerType):
            return 'std::int64_t'

        elif isinstance(vtype, FunctionType):
            return_type = self.get_type_name(vtype.return_type)
            parameters = ', '.join(self.get_type_name(param) for param in vtype.parameters)
            return f'std::function<{return_type} ({parameters})>'

        elif isinstance(vtype, (ClassType, InterfaceType, StructType, EnumType)):
            return f'::{self.get_module_name(vtype.module)}::{vtype.name}'

        raise NotImplementedError

    def get_function_declaration(self, symbol: Function, is_self=False):
        parameters = symbol.parameters if not is_self else symbol.parameters[1:]
        parameters = ', '.join(f'{self.get_type_name(param.type)} {param.name}' for param in parameters)
        return parameters

    def get_value(self, value: Value, func: Function = None) -> str:
        stream = StringIO()
        ValueSourceWriter(func, Stream(stream)).write_value(value)
        return stream.getvalue()


class MemberWriter(Writer):
    def write_parents(self, parents: Sequence[Type]):
        self.stream.write_line(' : ')
        for idx, parent in enumerate(parents):
            if idx:
                self.stream.write(', ')
            self.stream.write(f'public {self.get_type_name(parent)}')

    def write_member(self, symbol: Symbol):
        if isinstance(symbol, Type):
            return self.write_type(symbol)
        elif isinstance(symbol, Function):
            return self.write_function(symbol)
        elif isinstance(symbol, Field):
            return self.write_field(symbol)

        raise NotImplementedError

    def write_type(self, symbol: Type):
        if isinstance(symbol, (IntegerType, StringType, FunctionType)):
            return
        elif isinstance(symbol, StructType):
            return self.write_struct(symbol)
        elif isinstance(symbol, ClassType):
            return self.write_class(symbol)
        elif isinstance(symbol, InterfaceType):
            return self.write_interface(symbol)
        elif isinstance(symbol, EnumType):
            return self.write_enum(symbol)

        raise NotImplementedError

    def write_function(self, symbol: Function):
        raise NotImplementedError

    def write_class(self, member: ClassType):
        raise NotImplementedError

    def write_struct(self, member: StructType):
        raise NotImplementedError

    def write_interface(self, member: InterfaceType):
        raise NotImplementedError

    def write_enum(self, member: EnumType):
        raise NotImplementedError

    def write_field(self, symbol: Field):
        raise NotImplementedError


class MemberHeaderWriter(MemberWriter):
    def write_function(self, symbol: Function):
        raise NotImplementedError

    def write_class(self, member: ClassType):
        writer = ClassMemberHeaderWriter(self.stream)
        self.stream.write(f'class {member.name}')
        self.write_parents(member.parents)
        self.stream.write_line('{')
        self.stream.write_line('public:')
        self.stream.write_line(f'{member.name}(const {member.name}&) =delete;')
        self.stream.write_line(f'{member.name}& operator=(const {member.name}&) =delete;')

        for member in member.members:
            writer.write_member(member)
        self.stream.write_line('};')

    def write_struct(self, member: StructType):
        writer = StructMemberHeaderWriter(self.stream)
        self.stream.write(f'class {member.name}')
        self.write_parents(member.parents)
        self.stream.write_line('{')
        self.stream.write_line('public:')

        for member in member.members:
            writer.write_member(member)
        self.stream.write_line('};')

    def write_interface(self, member: InterfaceType):
        writer = InterfaceMemberHeaderWriter(self.stream)
        self.stream.write(f'class {member.name}')
        self.write_parents(member.parents)
        self.stream.write_line('{')
        self.stream.write_line('public:')

        self.stream.write_line(f'{member.name}(const {member.name}&) =delete;')
        self.stream.write_line(f'{member.name}& operator=(const {member.name}&) =delete;')

        for member in member.members:
            writer.write_member(member)
        self.stream.write_line('};')

    def write_enum(self, member: EnumType):
        writer = TypeMemberHeaderWriter(self.stream)
        self.stream.write_line(f'enum class {member.name} {{')
        for idx, member in enumerate(member.values):
            if idx:
                self.stream.write(', ')
            if member.value:
                self.stream.write(f'{member.name} = {self.get_value(member.value)}')
            else:
                self.stream.write(member.name)
        self.stream.write_line('};')

    def write_field(self, symbol: Field):
        raise NotImplementedError


# noinspection PyAbstractClass
class ModuleMemberHeaderWriter(MemberHeaderWriter):
    pass


# noinspection PyAbstractClass
class TypeMemberHeaderWriter(MemberHeaderWriter):
    def write_function(self, symbol: Function):
        if symbol.name == '__init__':
            return self.write_constructor(symbol)

        return_type = self.get_type_name(symbol.return_type)
        self.stream.write_line(
            f"{return_type} {symbol.name}({self.get_function_declaration(symbol, True)});"
        )

    def write_constructor(self, symbol: Function):
        self.stream.write_line(f"{symbol.owner.name}({self.get_function_declaration(symbol, True)});")


# noinspection PyAbstractClass
class InterfaceMemberHeaderWriter(TypeMemberHeaderWriter):
    def write_field(self, symbol: Field):
        self.stream.write_line(f'virtual {self.get_type_name(symbol.type)} get_{symbol.name}() const =0;')
        self.stream.write_line(f'virtual void set_{symbol.name}({self.get_type_name(symbol.type)} value)  =0;')


# noinspection PyAbstractClass
class FieldMemberHeaderWriter(TypeMemberHeaderWriter):
    def write_field(self, symbol: Field):
        self.stream.write_line(f'{self.get_type_name(symbol.type)} __{symbol.name};')

        self.stream.write_line(f'{self.get_type_name(symbol.type)} get_{symbol.name}() const {{')
        self.stream.write_line(f'    return this->__{symbol.name};')
        self.stream.write_line('}')

        self.stream.write_line(f'void set_{symbol.name}({self.get_type_name(symbol.type)} value) {{')
        self.stream.write_line(f'    this->__{symbol.name} = value;')
        self.stream.write_line('}')


# noinspection PyAbstractClass
class StructMemberHeaderWriter(FieldMemberHeaderWriter):
    pass


# noinspection PyAbstractClass
class ClassMemberHeaderWriter(FieldMemberHeaderWriter):
    pass


# noinspection PyAbstractClass
class MemberSourceWriter(MemberWriter):
    def write_class(self, member: ClassType):
        writer = ClassMemberSourceWriter(self.stream)
        for child in member.members:
            writer.write_member(child)

    def write_struct(self, member: StructType):
        writer = ClassMemberSourceWriter(self.stream)
        for child in member.members:
            writer.write_member(child)

    def write_enum(self, member: EnumType):
        pass

    def write_interface(self, member: InterfaceType):
        writer = ClassMemberSourceWriter(self.stream)
        for child in member.members:
            writer.write_member(child)

    def write_field(self, symbol: Field):
        raise NotImplementedError


# noinspection PyAbstractClass
class ModuleMemberSourceWriter(MemberSourceWriter):
    pass


# noinspection PyAbstractClass
class TypeMemberSourceWriter(MemberSourceWriter):
    def write_function(self, symbol: Function):
        if not symbol.statement:
            return

        if symbol.name == '__init__':
            self.write_constructor(symbol)
        else:
            type_name = symbol.owner.name
            module_name = self.get_module_name(symbol.module)
            return_type = self.get_type_name(symbol.return_type)
            declaration = self.get_function_declaration(symbol, True)
            self.stream.write_line(
                f"{return_type} ::{module_name}::{type_name}::{symbol.name}({declaration}) {{"
            )

        write = StatementWriter(symbol, self.stream)
        write.write_statement(symbol.statement)
        self.stream.write_line('}')

    def write_constructor(self, symbol: Function):
        type_name = symbol.owner.name
        module_name = self.get_module_name(symbol.module)
        self.stream.write_line(
            f"::{module_name}::{type_name}::{type_name}({self.get_function_declaration(symbol, True)}) {{"
        )


# noinspection PyAbstractClass
class FieldMemberSourceWriter(TypeMemberSourceWriter):
    pass


# noinspection PyAbstractClass
class ClassMemberSourceWriter(FieldMemberSourceWriter):
    def write_field(self, symbol: Field):
        pass  # skip


# noinspection PyAbstractClass
class StructMemberSourceWriter(FieldMemberSourceWriter):
    pass


class StatementWriter(Writer):
    def __init__(self, function: Function, stream: Stream):
        super(StatementWriter, self).__init__(stream)
        self.function = function

    def get_value(self, value: Value, func: Function = None) -> str:
        stream = StringIO()
        ValueSourceWriter(func or self.function, Stream(stream)).write_value(value)
        return stream.getvalue()

    def write_statement(self, stmt: Statement):
        if isinstance(stmt, BlockStatement):
            return self.write_block_statement(stmt)
        elif isinstance(stmt, AssignStatement):
            return self.write_assign_statement(stmt)

        class_name = type(stmt).__name__
        raise NotImplementedError(f'Not implemented: {class_name}')

    def write_block_statement(self, stmt: BlockStatement):
        for child in stmt.statements:
            self.write_statement(child)

    def write_assign_statement(self, stmt: AssignStatement):
        write = ValueTargetWriter(self.function, self.stream)
        write.write_value(stmt.target, stmt.source)
        self.stream.write(';')


class ValueWriter(Writer):
    def __init__(self, function: Function, stream: Stream):
        super(ValueWriter, self).__init__(stream)
        self.function = function

    def get_value(self, value: Value, func: Function = None) -> str:
        stream = StringIO()
        ValueSourceWriter(func or self.function, Stream(stream)).write_value(value)
        return stream.getvalue()

    def is_self_argument(self, value: Parameter):
        if not isinstance(value, Parameter):
            return False
        elif not self.function.parameters:
            return False
        elif not isinstance(self.function.owner, Type):
            return False
        elif value.type != self.function.owner:
            return False

        return self.function.parameters[0] is value

    def write_bounded_value(self, instance: Value):
        self.stream.write(self.get_value(instance))

        is_self = self.is_self_argument(instance)
        is_class = isinstance(instance.type, ClassType)
        if is_self or is_class:
            self.stream.write('->')
        else:
            self.stream.write('.')

    def write_arguments(self, sequence: Sequence[Value]):
        for idx, value in enumerate(sequence):
            if idx:
                self.stream.write(', ')
            self.stream.write(self.get_value(value))


class ValueTargetWriter(ValueWriter):
    def write_value(self, target: Value, source: Value):
        if isinstance(target, BoundedField):
            return self.write_bounded_field(target, source)

        class_name = type(target).__name__
        raise NotImplementedError(f'Not implemented: {class_name}')

    def write_bounded_field(self, value: BoundedField, source: Value):
        self.write_bounded_value(value.instance)
        self.stream.write(f'set_{value.field.name}({self.get_value(source)})')


class ValueSourceWriter(ValueWriter):
    def write_value(self, value: Value):
        if isinstance(value, BoundedField):
            return self.write_bounded_field(value)
        elif isinstance(value, StringConstant):
            return self.write_string(value)
        elif isinstance(value, IntegerConstant):
            return self.write_integer(value)
        elif isinstance(value, Parameter):
            return self.write_parameter(value)
        elif isinstance(value, NewInstruction):
            return self.write_new_inst(value)
        elif isinstance(value, CallInstruction):
            return self.write_call_inst(value)

        class_name = type(value).__name__
        raise NotImplementedError(f'Not implemented: {class_name}')

    def write_bounded_field(self, value: BoundedField):
        self.write_bounded_value(value.instance)
        self.stream.write(f'get_{value.field.name}()')

    def write_integer(self, value: IntegerConstant):
        self.stream.write(str(value.value))

    def write_string(self, value: StringConstant):
        self.stream.write('"')
        self.stream.write(str(value.value).replace('"', '\\"'))
        self.stream.write('"')

    def write_parameter(self, value: Parameter):
        if self.is_self_argument(value):
            self.stream.write('this')
        else:
            self.stream.write(value.name)

    def write_new_inst(self, value: NewInstruction):
        if isinstance(value.type, ClassType):
            self.stream.write(f'new {self.get_type_name(value.type)}')
        elif isinstance(value.type, StructType):
            self.stream.write(f'{self.get_type_name(value.type)}')
        self.stream.write('(')
        self.write_arguments(value.arguments)
        self.stream.write(')')

    def write_call_inst(self, value: CallInstruction):
        if value.function.name == '__neg__':
            self.stream.write('-')
        else:
            self.stream.write(value.function.name)
        self.stream.write('(')
        self.write_arguments(value.arguments)
        self.stream.write(')')


class HeaderWriter(Writer):
    def write_module(self, module: Module):
        self.write_pragma('once')
        self.stream.write_line()

        self.write_include('cstdint', is_system=True)
        self.write_include('string', is_system=True)
        self.write_include('vector', is_system=True)

        for dependency in module.dependencies:
            self.write_include(dependency.name + '.hpp')

        self.stream.write_line()
        self.stream.write_line(f"namespace {self.get_module_name(module)} {{")

        writer = ModuleMemberHeaderWriter(self.stream)
        for member in module.members:
            writer.write_member(member)

        self.stream.write_line('}')


class SourceWriter(Writer):
    def write_module(self, module: Module):
        self.write_include(module.name + '.hpp')
        self.stream.write_line()

        writer = ModuleMemberSourceWriter(self.stream)
        for member in module.members:
            writer.write_member(member)
