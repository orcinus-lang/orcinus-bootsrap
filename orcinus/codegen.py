# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from typing import Mapping

from llvmlite import binding
from llvmlite import ir

from orcinus.language.semantic import *
from orcinus.utils import cached_property


class ModuleCodegen:
    def __init__(self, context: SemanticContext, name='<stdin>'):
        self.llvm_module = ir.Module(name)
        self.llvm_module.triple = binding.Target.from_default_triple().triple

        # names to symbol
        self.types = {}
        self.functions = MultiDict()

        # symbol to llvm
        self.llvm_types = LazyDict(constructor=self.declare_type, initializer=self.initialize_type)
        self.llvm_functions = LazyDict(constructor=self.declare_function)

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
    def declare_type(self, _: CharacterType):
        return ir.IntType(32)

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
    def int_pos(self: FunctionCodegen, _: Function, arguments: Sequence[Value], _1: Location):
        return self.emit_value(arguments[0])

    @staticmethod
    def int_neg(self: FunctionCodegen, _: Function, arguments: Sequence[Value], _1: Location):
        return self.llvm_builder.neg(self.emit_value(arguments[0]))

    @staticmethod
    def int_add(self: FunctionCodegen, _: Function, arguments: Sequence[Value], _1: Location):
        llvm_args = (self.emit_value(arg) for arg in arguments)
        return self.llvm_builder.add(*llvm_args)

    @staticmethod
    def int_sub(self: FunctionCodegen, _: Function, arguments: Sequence[Value], _1: Location):
        llvm_args = (self.emit_value(arg) for arg in arguments)
        return self.llvm_builder.sub(*llvm_args)

    @staticmethod
    def int_mul(self: FunctionCodegen, _: Function, arguments: Sequence[Value], _1: Location):
        llvm_args = (self.emit_value(arg) for arg in arguments)
        return self.llvm_builder.mul(*llvm_args)
