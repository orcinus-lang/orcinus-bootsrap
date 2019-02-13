# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

import heapq
import itertools
import logging
from contextlib import contextmanager
from typing import Tuple, Iterable

from multidict import MultiDict
from multimethod import multimethod

from orcinus.collections import LazyDict, LazyEmptyDict
from orcinus.core.diagnostics import DiagnosticSeverity, Diagnostic, DiagnosticManager
from orcinus.exceptions import OrcinusError
from orcinus.language.syntax import *
from orcinus.signals import Signal
from orcinus.utils import cached_property

logger = logging.getLogger('orcinus')


class Scope(abc.ABC):
    @abc.abstractmethod
    def resolve(self, name: str) -> Optional[NamedSymbol]:
        raise NotImplementedError

    @abc.abstractmethod
    def insert(self, name: str, symbol: NamedSymbol):
        raise NotImplementedError

    @abc.abstractmethod
    def define(self, name: str, node: SyntaxNode):
        raise NotImplementedError


class LexicalScope(Scope):
    def __init__(self, name: str, *, model: SemanticModel = None, parent: LexicalScope = None, location: Location):
        if not model and not parent:
            raise RuntimeError('Required model or parent')
        elif not model:
            model = parent.model

        self.__name = name
        self.__location = location
        self.__model = weakref.ref(model)
        self.__parent = parent

        self.__defined = MultiDict()  # Defined symbols
        self.__imported = dict()  # Imported symbols
        self.__resolved = dict()  # Resolved symbols

    @property
    def name(self) -> str:
        return self.__name

    @property
    def model(self) -> SemanticModel:
        return self.__model()

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
        symbol = None
        symbols = [self.model.symbols[node] for node in self.__defined.getall(name, ())]
        is_populate = False

        # Found single symbol: import overload function
        if len(symbols) == 1:
            symbol = symbols[0]
            if isinstance(symbol, Function):
                symbol = Overload(name, [symbol])

        # Found multiple symbols: import overload function
        elif symbols:
            is_other = any(not isinstance(symbol, Function) for symbol in symbols)
            if is_other:
                for symbol in symbols:
                    self.model.diagnostics.error(symbol.location, f"Already declared symbol `{symbol.name}`")

            is_populate = True
            symbol = Overload(name, (symbol for symbol in symbols if isinstance(symbol, Function)))

        # Resolve symbol in parent scope
        elif self.parent:
            is_populate = False
            symbol = self.parent.resolve(name)
            if isinstance(symbol, Overload):
                symbol = Overload(name, symbol.functions)

        # Import nodes from root scope
        elif name in self.__imported:
            is_populate = False
            symbol = self.__imported[name]

        # Return None, if symbol is not found in current and ascendant scopes
        if not symbol:
            return None

        # Populate overload from parent scopes
        elif is_populate and isinstance(symbol, Overload):
            overload = self.parent.resolve(name)
            if isinstance(overload, Overload):
                symbol.extend(overload)

            overload = self.__imported.get(name)
            if isinstance(overload, Overload):
                symbol.extend(overload)

        # Save resolved symbol
        self.__resolved[name] = symbol
        return symbol

    def insert(self, name: str, symbol: NamedSymbol):
        self.__imported[name] = symbol

    def define(self, name: str, node: SyntaxNode):
        self.__defined[name] = node  # Define symbols

    def __str__(self):
        if self.parent:
            return f'{self.parent} -> {self.name}'
        return self.name

    def __repr__(self):
        class_name = type(self).__name__
        return f'<{class_name}: {self}>'


class ContainerScope(Scope):
    def __init__(self, owner: ContainerSymbol):
        self.__owner = owner
        self.__imported = dict()  # Imported symbols
        self.__resolved = dict()  # Resolved symbols

    @property
    def owner(self):
        return self.__owner

    @cached_property
    def model(self) -> SemanticModel:
        return self.owner.model if isinstance(self.owner, Module) else self.owner.module.model

    def resolve(self, name: str) -> Optional[NamedSymbol]:
        if name in self.__resolved:
            return self.__resolved[name]

        if name in self.__imported:
            return self.model.symbols[self.__imported[name]]

        return None

    def insert(self, name: str, symbol: NamedSymbol):
        if isinstance(symbol, Function):
            if symbol.name in self.__resolved:
                owned = self.__resolved[symbol.name]
                if isinstance(owned, Overload):
                    owned.append(symbol)
            else:
                self.__resolved[symbol.name] = Overload(name, [symbol])
        else:
            self.__resolved[symbol.name] = symbol

    def define(self, name: str, node: SyntaxNode):
        self.__imported[name] = node


class SemanticContext:
    def __init__(self, workspace: Workspace, *, diagnostics: DiagnosticManager = None):
        self.diagnostics = diagnostics if diagnostics is not None else DiagnosticManager()
        self.workspace = workspace
        self.models = {}

    @cached_property
    def builtins_model(self) -> SemanticModel:
        return self.load('__builtins__')

    @cached_property
    def builtins_module(self) -> Module:
        return self.builtins_model.module

    @cached_property
    def boolean_type(self) -> BooleanType:
        type_symbol = self.builtins_module.scope.resolve('bool')
        assert type_symbol, "Not found boolean type in builtins module"
        return cast(BooleanType, type_symbol)

    @cached_property
    def integer_type(self) -> IntegerType:
        type_symbol = self.builtins_module.scope.resolve('int')
        assert type_symbol, "Not found integer type in builtins module"
        return cast(IntegerType, type_symbol)

    @cached_property
    def void_type(self) -> VoidType:
        type_symbol = self.builtins_module.scope.resolve('void')
        assert type_symbol, "Not found void type in builtins module"
        return cast(VoidType, type_symbol)

    @cached_property
    def string_type(self) -> StringType:
        type_symbol = self.builtins_module.scope.resolve('str')
        assert type_symbol, "Not found string type in builtins module"
        return cast(StringType, type_symbol)

    def open(self, document: Document) -> SemanticModel:
        """ Open module from file """
        if document.uri in self.models:
            return self.models[document.uri]

        model = SemanticModel(self, document.name, document.tree, diagnostics=self.diagnostics)
        self.models[document.uri] = model
        model.analyze()
        return model

    def load(self, module_name) -> SemanticModel:
        document = self.workspace.load_document(module_name)
        return self.open(document)


class SemanticModel:
    def __init__(self, context: SemanticContext, module_name: str, tree: ModuleAST, *,
                 diagnostics: DiagnosticManager = None):
        self.diagnostics = diagnostics if diagnostics is not None else DiagnosticManager()
        self.context = context
        self.module_name = module_name
        self.tree = tree

        self.scopes = LazyDict(constructor=self.annotate_scope, initializer=self.initialize_scope)
        self.symbols = LazyEmptyDict(constructor=self.annotate_symbol, initializer=self.initialize_symbol)
        self.types = LazyEmptyDict(constructor=self.resolve_type)

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
        self.import_symbols(self.tree)
        self.define_symbols(self.tree)
        self.emit_functions(self.tree)

    def initialize_scope(self, node: SyntaxNode):
        """ Recursive initialize scope for node """
        for child in node:
            self.scopes.get(child)

    @multimethod
    def annotate_scope(self, node: SyntaxNode) -> LexicalScope:
        if node in self.scopes:
            return self.scopes[node]

        return self.scopes[node.parent]

    @multimethod
    def annotate_scope(self, node: ModuleAST) -> LexicalScope:
        if node in self.scopes:
            return self.scopes[node]

        return LexicalScope(name=self.module_name, model=self, location=node.location)

    @multimethod
    def annotate_scope(self, node: FunctionAST) -> LexicalScope:
        parent = self.scopes[node.parent]
        if node in self.scopes:
            return self.scopes[node]

        return LexicalScope(name=node.name, parent=parent, location=node.location)

    @multimethod
    def annotate_scope(self, node: BlockStatementAST) -> LexicalScope:
        parent = self.scopes[node.parent]
        if node in self.scopes:
            return self.scopes[node]

        name = f'block#{node.location.begin.line}'
        return LexicalScope(name=name, parent=parent, location=node.location)

    @multimethod
    def annotate_scope(self, node: ClassAST) -> LexicalScope:
        parent = self.scopes[node.parent]
        if node in self.scopes:
            return self.scopes[node]

        return LexicalScope(name=node.name, parent=parent, location=node.location)

    @multimethod
    def annotate_scope(self, node: StructAST) -> LexicalScope:
        parent = self.scopes[node.parent]
        if node in self.scopes:
            return self.scopes[node]

        return LexicalScope(name=node.name, parent=parent, location=node.location)

    @multimethod
    def annotate_scope(self, node: InterfaceAST) -> LexicalScope:
        parent = self.scopes[node.parent]
        if node in self.scopes:
            return self.scopes[node]

        return LexicalScope(name=node.name, parent=parent, location=node.location)

    @multimethod
    def annotate_scope(self, node: EnumAST) -> LexicalScope:
        return LexicalScope(self.scopes[node.parent])

    def import_symbols(self, node: ModuleAST):
        """ Import symbols from imported scopes """

        # TODO: Import builtin module
        scope: LexicalScope = self.scopes[node]

        for child in node.imports:
            if isinstance(child, ImportFromAST):
                imported_model = self.context.load(child.module)
                module = imported_model.module

                self.module.add_dependency(module)

                for alias in child.aliases:
                    symbol = module.scope.resolve(alias.name)
                    if not symbol:
                        self.diagnostics.error(
                            alias.location, f"Not found symbol ‘{alias.name}’ in module ‘{child.module}’")
                    scope.insert(alias.alias or alias.name, symbol)

            else:
                self.diagnostics.error(node.location, "Not implemented symbol importing")

    def define_symbols(self, node: SyntaxNode):
        if node.parent:
            scope = self.scopes[node.parent]
            self.define_symbol(scope, node)

        for child in node:
            self.define_symbols(child)

    @multimethod
    def define_symbol(self, scope: LexicalScope, node: SyntaxNode):
        pass  # skip node

    @multimethod
    def define_symbol(self, scope: LexicalScope, node: FunctionAST):
        scope.define(node.name, node)

    @multimethod
    def define_symbol(self, scope: LexicalScope, node: TypeDeclarationAST):
        scope.define(node.name, node)

    @multimethod
    def define_symbol(self, scope: LexicalScope, node: GenericParameterAST):
        scope.define(node.name, node)

    @multimethod
    def define_symbol(self, scope: LexicalScope, node: ParameterAST):
        scope.define(node.name, node)

    @multimethod
    def resolve_type(self, node: TypeAST) -> Type:
        self.diagnostics.error(node.location, "Not implemented type resolving")
        return ErrorType(self.module, node.location)

    @multimethod
    def resolve_type(self, node: NamedTypeAST) -> Type:
        if node.name == 'void':
            return self.context.void_type
        elif node.name == 'bool':
            return self.context.boolean_type
        elif node.name == 'int':
            return self.context.integer_type
        elif node.name == 'str':
            return self.context.string_type

        symbol = self.scopes[node].resolve(node.name)
        if isinstance(symbol, Type):
            return symbol
        elif symbol is None:
            self.diagnostics.error(node.location, f"Type `{node.name}` is not founded in current scope")
        else:
            self.diagnostics.error(node.location,
                                   f"Symbol `{node.name}` is founded in current scope, but it's not type")

        return ErrorType(self.module, node.location)

    @multimethod
    def resolve_type(self, node: ParameterizedTypeAST) -> Type:
        instance_type = self.types[node.type]
        arguments = [self.types[arg] for arg in node.arguments]
        return instance_type.instantiate(self.module, arguments, node.location)

    def annotate_attributes(self, attributes: Sequence[AttributeAST]) -> Sequence[Attribute]:
        return [self.symbols[attrib] for attrib in attributes]

    def annotate_generics(self, parameters: Sequence[GenericParameterAST]) -> Sequence[GenericParameter]:
        return [self.symbols[generic_node] for generic_node in parameters]

    @multimethod
    def annotate_symbol(self, _: SyntaxNode) -> None:
        return None

    @multimethod
    def annotate_symbol(self, node: ModuleAST) -> Module:
        return Module(self, self.module_name, Location(node.location.filename))

    @multimethod
    def annotate_symbol(self, node: FunctionAST) -> Function:
        parent = self.symbols[node.parent]
        if node in self.symbols:
            return self.symbols[node]

        generic_parameters = self.annotate_generics(node.generic_parameters)
        attributes = self.annotate_attributes(node.attributes)

        # if function is method of struct/class/interface and first arguments type is auto, then it type can
        # be inferred to owner type
        if isinstance(parent, Type) and node.parameters and isinstance(node.parameters[0].type, AutoTypeAST):
            parameters = [parent]
            parameters.extend(self.types[param.type] for param in node.parameters[1:])
        else:
            parameters = [self.types[param.type] for param in node.parameters]

        # if return type of function is auto it can be inferred to void
        if isinstance(node.return_type, AutoTypeAST):
            return_type = self.context.void_type
        else:
            return_type = self.types[node.return_type]

        func_type = FunctionType(self.module, parameters, return_type, node.location)
        func = Function(
            parent, node.name, func_type, node.location, generic_parameters=generic_parameters, attributes=attributes)

        for node_param, func_param in zip(node.parameters, func.parameters):
            func_param.name = node_param.name
            func_param.location = node_param.location

            self.symbols[node_param] = func_param

        # assert node not in self.symbols
        return func

    @multimethod
    def annotate_symbol(self, node: StructAST) -> Type:
        parent = self.symbols[node.parent]
        if node in self.symbols:
            return self.symbols[node]

        if self.module == self.context.builtins_module:
            if node.name == "int":
                return IntegerType(parent, node.location)
            elif node.name == "bool":
                return BooleanType(parent, node.location)
            elif node.name == "void":
                return VoidType(parent, node.location)

        attributes = self.annotate_attributes(node.attributes)
        generic_parameters = self.annotate_generics(node.generic_parameters)
        return StructType(parent, node.name, node.location, generic_parameters=generic_parameters,
                          attributes=attributes)

    @multimethod
    def annotate_symbol(self, node: ClassAST) -> Type:
        parent = self.symbols[node.parent]
        if node in self.symbols:
            return self.symbols[node]

        if self.module == self.context.builtins_module:
            if node.name == "str":
                return StringType(parent, node.location)

        attributes = self.annotate_attributes(node.attributes)
        generic_parameters = self.annotate_generics(node.generic_parameters)
        return ClassType(parent, node.name, node.location, generic_parameters=generic_parameters, attributes=attributes)

    @multimethod
    def annotate_symbol(self, node: InterfaceAST) -> Type:
        parent = self.symbols[node.parent]
        if node in self.symbols:
            return self.symbols[node]

        attributes = self.annotate_attributes(node.attributes)
        generic_parameters = self.annotate_generics(node.generic_parameters)
        return InterfaceType(parent, node.name, node.location, generic_parameters=generic_parameters,
                             attributes=attributes)

    @multimethod
    def annotate_symbol(self, node: EnumAST) -> Type:
        parent = self.symbols[node.parent]
        if node in self.symbols:
            return self.symbols[node]

        attributes = self.annotate_attributes(node.attributes)
        return EnumType(parent, node.name, node.location, attributes=attributes)

    @multimethod
    def annotate_symbol(self, node: GenericParameterAST) -> GenericType:
        return GenericType(self.module, node.name, node.location)

    @multimethod
    def annotate_symbol(self, node: FieldAST) -> Symbol:
        parent = self.symbols[node.parent]
        if node in self.symbols:
            return self.symbols[node]

        if not isinstance(parent, Type):
            self.diagnostics.error(node.location, "Field member must be declared in type")
            return ErrorSymbol(self.module, node.location)

        field_type = self.types[node.type]
        return Field(cast(Type, parent), node.name, field_type, node.location)

    @multimethod
    def initialize_symbol(self, node: SyntaxNode):
        pass

    @multimethod
    def initialize_symbol(self, node: ModuleAST):
        module = self.symbols[node]
        for member in node.members:
            if getattr(member, 'name', None):
                module.scope.define(member.name, member)

        for member in node.members:
            symbol = self.symbols[member]
            if member:
                module.add_member(symbol)

    @multimethod
    def initialize_symbol(self, node: GenericParameterAST):
        generic_type: GenericType = self.symbols[node]
        concepts = [self.types[child] for child in node.concepts]
        generic_type.configure(concepts)

    @multimethod
    def initialize_symbol(self, node: TypeDeclarationAST):
        type_symbol = self.symbols[node]
        for member in node.members:
            if getattr(member, 'name', None):
                type_symbol.scope.define(member.name, member)

        for member in node.members:
            symbol = self.symbols[member]
            if member:
                type_symbol.add_member(symbol)

    def emit_functions(self, module: ModuleAST):
        for member in module.members:
            if isinstance(member, FunctionAST):
                self.emit_function(member)

            elif isinstance(member, TypeDeclarationAST):
                for child in member.members:
                    if isinstance(child, FunctionAST):
                        self.emit_function(child)

    def emit_function(self, node: FunctionAST):
        func = self.symbols[node]
        if not isinstance(node.statement, EllipsisStatementAST):
            with self.with_function(func):
                statement = self.emit_statement(node.statement)
                func.statement = statement

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
            -> Optional[Function]:
        func = self.find_function(scope, name, arguments, location)
        if not func:
            arguments = ', '.join(str(arg.type) for arg in arguments)
            message = f'Not found function ‘{name}({arguments})’ in current scope'
            self.diagnostics.error(location, message)
            return None
        return func

    @multimethod
    def emit_statement(self, node: StatementAST) -> Statement:
        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Not implemented statement emitting")

    @multimethod
    def emit_statement(self, node: BlockStatementAST) -> Statement:
        statements = [self.emit_statement(statement) for statement in node.statements]
        return BlockStatement(statements, node.location)

    @multimethod
    def emit_statement(self, node: ElseStatementAST) -> Statement:
        return self.emit_statement(node.statement)

    @multimethod
    def emit_statement(self, node: PassStatementAST) -> Statement:
        return PassStatement(node.location)

    @multimethod
    def emit_statement(self, node: ReturnStatementAST) -> Statement:
        value = self.emit_value(node.value) if node.value else None
        return_type = self.current_function.return_type
        void_type = self.context.void_type

        if isinstance(return_type, ErrorType) or isinstance(value.type, ErrorType):
            pass  # Skip error propagation
        elif value and value.type != return_type:
            message = f"Return statement value must have ‘{return_type}’ type, got ‘{value.type}’"
            self.diagnostics.error(node.location, message)
        elif not value and void_type != return_type:
            message = f"Return statement value must have ‘{return_type}’ type, got ‘{void_type}’"
            self.diagnostics.error(node.location, message)
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
    def emit_assignment(self, node: AttributeExpressionAST, value: Value, location: Location) -> Statement:
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
    def emit_value(self, node: StringExpressionAST) -> Value:
        return cast(Value, self.emit_symbol(node, True))

    @multimethod
    def emit_value(self, node: NamedExpressionAST) -> Value:
        value = self.emit_symbol(node, True)
        if isinstance(value, Value):
            return value

        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Required value, but got another object")

    @multimethod
    def emit_value(self, node: AttributeExpressionAST) -> Value:
        value = self.emit_symbol(node, True)
        if isinstance(value, Value):
            return value

        raise Diagnostic(node.location, DiagnosticSeverity.Error, "Required value, but got another object")

    @multimethod
    def emit_value(self, node: CallExpressionAST) -> Value:
        arguments = [self.emit_value(arg) for arg in node.arguments]
        if any(isinstance(arg.type, ErrorType) for arg in arguments):
            return ErrorValue(self.module, node.location)

        symbol = self.emit_symbol(node.value, False)

        # function call
        if isinstance(symbol, (Overload, Function)):
            func = self.resolve_function(self.scopes[node], symbol.name, arguments, node.location)
            if not func:
                return ErrorValue(self.module, node.location)
            return CallInstruction(func, arguments, node.location)

        # type instance instantiate
        elif isinstance(symbol, Type):
            return NewInstruction(symbol, arguments, node.location)

        # instance function call (uniform calls)
        elif not symbol and isinstance(node.value, NamedExpressionAST):
            func = self.resolve_function(self.scopes[node], node.value.name, arguments, node.location)
            if not func:
                return ErrorValue(self.module, node.location)
            return CallInstruction(func, arguments, node.location)

        self.diagnostics.error(node.location, f'Not found function for call')
        return ErrorValue(self.module, node.location)

    @multimethod
    def emit_value(self, node: UnaryExpressionAST) -> Value:
        arguments = [self.emit_value(node.operand)]
        if any(isinstance(arg.type, ErrorType) for arg in arguments):
            return ErrorValue(self.module, node.location)

        if node.operator == UnaryID.Pos:
            func = self.resolve_function(self.scopes[node], '__pos__', arguments, node.location)
        elif node.operator == UnaryID.Neg:
            func = self.resolve_function(self.scopes[node], '__neg__', arguments, node.location)
        elif node.operator == UnaryID.Not:
            func = self.resolve_function(self.scopes[node], '__not__', arguments, node.location)
        else:
            func = None
            self.diagnostics.error(node.location, "Not implemented unary operator")

        if not func:
            return ErrorValue(self.module, node.location)
        return CallInstruction(func, arguments, node.location)

    @multimethod
    def emit_value(self, node: BinaryExpressionAST) -> Value:
        arguments = [self.emit_value(node.left_operand), self.emit_value(node.right_operand)]
        if any(isinstance(arg.type, ErrorType) for arg in arguments):
            return ErrorValue(self.module, node.location)

        if node.operator == BinaryID.Add:
            func = self.resolve_function(self.scopes[node], '__add__', arguments, node.location)
        elif node.operator == BinaryID.Sub:
            func = self.resolve_function(self.scopes[node], '__sub__', arguments, node.location)
        elif node.operator == BinaryID.Mul:
            func = self.resolve_function(self.scopes[node], '__mul__', arguments, node.location)
        elif node.operator == BinaryID.Div:
            func = self.resolve_function(self.scopes[node], '__div__', arguments, node.location)
        else:
            func = None
            self.diagnostics.error(node.location, "Not implemented binary operator")

        if not func:
            return ErrorValue(self.module, node.location)
        return CallInstruction(func, arguments, node.location)

    @multimethod
    def emit_symbol(self, node: ExpressionAST, is_exists: bool) -> Symbol:
        self.diagnostics.error(node.location, "Not implemented symbol emitting")
        return ErrorSymbol(self.module, node.location)

    @multimethod
    def emit_symbol(self, node: IntegerExpressionAST, is_exists: bool) -> Symbol:
        return IntegerConstant(self.context.integer_type, node.value, node.location)

    @multimethod
    def emit_symbol(self, node: StringExpressionAST, is_exists: bool) -> Symbol:
        return StringConstant(self.context.string_type, node.value, node.location)

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
        elif node.name == 'str':
            return self.context.string_type

        scope = self.scopes[node]
        symbol = scope.resolve(node.name)
        if is_exists and not symbol:
            self.diagnostics.error(node.location, f"Not found symbol `{node.name} in current scope`")
            return ErrorSymbol(self.module, node.location)
        return symbol

    @multimethod
    def emit_symbol(self, node: AttributeExpressionAST, is_exists: bool) -> Symbol:
        instance = self.emit_symbol(node.value, True)
        if isinstance(instance, Value):
            value_type = instance.type
            symbol = value_type.scope.resolve(node.name)

            if isinstance(symbol, Field):
                return BoundedField(instance, symbol, node.location)
            # elif isinstance(symbol, Overload):
            #     return BoundedOverload(instance, symbol)
            elif is_exists and not symbol:
                self.diagnostics.error(node.location, f"Not found symbol `{node.name}` in type {value_type}`")
                return ErrorSymbol(self.module, node.location)

        self.diagnostics.error(node.location, "Not implemented symbol emitting")
        return ErrorSymbol(self.module, node.location)

    @multimethod
    def emit_symbol(self, node: SubscribeExpressionAST, is_exists: bool) -> Symbol:
        symbol = self.emit_symbol(node.value, True)
        arguments = [self.emit_symbol(arg, True) for arg in node.arguments]
        if any(isinstance(arg, ErrorType) or not isinstance(arg, Type) for arg in arguments):
            return ErrorType(self.module, node.location)

        if isinstance(symbol, Type):
            return symbol.instantiate(self.module, cast(Sequence[Type], arguments), node.location)

        self.diagnostics.error(node.location, "Not implemented symbol emitting")
        return ErrorSymbol(self.module, node.location)


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
    def instantiate(self, func: Function, location: Location) -> Function:
        if not func.is_generic:
            return func

        if func.generic_arguments:
            pass

        func_type = self.instantiate(func.type, location)
        new_func = Function(self.module, func.name, func_type)
        raise NotImplementedError

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

    @multimethod
    def instantiate(self, value: CallInstruction, location: Location):
        arguments = [self.instantiate(arg, location) for arg in value.arguments]
        func = self.instantiate(value.function, location)
        return CallInstruction(func, arguments, func.location)


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


class InferenceError(OrcinusError):
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


class ContainerSymbol(abc.ABC):
    """ Abstract base for container symbols """

    def __init__(self):
        self.__members = []
        self.__scope = ContainerScope(self)
        self.on_add_member = Signal()

    @property
    def scope(self) -> ContainerScope:
        return self.__scope

    @property
    def members(self) -> Sequence[OwnedSymbol]:
        return self.__members

    def add_member(self, symbol: OwnedSymbol):
        if symbol in self.__members:
            return

        self.__members.append(symbol)
        if isinstance(symbol, NamedSymbol):
            self.__scope.insert(symbol.name, symbol)
        self.on_add_member(symbol)


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


class ErrorSymbol(Symbol, ContainerSymbol):
    __location: Location

    def __init__(self, module: Module, location: Location):
        super().__init__()

        self.__module = module
        self.__location = location

    @property
    def module(self):
        return self.__module

    @property
    def location(self) -> Location:
        return self.__location

    def __str__(self):
        return '<error>'


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


class Attribute(NamedSymbol):
    def __init__(self, name: str, arguments: Sequence[Value], location: Location):
        self.__location = location
        self.__name = name
        self.__arguments = arguments

    @property
    def name(self) -> str:
        return self.__name

    @property
    def location(self) -> Location:
        return self.__location

    @property
    def arguments(self) -> Sequence[Value]:
        return self.__arguments


class Module(NamedSymbol, ContainerSymbol):
    def __init__(self, model: SemanticModel, name, location: Location):
        super(Module, self).__init__()

        self.__model = model
        self.__name = name
        self.__location = location
        self.__instances = {}  # Map of all generic instances
        self.__functions = []
        self.__types = []
        self.__dependencies = []

    @property
    def model(self) -> SemanticModel:
        return self.__model

    @property
    def name(self) -> str:
        return self.__name

    @property
    def location(self) -> Location:
        return self.__location

    @property
    def dependencies(self) -> Sequence[Module]:
        return self.__dependencies

    @property
    def functions(self) -> Sequence[Function]:
        return self.__functions

    @property
    def types(self) -> Sequence[Type]:
        return self.__types

    def add_function(self, func: Function):
        self.__functions.append(func)

    def add_type(self, type: Type):
        self.__types.append(type)

    def find_instance(self, generic: GenericSymbol, generic_arguments):
        key = (generic, tuple(generic_arguments))
        return self.__instances.get(key)

    def register_instance(self, generic: GenericSymbol, generic_arguments, instance: GenericSymbol):
        key = (generic, tuple(generic_arguments))
        self.__instances[key] = instance

    def add_dependency(self, module: Module):
        self.__dependencies.append(module)


class Type(MangledSymbol, GenericSymbol, OwnedSymbol, ContainerSymbol, abc.ABC):
    """ Abstract base for all types """

    def __init__(self, owner: ContainerSymbol, name: str, location: Location, *,
                 attributes=None, generic_parameters=None, generic_arguments=None, definition=None):
        super(Type, self).__init__()

        self.__owner = owner
        self.__name = name
        self.__location = location
        self.__attributes = tuple(attributes or [])
        self.__generic_parameters = tuple(generic_parameters or [])
        self.__generic_arguments = tuple(generic_arguments or [])
        self.__definition = definition

        self.module.add_type(self)

    @property
    def attributes(self) -> Sequence[Attribute]:
        return self.__attributes

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

    @property
    def methods(self) -> Sequence[Function]:
        return [member for member in self.members if isinstance(member, Function)]

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other

    def __ne__(self, other):
        return not (self == other)

    def instantiate(self, module: Module, generic_arguments: Sequence[Type], location: Location):
        raise Diagnostic(location, DiagnosticSeverity.Error, "Can not instantiate non generic type")


class ErrorType(Type):
    """ Instance of this type is represented errors in semantic analyze """

    def __init__(self, module: Module, location: Location):
        super(ErrorType, self).__init__(module, '<error>', location)


class VoidType(Type):
    def __init__(self, owner: ContainerSymbol, location: Location):
        super(VoidType, self).__init__(owner, 'void', location)


class BooleanType(Type):
    def __init__(self, owner: ContainerSymbol, location: Location):
        super(BooleanType, self).__init__(owner, 'bool', location)


class StringType(Type):
    def __init__(self, owner: ContainerSymbol, location: Location):
        super(StringType, self).__init__(owner, 'str', location)


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
    @property
    def fields(self) -> Sequence[Field]:
        return tuple(field for field in self.members if isinstance(field, Field))

    def instantiate(self, module: Module, generic_arguments: Sequence[Type], location: Location):
        instance = module.find_instance(self.definition or self, generic_arguments)
        if not instance:
            context = InstantiateContext(module)
            context.aggregate(self.generic_parameters, generic_arguments)
            return StructType(module, self.name, self.location, generic_arguments=generic_arguments, definition=self)
        module.register_instance(self.definition or self, generic_arguments, instance)
        return instance


class InterfaceType(Type):
    @property
    def fields(self) -> Sequence[Field]:
        return tuple(field for field in self.members if isinstance(field, Field))

    def instantiate(self, module: Module, generic_arguments: Sequence[Type], location: Location):
        instance = module.find_instance(self.definition or self, generic_arguments)
        if not instance:
            context = InstantiateContext(module)
            context.aggregate(self.generic_parameters, generic_arguments)
            return InterfaceType(module, self.name, self.location, generic_arguments=generic_arguments, definition=self)
        module.register_instance(self.definition or self, generic_arguments, instance)
        return instance


class EnumType(Type):
    pass


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
    def __init__(self,
                 owner: ContainerSymbol,
                 name: str,
                 location: Location, *,
                 generic_parameters=None,
                 generic_arguments=None,
                 definition=None):
        super().__init__(
            owner, name, location,
            generic_parameters=generic_parameters,
            generic_arguments=generic_arguments,
            definition=definition
        )

        self.__concepts = tuple()

    @property
    def is_generic(self) -> bool:
        return True

    @property
    def concepts(self) -> Sequence[InterfaceType]:
        return self.__concepts

    def configure(self, concepts: Sequence[InterfaceType]):
        assert not self.__concepts
        self.__concepts = concepts

        for concept in concepts:
            concept.on_add_member.connect(self.clone_member, True)
            for member in concept.members:
                self.clone_member(member)

    @multimethod
    def clone_member(self, member: Symbol):
        raise NotImplementedError

    @multimethod
    def clone_member(self, member: Function):
        parameters = [self] + list(member.function_type.parameters[1:])
        function_type = FunctionType(self.owner, parameters, member.function_type.return_type,
                                     location=member.function_type.location)
        func = Function(self,
                        member.name,
                        function_type,
                        location=member.location,
                        generic_arguments=member.generic_arguments,
                        generic_parameters=member.generic_parameters,
                        attributes=member.attributes
                        )

        for param, original in zip(func.parameters, member.parameters):
            param.name = original.name
            param.location = original.location

        self.add_member(func)


class ErrorValue(Value):
    """ Instance of this class is represented errors in semantic analyze """

    def __init__(self, module: Module, location: Location):
        super(ErrorValue, self).__init__(ErrorType(module, location), location)

    def __str__(self):
        return '<error>'


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
                 generic_parameters=None, generic_arguments=None, definition=None, attributes=None):
        super(Function, self).__init__(func_type, location)
        self.__owner = owner
        self.__name = name
        self.__parameters = tuple(
            Parameter(self, f'arg{idx}', param_type) for idx, param_type in enumerate(func_type.parameters)
        )
        self.__statement = None
        self.__generic_parameters = tuple(generic_parameters or [])
        self.__generic_arguments = tuple(generic_arguments or [])
        self.__definition = definition
        self.__variables = []
        self.__attributes = tuple(attributes or [])

        self.module.add_function(self)

    @property
    def owner(self) -> ContainerSymbol:
        return self.__owner

    @property
    def name(self) -> str:
        return self.__name

    @property
    def attributes(self) -> Sequence[Attribute]:
        return self.__attributes

    @cached_property
    def is_native(self) -> bool:
        return any(attr.name == 'native' for attr in self.attributes)

    @cached_property
    def native_name(self) -> Optional[str]:
        native_attr = next((attr for attr in self.attributes if attr.name == 'native'), None)
        if native_attr:
            if len(native_attr.arguments) == 1:
                assert isinstance(native_attr.arguments[0], StringConstant)
                return cast(StringConstant, native_attr.arguments[0]).value
            return self.name

    @cached_property
    def mangled_name(self) -> str:
        if self.is_native:
            return self.native_name
        context = MangledContext()
        return context.mangle(self)

    @cached_property
    def is_inline(self) -> bool:
        return any(attr.name == 'inline' for attr in self.attributes)

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
    def __init__(self, name: str, functions: Iterable[Function]):
        self.__name = name
        self.__functions = list(functions)

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


class StringConstant(Value):
    def __init__(self, value_type: StringType, value: str, location: Location):
        super(StringConstant, self).__init__(value_type, location)

        self.value = value

    def __str__(self):
        value = self.value.replace('"', '\\"')
        return f'"{value}"'


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
