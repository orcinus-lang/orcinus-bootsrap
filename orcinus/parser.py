# Copyright (C) 2019 Vasiliy Sheredeko
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
from __future__ import annotations

from orcinus.core.diagnostics import DiagnosticSeverity, Diagnostic
from orcinus.scanner import Scanner
from orcinus.syntax import *


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
