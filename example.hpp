#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace example {
    class Position : {
    public:
        std::int64_t __line;
        std::int64_t get_line() const {
            return this->__line;
        }
        void set_line(std::int64_t value) {
            this->__line = value;
        }
        std::int64_t __column;
        std::int64_t get_column() const {
            return this->__column;
        }
        void set_column(std::int64_t value) {
            this->__column = value;
        }
        Position();
        Position(std::int64_t line);
        Position(std::int64_t line, std::int64_t column);
    };
    class Location : {
    public:
        std::string __filename;
        std::string get_filename() const {
            return this->__filename;
        }
        void set_filename(std::string value) {
            this->__filename = value;
        }
        ::example::Position __begin;
        ::example::Position get_begin() const {
            return this->__begin;
        }
        void set_begin(::example::Position value) {
            this->__begin = value;
        }
        ::example::Position __end;
        ::example::Position get_end() const {
            return this->__end;
        }
        void set_end(::example::Position value) {
            this->__end = value;
        }
        Location(std::string filename);
        Location(std::string filename, ::example::Position position);
    };
    enum class TokenID {
        EndOfLine = -(1),
        NewLine   = 0,
        Whitespace,
        Name,
        Number,
        LeftParenthesis,
        RightParenthesis,
        LeftSquare,
        RightSquare,
        LeftCurly,
        RightCurly,
        Dot,
        Comma,
        Colon,
        Semicolon,
        Comment,
        Indent,
        Undent,
        Def,
        Pass,
        Import,
        From,
        Return,
        Yield,
        As,
        Then,
        Ellipsis,
        If,
        Elif,
        Else,
        While,
        Struct,
        Class,
        Interface,
        Enum,
        Equal,
        EqEqual,
        NotEqual,
        LessEqual,
        GreatEqual,
        Less,
        Great,
        Star,
        DoubleStar,
        Plus,
        Minus,
        Slash,
        DoubleSlash,
        Tilde,
        String,
        And,
        Or,
        Xor,
        LogicAnd,
        LogicOr,
        For,
        In
    };
    class SyntaxSymbol : {
    public:
        SyntaxSymbol(const SyntaxSymbol&)   = delete;
        SyntaxSymbol&               operator=(const SyntaxSymbol&)          = delete;
        virtual ::example::Location get_location() const                    = 0;
        virtual void                set_location(::example::Location value) = 0;
    };
    class SyntaxToken : public ::example::SyntaxSymbol {
    public:
        ::example::TokenID __id;
        ::example::TokenID get_id() const {
            return this->__id;
        }
        void set_id(::example::TokenID value) {
            this->__id = value;
        }
        std::string __value;
        std::string get_value() const {
            return this->__value;
        }
        void set_value(std::string value) {
            this->__value = value;
        }
        ::example::Location __location;
        ::example::Location get_location() const {
            return this->__location;
        }
        void set_location(::example::Location value) {
            this->__location = value;
        }
        SyntaxToken(::example::TokenID id, std::string value, ::example::Location location);
    };
} // namespace example
