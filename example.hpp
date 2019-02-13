#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace example {
    class Position {
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
    class Location {
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
} // namespace example
