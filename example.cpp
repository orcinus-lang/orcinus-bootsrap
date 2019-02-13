#include "example.hpp"

::example::Position::Position() {
    this->set_line(1);
    this->set_column(1);
}
::example::Position::Position(std::int64_t line) {
    this->set_line(line);
    this->set_column(1);
}
::example::Position::Position(std::int64_t line, std::int64_t column) {
    this->set_line(line);
    this->set_column(column);
}
::example::Location::Location(std::string filename) {
    this->set_filename(filename);
    this->set_begin(::example::Position());
    this->set_end(::example::Position());
}
::example::Location::Location(std::string filename, ::example::Position position) {
    this->set_filename(filename);
    this->set_begin(position);
    this->set_end(position);
}
