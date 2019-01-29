
#ifndef Power_EXCEPTION_H
#define Power_EXCEPTION_H

#include <string>
#include <sstream>
#include <exception>

namespace Power {
    const std::string exception_prefix = "Power C++ Exception: ";
    /**
     * custom C++ exception
     */
    struct PowerException: public std::exception {
        std::string message;
        PowerException(const char *detail, const char *file, const int line) {
            std::stringstream ss;
            ss << exception_prefix << "-Custom Exception- ";
            ss << "| [in file] -:- [" << file << "] ";
            ss << "| [on line] -:- [" << line << "] ";
            ss << "| [detail] -:- [" << detail << "].";
            message = ss.str();
        }

        virtual const char *what() const throw() {
            return message.c_str();
        }
    };
};

#define throw_exception(...) {char buffer[1000]; sprintf(buffer, __VA_ARGS__); std::string detail{buffer}; throw PowerException(detail.c_str(), __FILE__, __LINE__);}

#endif
