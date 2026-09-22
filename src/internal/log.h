#pragma once

#include <format>
#include <iostream>
#include <string_view>
#include <utility>

namespace velomind::internal {

enum class LogLevel {
    Debug,
    Info,
    Warn,
    Error,
};

inline constexpr auto log_level_name(LogLevel level) -> const char* {
    switch (level) {
        case LogLevel::Debug: return "Debug";
        case LogLevel::Info:  return "Info";
        case LogLevel::Warn:  return "Warn";
        case LogLevel::Error: return "Error";
    }
    return "Unknown";
}

using LogHandler = void (*)(LogLevel level, std::string_view message);

namespace detail {

    inline auto _default_handler(LogLevel level, std::string_view msg) -> void {
        std::cerr << std::format("[velomind {}] {}\n", log_level_name(level), msg);
    }

    inline LogHandler& _handler_slot() {
        static LogHandler h = &_default_handler;
        return h;
    }

    inline auto _log_emit(LogLevel level, std::string_view message) -> void {
        _handler_slot()(level, message);
    }

}

inline auto set_log_handler(LogHandler handler) -> void {
    detail::_handler_slot() = (handler != nullptr) ? handler : &detail::_default_handler;
}

inline auto get_log_handler() -> LogHandler {
    return detail::_handler_slot();
}

template <typename... Args>
auto log(LogLevel level, std::format_string<Args...> fmt, Args&&... args) -> void {
    detail::_log_emit(level, std::format(fmt, std::forward<Args>(args)...));
}

} // namespace velomind::internal


#define VELOMIND_LOG(level, ...)                                                \
    ::velomind::internal::log(::velomind::internal::LogLevel::level, __VA_ARGS__)
