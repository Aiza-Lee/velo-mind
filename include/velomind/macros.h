#pragma once

#if defined(__MSVC__)
#    define VELOMIND_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#    define VELOMIND_FORCE_INLINE inline __attribute__((always_inline))
#else
#    define VELOMIND_FORCE_INLINE inline
#endif
