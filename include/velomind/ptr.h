#pragma once

#define VELOMIND_EXPORT_PTR(type)                 \
    using p##type         = type*;                \
    using pp##type        = type**;               \
    using pConst##type    = const type*;          \
    using ppConst##type   = const type**;
