include(FetchContent)
message(STATUS "通过 FetchContent 下载 Catch2...")
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.15.3
)
FetchContent_MakeAvailable(Catch2)
