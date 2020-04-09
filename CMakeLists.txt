cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(mlpga LANGUAGES CXX CUDA)

set(MLPGA_USE_GPU true CACHE BOOL "Use the GPU via CUDA")

find_package(Threads)

set(APP ${PROJECT_NAME}_test)

set(SOURCES
    src/genetic.h
    src/init.h
    src/Network.h
    src/transwarp.h
    src/utils.h
)

if (MLPGA_USE_GPU)
    list(APPEND SOURCES
         src/gpufunc.h
         src/gpufunc.cu
         )
endif()

add_executable(${APP} ${SOURCES} src/test.cpp)

target_compile_features(${APP} PUBLIC cxx_std_11)

if (MLPGA_USE_GPU)
    target_compile_definitions(${APP} PUBLIC MLPGA_USE_GPU)
    target_link_libraries(${APP} curand)
endif()

if (MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /bigobj /EHsc /wd4503 /wd4996 /wd4702 /wd4100")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall -Wextra -Wconversion -Wno-unused-but-set-parameter")
	if (CMAKE_COMPILER_IS_GNUCC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
	endif()
    target_link_libraries(${APP} ${CMAKE_THREAD_LIBS_INIT})
endif()
