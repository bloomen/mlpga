cmake_minimum_required(VERSION 2.8)
project(mlpga)

find_package(Threads)

set(APP ${PROJECT_NAME}_test)

set(SOURCES
src/genetic.h
src/init.h
src/Network.h
src/utils.h
src/transwarp.h
)

add_executable(${APP} ${SOURCES} src/test.cpp)

if (MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17 /W4 /bigobj /EHsc /wd4503 /wd4996 /wd4702 /wd4100")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pedantic -Wall -Wextra -Wconversion -Wno-unused-but-set-parameter")
	if (CMAKE_COMPILER_IS_GNUCC)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
	endif()
    target_link_libraries(${APP} ${CMAKE_THREAD_LIBS_INIT})
endif()
