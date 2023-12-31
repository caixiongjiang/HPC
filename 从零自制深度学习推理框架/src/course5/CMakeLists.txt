cmake_minimum_required(VERSION 3.16)
project(kuiper_datawhale_course5)
set(CMAKE_CXX_STANDARD 17)
find_package(benchmark REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Armadillo REQUIRED)
find_package(glog REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(GTest REQUIRED)

set(link_lib glog::glog GTest::gtest)
if (!WIN32)
    set(link_lib "${link_lib} pthread")
endif ()
set(link_math_lib ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
aux_source_directory(./test DIR_TEST_ARMA)
aux_source_directory(./source DIR_SOURCE_ARMA)
aux_source_directory(./source/layer/abstract DIR_ABSTRACT_LAYER)
aux_source_directory(./source/layer/details DIR_DETAIL_LAYER)

add_executable(kuiper_datawhale_course5 main.cpp ${DIR_TEST_ARMA} ${DIR_SOURCE_ARMA} ${DIR_DETAIL_LAYER} ${DIR_ABSTRACT_LAYER} source/layer/details/sigmoid.cpp source/layer/details/sigmoid.cpp source/layer/details/sigmoid.hpp)
target_link_libraries(kuiper_datawhale_course5 ${link_lib} ${link_math_lib} OpenMP::OpenMP_CXX)

target_include_directories(kuiper_datawhale_course5 PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(kuiper_datawhale_course5 PUBLIC ${GTest_INCLUDE_DIR})
target_include_directories(kuiper_datawhale_course5 PUBLIC ${Armadillo_INCLUDE_DIR})
target_include_directories(kuiper_datawhale_course5 PUBLIC ./include)

enable_testing()
