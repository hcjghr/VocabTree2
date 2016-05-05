cmake_minimum_required(VERSION 2.8)
project(VocabTreeOpenMVG)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -ffast-math -Wall -mfpmath=sse -msse2 -funroll-loops -march=core2 -Wall -fopenmp")

get_filename_component(VOCABTREE_LIB_PATH "../.." ABSOLUTE)

#set(VOCABTREE_LIB_PATH ${VOCABTREE_ABS_PATH})

FIND_PACKAGE(OpenMVG REQUIRED)
INCLUDE_DIRECTORIES(${OPENMVG_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${VOCABTREE_LIB_PATH}/lib/ann_1.1/include/ANN
${VOCABTREE_LIB_PATH}/lib/ann_1.1_char/include/ANN
${VOCABTREE_LIB_PATH}/lib/imagelib
${VOCABTREE_LIB_PATH}/VocabLib
${VOCABTREE_LIB_PATH}/lib/zlib/include)

LINK_DIRECTORIES(${VOCABTREE_LIB_PATH}/lib ${VOCABTREE_LIB_PATH}/VocabLib ${VOCABTREE_LIB_PATH}/lib/zlib/lib)

ADD_EXECUTABLE(VocabLearn_OpenMVG main_VocabLearn_openmvg.cpp)
TARGET_LINK_LIBRARIES(VocabLearn_OpenMVG ${OPENMVG_LIBRARIES} vocab ANN ANN_char image z)
ADD_EXECUTABLE(VocabBuildDB_OpenMVG main_VocabBuildDB_openmvg.cpp)
TARGET_LINK_LIBRARIES(VocabBuildDB_OpenMVG ${OPENMVG_LIBRARIES} vocab ANN ANN_char image z)
ADD_EXECUTABLE(VocabMatch_OpenMVG main_VocabMatch_openmvg.cpp)
TARGET_LINK_LIBRARIES(VocabMatch_OpenMVG ${OPENMVG_LIBRARIES} vocab ANN ANN_char image z)


ADD_EXECUTABLE(VocabLearnBuildDB_OpenMVG main_VocabLearnBuildDB_openmvg.cpp)
TARGET_LINK_LIBRARIES(VocabLearnBuildDB_OpenMVG ${OPENMVG_LIBRARIES} vocab ANN ANN_char image z)


