project(tools LANGUAGES CXX)
find_package(Threads REQUIRED)

# Linking
set(TOOLS_LINK_LIBS Threads::Threads)

add_executable(bpp9000_score_test bpp9000_score_test.cpp)

target_include_directories(bpp9000_score_test
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(bpp9000_score_test
    PRIVATE
    ${TOOLS_LINK_LIBS})

add_executable(bpp9000_test_generator bpp9000_test_generator.cpp)

target_include_directories(bpp9000_test_generator
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(bpp9000_test_generator
    PRIVATE
    ${TOOLS_LINK_LIBS})
