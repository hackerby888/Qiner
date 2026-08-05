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

# Convert cached BTC candles into the bpp9000 task file
add_executable(bpp9000_task_from_candles bpp9000_task_from_candles.cpp)

target_include_directories(bpp9000_task_from_candles
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(bpp9000_task_from_candles
    PRIVATE
    ${TOOLS_LINK_LIBS})

add_executable(broadcastMessageSolution
    broadcastMessageSolution.cpp
    ${CMAKE_SOURCE_DIR}/src/keyUtils.cpp)

target_include_directories(broadcastMessageSolution
    PRIVATE
    ${CMAKE_SOURCE_DIR}/src)

target_link_libraries(broadcastMessageSolution
    PRIVATE
    ${TOOLS_LINK_LIBS})

if(WIN32)
    target_compile_definitions(broadcastMessageSolution PRIVATE NOMINMAX)
    target_link_libraries(broadcastMessageSolution PRIVATE wsock32 ws2_32)
endif()
