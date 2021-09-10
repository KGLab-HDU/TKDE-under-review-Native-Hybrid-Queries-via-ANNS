set(CMAKE_CXX_STANDARD 11)

add_executable(test_dng_index test_dng_index.cpp)
target_link_libraries(test_dng_index ${PROJECT_NAME})

add_executable(test_dng_optimized_search test_dng_optimized_search.cpp)
target_link_libraries(test_dng_optimized_search ${PROJECT_NAME})

add_executable(test_dng_hybrid_search test_dng_hybrid_search.cpp)
target_link_libraries(test_dng_hybrid_search ${PROJECT_NAME})
