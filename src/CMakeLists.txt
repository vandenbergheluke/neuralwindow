add_library(neuralwindow STATIC tensor.c)

target_include_directories(neuralwindow PUBLIC include)
set_target_properties(neuralwindow PROPERTIES PUBLIC_HEADER include/tensor.h)

set_target_properties(neuralwindow PROPERTIES SOVERSION 1)
install(TARGETS neuralwindow RUNTIME DESTINATION bin LIBRARY DESTINATION lib ARCHIVE DESTINATION lib PUBLIC_HEADER DESTINATION include)
