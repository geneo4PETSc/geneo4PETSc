# Add test.

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/dummy.sh" "${CMAKE_CURRENT_BINARY_DIR}/dummy.sh" @ONLY)

file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.inp)
foreach(file ${files})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${file}" "${CMAKE_CURRENT_BINARY_DIR}/${file}" COPYONLY)
endforeach()

file(GLOB files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*.ref)
foreach(file ${files})
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${file}" "${CMAKE_CURRENT_BINARY_DIR}/${file}" COPYONLY)
endforeach()

add_test(NAME dummy COMMAND ./dummy.sh WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
