# src/examples/periodic

set(EXAMPLE_SOURCES
    testpercoul)
 
# Create executables for example applications
foreach(example ${EXAMPLE_SOURCES})
  add_mad_executable(${example} ${example}.cc "MADchem")
  add_dependencies(numerical-examples-madness ${example})
endforeach()

find_package(FFTW)
if (TARGET FFTW::Double)
  add_mad_executable(testpercoulfft "test.cc;testfuns.cc;testpc.h" "MADchem")
  target_link_libraries(testpercoulfft PRIVATE FFTW::Double)
  add_dependencies(numerical-examples-madness testpercoulfft)
endif()
