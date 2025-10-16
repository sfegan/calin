# - Find Geant4 library
# This module sets up Geant4 information
# It defines:
# GEANT4_FOUND               If the Geant4 is found
# GEANT4_INCLUDE_DIR         PATH to the include directory
# GEANT4_LIBRARY_DIR         PATH to the library directory
# GEANT4_LIBRARIES           Most common libraries
# GEANT4_LIBRARIES_WITH_VIS  Most common libraries with visualization

find_program(GEANT4_CONFIG NAMES geant4-config
             PATHS $ENV{GEANT4_INSTALL}/bin
                   ${GEANT4_INSTALL}/bin
                   /usr/local/bin /opt/local/bin)

if(GEANT4_CONFIG)
  set(GEANT4_FOUND TRUE)

  execute_process(COMMAND ${GEANT4_CONFIG} --prefix
                  OUTPUT_VARIABLE GEANT4_PREFIX
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${GEANT4_CONFIG} --version
                  OUTPUT_VARIABLE GEANT4_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE)

  message(STATUS "Found Geant4: ${GEANT4_PREFIX} (${GEANT4_VERSION})")

  string(REPLACE "." ";" GEANT4_VERSION_LIST ${GEANT4_VERSION})
  list(GET GEANT4_VERSION_LIST 0 GEANT4_VERSION_MAJOR)
  list(GET GEANT4_VERSION_LIST 1 GEANT4_VERSION_MINOR)
  list(GET GEANT4_VERSION_LIST 2 GEANT4_VERSION_PATCH)

  find_path(GEANT4_INCLUDE_DIR NAMES G4RunManager.hh
    PATHS ${GEANT4_PREFIX}/include
          ${GEANT4_PREFIX}/include/Geant4
          ${GEANT4_PREFIX}/include/Geant4/Geant4.${GEANT4_VERSION}
          ${GEANT4_PREFIX}/include/Geant4/Geant4.${GEANT4_VERSION}/Geant4
          ${GEANT4_PREFIX}/include/Geant4/Geant4.${GEANT4_VERSION_MAJOR}.${GEANT4_VERSION_MINOR}/
          ${GEANT4_PREFIX}/include/Geant4/Geant4.${GEANT4_VERSION_MAJOR}.${GEANT4_VERSION_MINOR}/Geant4
          ${GEANT4_PREFIX}/include/Geant4/Geant4.${GEANT4_VERSION_MAJOR}/
          ${GEANT4_PREFIX}/include/Geant4/Geant4.${GEANT4_VERSION_MAJOR}/Geant4)

  find_library(GEANT4_G4RUN_LIB NAMES G4run
    PATHS ${GEANT4_PREFIX}/lib
          ${GEANT4_PREFIX}/lib/Geant4
          ${GEANT4_PREFIX}/lib/Geant4/Geant4.${GEANT4_VERSION}
          ${GEANT4_PREFIX}/lib/Geant4/Geant4.${GEANT4_VERSION}/Geant4
          ${GEANT4_PREFIX}/lib/Geant4/Geant4.${GEANT4_VERSION_MAJOR}.${GEANT4_VERSION_MINOR}/
          ${GEANT4_PREFIX}/lib/Geant4/Geant4.${GEANT4_VERSION_MAJOR}.${GEANT4_VERSION_MINOR}/Geant4
          ${GEANT4_PREFIX}/lib/Geant4/Geant4.${GEANT4_VERSION_MAJOR}/
          ${GEANT4_PREFIX}/lib/Geant4/Geant4.${GEANT4_VERSION_MAJOR}/Geant4)

get_filename_component(GEANT4_LIBRARY_DIR ${GEANT4_G4RUN_LIB} DIRECTORY)

execute_process(COMMAND ${GEANT4_CONFIG} --libs
                OUTPUT_VARIABLE GEANT4_LIBRARIES
                OUTPUT_STRIP_TRAILING_WHITESPACE)
else()
  set(GEANT4_FOUND FALSE)
  message(WARNING "NOT Found Geant4: set GEANT4_INSTALL env.")
endif()
