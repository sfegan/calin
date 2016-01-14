# - Try to find CTA CamerasToACTL
# Once done this will define
# CTA_CAMERASTOACTL_FOUND - System has CTA CamerasToACTL
# CTA_CAMERASTOACTL_INCLUDE_DIRS - The CTA CamerasToACTL include directories
# CTA_CAMERASTOACTL_LIBRARIES - The libraries needed to use CTA CamerasToACTL

FIND_PATH ( CTA_CAMERASTOACTL_INCLUDE_DIR L0.pb.h
  HINTS ${CTA_CAMERASTOACT_DIR} PATH_SUFFIXES include)
FIND_LIBRARY ( CTA_CAMERASTOACTL_LIBRARY_ZFITSIO NAMES ZFitsIO
  HINTS ${CTA_CAMERASTOACT_DIR} PATH_SUFFIXES lib)
FIND_LIBRARY ( CTA_CAMERASTOACTL_LIBRARY_ACTLCORE NAMES ACTLCore
  HINTS ${CTA_CAMERASTOACT_DIR} PATH_SUFFIXES lib)

SET ( CTA_CAMERASTOACTL_LIBRARIES ${CTA_CAMERASTOACTL_LIBRARY_ZFITSIO}
${CTA_CAMERASTOACTL_LIBRARY_ACTLCORE})
SET ( CTA_CAMERASTOACTL_INCLUDE_DIRS ${CTA_CAMERASTOACTL_INCLUDE_DIR} )

INCLUDE ( FindPackageHandleStandardArgs )
# handle the QUIETLY and REQUIRED arguments and set ZMQ_FOUND to TRUE
# if all listed variables are TRUE
FIND_PACKAGE_HANDLE_STANDARD_ARGS( CTA_CAMERASTOACTL DEFAULT_MSG
  CTA_CAMERASTOACTL_LIBRARY_ZFITSIO CTA_CAMERASTOACTL_LIBRARY_ACTLCORE
  CTA_CAMERASTOACTL_INCLUDE_DIR )