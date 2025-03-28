# - Try to find CTA ADH_APIS
# Once done this will define
# CTA_ADH_APIS_FOUND - System has CTA ADH_APIS
# CTA_ADH_APIS_INCLUDE_DIRS - The CTA ADH_APIS include directories
# CTA_ADH_APIS_LIBRARIES - The libraries needed to use CTA ADH_APIS

FIND_PATH ( CTA_ADH_APIS_INCLUDE_DIRS R1v1.pb.h HINTS ${CTA_ADH_APIS_DIR})
FIND_LIBRARY ( CTA_ADH_APIS_LIBRARY_ZFITSIO NAMES ZFitsIO HINTS ${CTA_ADH_APIS_DIR})
FIND_LIBRARY ( CTA_ADH_APIS_LIBRARY_ADHCORE NAMES ADHCore HINTS ${CTA_ADH_APIS_DIR})

SET ( CTA_ADH_APIS_LIBRARIES ${CTA_ADH_APIS_LIBRARY_ZFITSIO}
      ${CTA_ADH_APIS_LIBRARY_ADHCORE})
SET ( CTA_ADH_APIS_INCLUDE_DIRS ${CTA_ADH_APIS_INCLUDE_DIRS} )

INCLUDE ( FindPackageHandleStandardArgs )
# handle the QUIETLY and REQUIRED arguments and set CTA_ADH_APIS_FOUND to TRUE
# if all listed variables are TRUE
FIND_PACKAGE_HANDLE_STANDARD_ARGS( CTA_ADH_APIS DEFAULT_MSG
  CTA_ADH_APIS_LIBRARY_ZFITSIO CTA_ADH_APIS_LIBRARY_ADHCORE
  CTA_ADH_APIS_INCLUDE_DIRS )
