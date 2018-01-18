# FindMetis.cmake: find Metis library.
#
# You can set Metis_DIR to give hint on the installation directory.
#
# FindMetis.cmake defines:
# - Metis_INCLUDE_DIRS: list of directories where to find headers to compile with.
# - Metis_LIBRARIES: list of libraries to link with.
# - Metis_FOUND: indicate whether the package was found or not.

# Look for headers.
find_path(Metis_INCLUDE_DIRS NAMES metis.h PATHS $ENV{Metis_DIR} $ENV{CPATH} $ENV{C_INCLUDE_PATH} $ENV{CPLUS_INCLUDE_PATH} PATH_SUFFIXES include)

# Look for libraries.
find_library(Metis_LIBRARIES NAMES metis PATHS $ENV{Metis_DIR} $ENV{LD_LIBRARY_PATH} PATH_SUFFIXES lib)

# Look for version.
set(Metis_VERSION "")
if(DEFINED Metis_FIND_VERSION)
  if(IS_DIRECTORY "${Metis_INCLUDE_DIRS}")
    set(Metis_VERSION "Metis_VERSION-NOTFOUND")

    file(READ "${Metis_INCLUDE_DIRS}/metis.h" Metis_HDR)
    if(DEFINED Metis_FIND_VERSION_MAJOR)
      string(REGEX MATCH   "define[ \t]+METIS_VER_MAJOR[ \t]+[0-9]" DEFINE_VERSION_MAJOR_X "${Metis_HDR}")
      string(REGEX REPLACE "define[ \t]+METIS_VER_MAJOR[ \t]+"      "" X "${DEFINE_VERSION_MAJOR_X}")
      set(Metis_VERSION "${X}")
    endif()
    if(DEFINED Metis_FIND_VERSION_MINOR)
      string(REGEX MATCH   "define[ \t]+METIS_VER_MINOR[ \t]+[0-9]" DEFINE_VERSION_MINOR_Y "${Metis_HDR}")
      string(REGEX REPLACE "define[ \t]+METIS_VER_MINOR[ \t]+"      "" Y "${DEFINE_VERSION_MINOR_Y}")
      set(Metis_VERSION "${Metis_VERSION}.${Y}")
    endif()
    if(DEFINED Metis_FIND_VERSION_SUBMINOR)
      string(REGEX MATCH   "define[ \t]+METIS_VER_SUBMINOR[ \t]+[0-9]" DEFINE_VERSION_SUBMINOR_Z "${Metis_HDR}")
      string(REGEX REPLACE "define[ \t]+METIS_VER_SUBMINOR[ \t]+"      "" Z "${DEFINE_VERSION_SUBMINOR_Z}")
      set(Metis_VERSION "${Metis_VERSION}.${Z}")
    endif()
  endif()
endif()

# Handle REQUIRED, QUIET and version-related arguments.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Metis FOUND_VAR Metis_FOUND VERSION_VAR Metis_VERSION REQUIRED_VARS Metis_INCLUDE_DIRS Metis_LIBRARIES)
