set(allom_sources FatesTestAllometry.F90)

set(NETCDF_C_DIR ${NETCDF_C_PATH})
set(NETCDF_FORTRAN_DIR ${NETCDF_F_PATH})

FIND_PATH(NETCDFC_FOUND libnetcdf.a ${NETCDF_C_DIR}/lib)
FIND_PATH(NETCDFF_FOUND libnetcdff.a ${NETCDF_FORTRAN_DIR}/lib)


include_directories(${NETCDF_C_DIR}/include
                    ${NETCDF_FORTRAN_DIR}/include)

link_directories(${NETCDF_C_DIR}/lib
                ${NETCDF_FORTRAN_DIR}/lib
                ${PFUNIT_TOP_DIR}/lib)
                
                
add_executable(FATES_allom_exe ${allom_sources})

target_link_libraries(FATES_allom_exe
                  netcdf
                  netcdff
                  fates
                  csm_share
                  funit)