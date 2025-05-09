set(phen_sources MLPhenologyTest.F90)

set(FTORCH_DIR "/glade/u/home/jatkinson/FTorch/bin/oneapi_2023.2.1_torch_2.1")

include_directories(${FTORCH_DIR}/include/ftorch)
link_directories(${FTORCH_DIR}/lib64)

link_directories(${PFUNIT_TOP_DIR}/lib)

add_executable(ml_phen_exe ${phen_sources})

target_include_directories(ml_phen_exe PRIVATE ${FTORCH_LOC}/include/ftorch)
target_link_directories(ml_phen_exe PRIVATE ${FTORCH_LOC}/lib64)
target_link_libraries(ml_phen_exe
                  fates
                  ftorch
                  csm_share
                  funit)