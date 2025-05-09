program MLPhenology

  use               FatesConstantsMod, only : r8 => fates_r8
  use              FatesArgumentUtils, only : command_line_arg
  use             FatesUnitTestIOMod,  only : OpenNCFile, GetVar, CloseNCFile, RegisterNCDims
  use               ftorch,            only : torch_tensor, torch_tensor_from_array, torch_kCPU
  
  implicit none

  ! define ML model
  character(len=256) :: cb_torch_model = "/glade/u/home/linnia/FTorch_example/constant_model.pt"
  !character(len=256) :: lstm_torch_model = "/glade/u/home/ayal/phenology-ml-clm/models/example_LSTM_model_v1.pt"
  !type(torch_model) :: model_pytorch                                    

  ! setup Fortran data structures  
  type(torch_tensor), dimension(1)         :: in_tensor, out_tensor
  integer                                  :: in_layout(1) = [1]
  integer                                  :: out_layout(1) = [1] 
  integer,                       parameter :: n_days = 365

  real(8), dimension(10) :: dummy_lai
  integer :: sos_flag, n

  character(len=:),                  allocatable :: datm_file            ! input DATM 
  real(r8),                          allocatable :: ta(:)         ! daily air temperature [degC]
  real(r8),                          allocatable :: pr(:)            ! daily precipitation [mm]
  real(r8),                          allocatable :: sw(:)                ! daily shortwave radiation (W/m2)
  real(r8),                          allocatable :: lai(:)              ! daily LAI (m2/m2)

  !real(r8)                                       :: in_data(1) = 1.0
  real(r8),              dimension(60,4), target :: in_data
  real(r8),                 dimension(1), target :: out_data

  !LOCAL VARIABLES:
  real(r8) :: crit_onset_gdd !critical onset growing degree-day sum
  real(r8) :: annavg_t2m_patch ! annual average 2m air temperature (C) *in CLM it is in K but gets converted. 
  real(r8) :: doy ! day of year (used to identify solstace) 
  real(r8) :: offset_flag 
  real(r8) :: offset_counter
  real(r8) :: dt                ! time step delta t (seconds)
  real(r8) :: dormant_flag
  real(r8) :: days_active
  real(r8) :: onset_flag
  real(r8) :: onset_counter
  integer :: ws_flag

  ! Load forcing data
  datm_file = command_line_arg(1) ! one year of daily ta, pr, sw, lai
  call load_met_forcing(datm_file, ta, pr, sw, lai)

  ! Populate input data (first n_input days)
  in_data(:,1) = lai(1:60)
  in_data(:,2) = ta(1:60)
  in_data(:,3) = pr(1:60)
  in_data(:,4) = sw(1:60)

  ! ======================================
  ! initialize
  offset_flag = 0._r8
  offset_counter = 0._r8
  onset_flag = 1._r8
  onset_counter = 0._r8
  dt = 1800._r8 ! 30 minutes in seconds
  doy = 10


  ! ====================================
  ! test SOS
  n = 10
  dummy_lai = (/1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0/)
  call get_sos(dummy_lai, n, sos_flag)
  print *, "SOS Flag:", sos_flag
  
  ! ======================================
  ! CLM CNSeasonDecidPhenology (approximate)

  ! set misc. constants
  annavg_t2m_patch = 15 ! annual average patch temperature (C)

  ! onset gdd sum from Biome-BGC, v4.1.2
  crit_onset_gdd = exp(4.8_r8 + 0.13_r8*(annavg_t2m_patch))
  print *, crit_onset_gdd

  ! set flag for solstice period (winter->summer = 1, summer->winter = 0)
  if (doy <= 171) then
     ws_flag = 1._r8
  else
     ws_flag = 0._r8
  end if

  ! update offset_counter and test for the end of the offset period
  if (offset_flag == 1.0_r8) then
   ! decrement counter for offset period
   offset_counter = offset_counter - dt
   ! if this is the end of the offset_period, reset phenology
   ! flags and indices
   if (offset_counter < dt/2._r8) then
      offset_flag = 0._r8
      offset_counter = 0._r8
      dormant_flag = 1._r8
      days_active = 0._r8

    end if
  end if


  ! update onset_counter and test for the end of the onset period
  if (onset_flag == 1.0_r8) then
     ! decrement counter for onset period
     onset_counter = onset_counter - dt
     ! if this is the end of the onset period, reset phenology
     ! flags and indices
     if (onset_counter < dt/2._r8) then
       onset_flag = 0.0_r8
       onset_counter = 0.0_r8
     end if
  end if


  !===============
  ! load pytorch model

  !call torch_model_load(model_pytorch, trim(cb_torch_model), torch_kCPU)
  
  ! set input data 
  !in_data = 0.5_r8
  !call random_number(in_data(:,1))   ! fill first column with random numbers in [0,1)

  !print *, in_data(1:5, :)
  
  !print *, in_data
  !print *, 'Shape of input_data:', size(in_data, 1), 'rows ×', size(in_data, 2), 'columns'
  

  !===============
  ! run pytorch model

  !call torch_tensor_from_array(in_tensor(1), in_data, in_layout, torch_kCPU)
  !in_tensor(1) = torch_tensor_from_array(in_data, in_layout, torch_kCPU)
  !print *, "Tensor shape: ", shape(in_tensor)
  !call torch_tensor_from_array(out_tensor(1), out_data, out_layout, torch_kCPU)
  !call torch_model_forward(model_pytorch, in_tensor, out_tensor)

  !call torch_delete(in_tensor(1))
  !call torch_delete(out_tensor(1)) 

  print *, "Hello Phenology"

  contains
  
    !-----------------------------------------------------------------------
    subroutine load_met_forcing ( datm_file, ta, pr, sw, lai)
      ! 
      use FatesConstantsMod, only: r8 => fates_r8
      use FatesUnitTestIOMod, only: OpenNCFile, GetVar, CloseNCFile
    
      implicit none
    
      ! Arguments
      character(len=*), intent(in) :: datm_file
      real(r8), allocatable, intent(out) :: ta(:), pr(:), sw(:), lai(:)
    
      ! Local
      integer :: ncid
    
      ! Allocate arrays
      allocate(ta(365), pr(365), sw(365), lai(365))
    
      ! Open and read
      call OpenNCFile(trim(datm_file), ncid, 'read')
      
      call GetVar(ncid, 'ta', ta)
      call GetVar(ncid, 'pr', pr)
      call GetVar(ncid, 'sw', sw)
      call GetVar(ncid, 'lai', lai)
    
      call CloseNCFile(ncid)
    
    end subroutine load_met_forcing
    
    ! ----------------------------------------------------------------------

    subroutine get_sos(lai, n, sos_flag)
        real(8), dimension(n), intent(in) :: lai
        integer, intent(in) :: n
        integer, intent(out) :: sos_flag
        real(8), dimension(n) :: t, kv
        real(8) :: a, b, c, d, max_annual_lai, annual_lai_amp
        real(8) :: z, z1, z2, z3, p1n, p1d, p2n, p2d
        integer :: i
      
        ! Initialize variables
        max_annual_lai = 1
        annual_lai_amp = 0.5
        do i = 1, n
          t(i) = real(i - 1, 8)
        end do
      
        ! Initial parameter guesses
        a = -5.0
        b = 0.1
        c = maxval(lai)
        d = minval(lai)
      
        ! Calculate curvature rate of change
        do i = 1, size(t)
            z = exp(a + b * t(i))
            z1 = 1.0 - z
            z2 = 1.0 + z
            z3 = (b * c * z)**2
            p1n = 3.0 * z * z1 * z2**3 * (2.0 * z2**3 + b**2 * c**2 * z)
            p1d = (z2**4 + z3)**(2.5)
            p2n = z2**2 * (1.0 + 2.0 * z - 5.0 * z**2)
            p2d = (z2**4 + z3)**(1.5)
            kv(i) = b**3 * c * z * ((p1n / p1d) - (p2n / p2d))
          end do
      
        if (maxval(lai) > max_annual_lai * 0.3 .and. maxval(lai) - minval(lai) > 0.3 * annual_lai_amp) then
          sos_flag = 0
          do i = 2, n - 1
            if (kv(i) > kv(i - 1) .and. kv(i) > kv(i + 1)) then
                sos_flag = 1
                return
            end if
          end do
        else
            sos_flag = 0
        end if
      
      
    end subroutine get_sos
        
end program MLPhenology
