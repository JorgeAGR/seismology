PROGRAM vespas

! 6/22/2010
! Code by Nick Schmerr
! Makes vespagrams

!USE IFPORT        ! <-- For Cluster
!USE f90_unix_dir ! <-- For NagWare f95
USE sac_i_o  ! <-- For reading in sacfiles

IMPLICIT NONE


!==============================================================================
! Variables
  INTEGER, PARAMETER :: kd=8
  REAL(KIND=kd), PARAMETER :: Re=6371.028
  REAL(KIND=kd), PARAMETER :: pi=3.141592653589793_kd 

  ! Slowness information
  REAL(KIND=kd) :: slw1,slw2,slwi,sphcor


  ! File names
  CHARACTER(len=300),ALLOCATABLE,DIMENSION(:) :: datafiles
  CHARACTER(len=6),ALLOCATABLE,DIMENSION(:) :: datanames
  CHARACTER(len=300) :: datafile,datadir,dataname,vespadir
 
  INTEGER :: n,i,j,iostat,ns

  


  ! Variable for rbsac
  REAL(KIND=k),DIMENSION(:),ALLOCATABLE :: yarray1             ! Yarray from rbsac, note this is in single precision
  CHARACTER(len=300) :: infile1
  REAL(k), DIMENSION(:,:), ALLOCATABLE :: seismos
  REAL(kd), DIMENSION(:,:), ALLOCATABLE :: seismodata

  REAL(k) :: root

!  REAL(KIND=kd),ALLOCATABLE,DIMENSION(:,:,:,:) :: datagrid
!  CHARACTER(len=6),ALLOCATABLE,DIMENSION(:,:,:) :: namegrid

  INTEGER :: inpts
  REAL(KIND=kd) :: sample  ! Sample Rate

  CHARACTER(len=6) :: stnm
  REAL(KIND=kd) :: stlat,stlon,evlat,evlon,evdep

  ! For finding average location
  REAL(KIND=kd) :: x1,y1,z1,x2,y2,z2,rad,dprod,alpha,s 

  REAL(KIND=kd) :: dist_ref


  INTEGER :: nslow, N1
  REAL(kd), DIMENSION(:,:), ALLOCATABLE :: vespa
  REAL(KIND=kd) :: slow,toff,dist_st

!!---------------------------------------------------
  REAL(kd), DIMENSION(:), ALLOCATABLE :: randmat             ! Random array
  REAL(kd), DIMENSION(:,:,:), ALLOCATABLE :: boot            ! Bootstrap array
  REAL(kd) :: mean,stdev
  INTEGER :: bs,nboots,r
!!---------------------------------------------------


!==============================================================================

!==============================================================================
! Read in input

! Data location
READ(*,101) datadir
READ(*,101) datafile

! Number of pts in each seismogram
READ(*,*) inpts

! Slowness window to use
READ(*,*) slw1,slw2,slwi,root

101 FORMAT(a300)


!==============================================================================
!==============================================================================
! Read in the seismic data
! All data should be preprocessed in sac (filtering, resampling, etc.) and
! should be the same length and sample rate. 

! Seismogram key:
! For arrays:
! datafiles  (characters)
! seismos    (reals)
! seismodata (reals)

! {i} [1:ns] number of seismograms
! {j} [1:npts] number of points in each seismogram 

! dataname       = datafiles(i)   ! name of the seismogram
! yarray(1:npts) = seismos(i,j)   ! amplitudes for each seismo


WRITE(*,*) "Reading in the seismic data..."
!-----------------------------------------------------------------------
! Get the number of seismograms
n=0
OPEN(UNIT=777,FILE=datafile)
DO
   READ(777,*,IOSTAT=iostat) dataname,stnm
   IF (iostat /= 0) EXIT

   n=n+1
END DO
CLOSE(777)

! Number of seismograms
ns = n
!-----------------------------------------------------------------------

!-----------------------------------------------------------------------
! Fill in the data arrays

! Names of the seismos
ALLOCATE(datafiles(1:ns))
ALLOCATE(datanames(1:ns))

! Array for storing the seismograms
ALLOCATE(seismos(1:ns,0:inpts))

! Array for storing info about seismos
! Purposefully left big to accomodate extra data as needed
ALLOCATE(seismodata(1:ns,1:20))

!Read in the list of files ss_ss and ss_rs
OPEN(UNIT=777,FILE=datafile)

DO i=1,ns

   READ(777,*) dataname,stnm

!   ! This matches the indices of the seismos to the index from the precalculated 
!   ! migration grid, taking care of stations at the same site
!   DO j=1,ng3
!      IF (stnm == namegrid(1,1,j,1) ) THEN
!         seismodata(i,1) = REAL(j)
!         !WRITE(*,*) i,j,TRIM(stnm)
!         EXIT
!      END IF
!   END DO

   datafiles(i) = dataname
   datanames(i) = stnm

END DO
CLOSE(777)

! Get current directory
CALL GETCWD(vespadir)
!istat = GETCWD(vespadir)


! Go to data directory
CALL CHDIR(TRIM(datadir))
!istat = CHDIR(TRIM(datadir))

! Read in the data
DO i=1,ns

   infile1=datafiles(i)

   !-------------------------------------
   ! Read in the sac data     
   CALL rbsac(infile1,delta,depmin,depmax,scale,odelta,b,e,o,a,internal1,            &
        t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,f,resp0,resp1,resp2,resp3,resp4,resp5,resp6,   &
        resp7,resp8,resp9,stla,stlo,stel,stdp,evla,evlo,evel,evdp,mag,user0,user1,   &
        user2,user3,user4,user5,user6,user7,user8,user9,dist,az,baz,gcarc,internal2, &
        internal3,depmen,cmpaz,cmpinc,xminimum,xmaximum,yminimum,ymaximum,unused1,   &
        unused2,unused3,unused4,unused5,unused6,unused7,nzyear,nzjday,nzhour,nzmin,  &
        nzsec,nzmsec,nvhdr,norid,nevid,npts,internal4,nwfid,nxsize,nysize,unused8,   &
        iftype,idep,iztype,unused9,iinst,istreg,ievreg,ievtyp,iqual,isynth,imagtyp,  &
        imagsrc,unused10,unused11,unused12,unused13,unused14,unused15,unused16,      &
        unused17,leven,lpspol,lovrok,lcalda,unused18,kevnm,kstnm,khole,ko,ka,kt0,kt1,&
        kt2,kt3,kt4,kt5,kt6,kt7,kt8,kt9,kf,kuser0,kuser1,kuser2,kcmpnm,knetwk,kdatrd,&
        kinst,yarray1)
   !-------------------------------------

   IF (inpts == npts) THEN
      seismos(i,1:npts) = yarray1(1:npts)
      sample = delta
   ELSE
      WRITE(*,*) "ERROR: Something wrong with the number of pts in this seismogram:", TRIM(infile1)
   END IF

   seismodata(i,2)  = REAL(delta) 
   seismodata(i,3)  = REAL(npts)
   seismodata(i,4)  = REAL(b)
   seismodata(i,5)  = REAL(e)
   seismodata(i,6)  = stla
   seismodata(i,7)  = stlo
   seismodata(i,8)  = evla
   seismodata(i,9)  = evlo
   seismodata(i,10) = evdp
   seismodata(i,11) = gcarc

END DO

! Go to back to working directory
CALL CHDIR(TRIM(vespadir))
!istat = CHDIR(TRIM(vespadir))

!-----------------------------------------------------------------------
!==============================================================================

WRITE(*,*) "Finding the center of the array..."
!==============================================================================
!-----------------------------------------------------------------------
! Find the center of the array (delta_ref)

! Initialize
n = 0
x2=0.0_k;y2=0.0_k;z2=0.0_k
dist_ref = 0.0

DO i=1,ns

   stlat = seismodata(i,6)
   stlon = seismodata(i,7)

   evlat = seismodata(i,8)
   evlon = seismodata(i,9)
   evdep = seismodata(i,10)

   CALL cartesian(stlat,stlon,Re,x1,y1,z1)
   x2=x2+x1;y2=y2+y1;z2=z2+z1
   n = n+1
END DO

x2=x2/n;y2=y2/n;z2=z2/n

! Event location
CALL cartesian(evlat,evlon,Re,x1,y1,z1)
! Distance 
CALL dot(x1,y1,z1,x2,y2,z2,dprod,alpha,s)
! Center of array
CALL latlon(x2,y2,z2,stlat,stlon,rad)

dist_ref = alpha
!-----------------------------------------------------------------------
!==============================================================================

WRITE(*,*) "Creating the vespagram..."
!==============================================================================
!-----------------------------------------------------------------------
! Stack on each slowness

nslow = NINT((slw2-slw1)/slwi)+1

! Vespas array
ALLOCATE(vespa(1:nslow,1:inpts))

vespa = 0_kd


OPEN(UNIT=111,FILE="boot_vespa.txt",STATUS='REPLACE')

WRITE(111,*) ">",dist_ref,stlat,stlon,evlat,evlon,evdep



!!---------------------
! Initialize arrays
ALLOCATE(randmat(ns))
randmat = 0.0_kd
nboots = 300


! Array with number of bootstraps x number of slownesses x number of time samples
ALLOCATE(boot(1:nboots,1:nslow,1:inpts))
boot=0_kd

DO bs=1,nboots
   
   WRITE(*,*) "We are on boot number ",bs
   
   ! Fill with random values (0-1)
   CALL RANDOM_NUMBER(randmat)

   ! Scale to number of records
   randmat = REAL(INT(randmat*ns)+1) 
   !!---------------------

   vespa=0.0_kd
   DO i=1,nslow

      slow = slw1+REAL(i-1)*slwi

      ! Stack seismos for this slowness
      DO j=1,ns

         !!---------------------
         ! Assign the record # from the random matrix
!         r=j !randmat(j)
         r=randmat(j)
         !!---------------------

         !!---------------------
         dist_st = seismodata(r,11)
         !!---------------------


         ! Spherical Earth correction:
         !sphcor = (180_kd/pi)*(1_kd-COS((dist_ref-dist_st)*pi/180_kd))
         sphcor = ABS((180_kd/pi)*(TAN((dist_ref-dist_st)*pi/180_kd))-(dist_ref-dist_st))
         
         !      toff=slow*(dist_ref-dist_st) ! Uncorrected version
         !!toff=slow*(dist_ref-dist_st-sphcor)
         toff=slow*((dist_ref-dist_st)+sphcor)


         !      toff=slow*(dist_st-dist_ref)
         N1 = NINT(toff/sample)

         !      ! Offset the seismogram and stack
         !      IF ( (N1 < 0) ) THEN
         !         vespa(i,1:(inpts-ABS(N1))) =vespa(i,1:(inpts-ABS(N1))) +  seismos(j,ABS(N1+1):inpts)
         !      ELSE IF ( (N1 > 0) ) THEN
         !         vespa(i,(N1+1):inpts) = vespa(i,(N1+1):inpts) + seismos(j,1:(inpts-N1))
         !      ELSE IF ( (N1 == 0) ) THEN
         !         vespa(i,1:inpts) = vespa(i,1:inpts) + seismos(j,1:inpts)
         !      END IF


         ! Nth root vespas:
         ! Offset the seismogram and stack
         IF ( (N1 < 0) ) THEN

            !!---------------------
            vespa(i,1:(inpts-ABS(N1))) = vespa(i,1:(inpts-ABS(N1))) +  SIGN((ABS(seismos(r,ABS(N1+1):inpts))**(1./root)),&
              (seismos(r,ABS(N1+1):inpts)))
         ELSE IF ( (N1 > 0) ) THEN
            vespa(i,(N1+1):inpts) = vespa(i,(N1+1):inpts) + SIGN((ABS(seismos(r,1:(inpts-N1)))**(1./root)),&
              (seismos(r,1:(inpts-N1))))
         ELSE IF ( (N1 == 0) ) THEN
            vespa(i,1:inpts) = vespa(i,1:inpts) + SIGN((ABS(seismos(r,1:inpts))**(1./root)),&
              (seismos(r,1:inpts)))
         END IF
         !!---------------------



      END DO

      !!---------------------
      !  vespa(i,1:inpts) = SIGN((ABS(vespa(i,1:inpts))**(root)),vespa(i,1:inpts))
      boot(bs,i,1:inpts) = SIGN((ABS(vespa(i,1:inpts))**(root)),vespa(i,1:inpts))
      !!---------------------


      !!---------------------
      !!   ! Write the output:
!      IF (i==1 .AND. bs==1) THEN
!        OPEN(UNIT=222,FILE="jnk.txt")
!         DO j=1,inpts
!            WRITE(222,*) (j-1)*sample,vespa(i,j),slow
!         END DO
!  
!      END IF
      !!   WRITE(111,*) ">"
      !!   DO j=1,inpts
      !!      WRITE(111,*) (j-1)*sample,vespa(i,j),slow
      !!   END DO
      !!---------------------

   END DO

!!---------------------
END DO ! bootstrap

! To compute the bootstrap for this vespagram
DO i =1,nslow

   slow = slw1+REAL(i-1)*slwi

   DO j =1,inpts

      ! Find the average for each point in time and slowness
      mean=0.0
      DO bs=1,nboots
         mean=mean+boot(bs,i,j)
         
      END DO

!      WRITE(*,*) i,j,mean,mean/REAL(nboots)
      mean=mean/REAL(nboots)

      stdev=0.0
      DO bs=1,nboots
        stdev=stdev + ( boot(bs,i,j) - mean )**2
      END DO
      stdev = 1.96 * SQRT( 1.0/(REAL(nboots)-1.0) * stdev)

      WRITE(111,*) (j-1)*sample,mean,slow,stdev


   END DO
END DO
!!---------------------








!-----------------------------------------------------------------------
!==============================================================================


!==============================================================================
END PROGRAM












!Latitude, Longitude to Cartesian Coordinates
!***************************************************
SUBROUTINE cartesian(Lat, Lon, Rad, x, y, z)
  ! Takes input Latitude, Longitude, and Radius
  ! Returns x, y, z coordinates
  IMPLICIT NONE
!  INTEGER, PARAMETER :: k=kind(0d0)
  INTEGER, PARAMETER :: k=8
  REAL(k), PARAMETER :: pi=3.141592653589793_k 
  REAL(k), INTENT(IN) :: Lat, Lon, Rad
  REAL(k), INTENT(OUT) :: x, y, z
  REAL(k) :: rho, phi, theta, dtr
  
  dtr = pi/180_k
  
  rho = Rad; theta = Lon*dtr
  Phi = (90_k - Lat)*dtr
  
  x = rho*SIN(Phi)*COS(theta)
  y = rho*SIN(Phi)*SIN(theta)
  z = rho*COS(Phi)
  
END SUBROUTINE cartesian

!Cartesian Coordinates to Latitude and Longitude
!***************************************************
SUBROUTINE latlon(x, y, z, Lat, Lon, Rad)
  IMPLICIT NONE
  ! For input cartesian coordinates x, y, and z
  ! this subroutine converts to that points
  ! Latitude and Longitude, and also gives the
  ! Radius of the sphere where the point is
  ! defined
!  INTEGER, PARAMETER :: k=kind(0d0)
  INTEGER, PARAMETER :: k=8
  REAL(k), PARAMETER :: pi=3.141592653589793_k
  REAL(k), INTENT(IN) :: x, y, z
  REAL(k), INTENT(OUT) :: Lat, Lon, Rad
  REAL(k) :: arg, rtd, phi
  
  rtd = 180_k/pi
  Rad = SQRT(x**2 + y**2 + z**2)
  Lon = (ATAN2(y,x))*rtd
  arg = z/Rad
  phi = (ACOS(arg))*rtd
  Lat = 90_k - phi 
  
END SUBROUTINE latlon

!Dot Product between two vectors.  Extremely usefull for
!determining the angle and distance between two points
!on a shpere. 
!***************************************************
SUBROUTINE dot(v1, v2, v3, w1, w2, w3, dprod, alpha, s)
  ! For vectors v and w, where
  ! v = (v1, v2, v3)
  ! w = (w1, w2, w3)
  ! the dot product output is the scalar variable dprod
  ! the angle between the vectors is given as alpha, where
  ! alpha = arccos(v*w/|v||w|)
  ! s is the distance along the arc between the two endpoints
  ! of vectors v and w, where s=r*theta.  alpha and s are
  ! valid for angles <= 180 deg
  IMPLICIT NONE
!  INTEGER, PARAMETER :: k=kind(0d0)
  INTEGER, PARAMETER :: k=8
  REAL(k), PARAMETER :: pi=3.141592653589793_k
  REAL(k), INTENT(IN) :: v1, v2, v3, w1, w2, w3
  REAL(k), INTENT(OUT) :: dprod, alpha, s
  REAL(k) :: rtd, magW, magV, arg, theta
  
  rtd = 180_k/pi
  
  dprod = v1*w1 + v2*w2 + v3*w3
  
  magV = SQRT(v1**2 + v2**2 + v3**2)
  magW = SQRT(w1**2 + w2**2 + w3**2)
        
  arg = dprod/(magV*magW)
  theta = ACOS(arg)
  s = magV*theta
  alpha = (ACOS(arg))*rtd
END SUBROUTINE dot
