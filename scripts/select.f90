PROGRAM selec

IMPLICIT NONE

! Set the kind type
INTEGER,PARAMETER ::  k=kind(0d0)

! Variables
REAL(k) :: dist,jk,bin
REAL(k) :: lat1, lon1, lat2, lon2, Rad
REAL(k) :: x1, y1, z1, x2, y2, z2
REAL(k) :: alpha,s,dprod
REAL(k) :: radius

REAL(k),ALLOCATABLE,DIMENSION(:,:) :: info
REAL(k),ALLOCATABLE,DIMENSION(:) :: infoj


INTEGER :: nf,hd,n,iostat,i,j

CHARACTER(len=300) :: infile,outfile,locfile
CHARACTER(len=300) :: jnk

CHARACTER(len=300),ALLOCATABLE,DIMENSION(:,:) :: infos
CHARACTER(len=300),ALLOCATABLE,DIMENSION(:) :: infosj



! Fixed Earth Radius
radius = 6371_k

! Requires the infile to have lon/lat in first column,hd is the number of headers
READ(*,'(a300)') infile
READ(*,*) nf,hd  ! Number of fields in the file past the initial lon/lat, Number of header lines

! Distance from points in locfile (lon/lat)
READ(*,*) dist
READ(*,'(a300)') locfile

! where to place the output
READ(*,'(a300)') outfile


n=0
OPEN(UNIT=111,FILE=infile)

ALLOCATE(infoj(2))
ALLOCATE(infosj(nf))

! Read headers
DO i=1,hd
   READ(111,*,IOSTAT=iostat) jnk
   IF (iostat /= 0) EXIT
END DO

DO
   READ(111,*,IOSTAT=iostat) (infoj(j),j=1,2),(infosj(j),j=1,nf)
   IF (iostat /= 0) EXIT
   n=n+1
END DO
CLOSE(111)

OPEN(UNIT=111,FILE=infile)
! Read headers
DO i=1,hd
   READ(111,*,IOSTAT=iostat) jnk
   IF (iostat /= 0) EXIT
END DO


ALLOCATE(info(1:n,1:nf))
ALLOCATE(infos(1:n,1:nf))

DO i=1,n
   READ(111,*) (info(i,j),j=1,2),(infos(i,j),j=1,nf)
!   WRITE(*,*)  (info(i,j),j=1,nf)
END DO
CLOSE(111)


OPEN(UNIT=111,FILE=locfile)
OPEN(UNIT=222,FILE=outfile,STATUS='REPLACE')

! Now loop through and find matching points for each point in point file
DO

   ! Read in the point location
   READ(111,*,IOSTAT=iostat) bin,lon2,lat2
   IF (iostat /= 0) EXIT

   ! Now see if anything matches it
   DO i = 1,n
      
      lon1 = info(i,1)
      lat1 = info(i,2)

      CALL cartesian(lat1,lon1,radius,x1,y1,z1)
      CALL cartesian(lat2,lon2,radius,x2,y2,z2)

      CALL dot(x1,y1,z1,x2,y2,z2,dprod,alpha,s)

      ! Only grab values that are within the selected distance
      IF (s <= dist) THEN
         ! TRIM removes leading/ending blanks, add one blank
         WRITE(222,*) (TRIM(infos(i,j))//" ",j=1,nf),NINT(bin),lon2,lat2
      END IF

   END DO

END DO





END PROGRAM



       !Latitude, Longitude to Cartesian Coordinates
        !***************************************************
        SUBROUTINE cartesian(Lat, Lon, Rad, x, y, z)
        ! Takes input Latitude, Longitude, and Radius
        ! Returns x, y, z coordinates
        IMPLICIT NONE
        INTEGER, PARAMETER :: k=kind(0d0)
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
        INTEGER, PARAMETER :: k=kind(0d0)
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
        
        !Cross Product for determining euler rotation pole
        !Angle between two points on a sphere and
        !Distance between two points on a sphere
        !***************************************************
        SUBROUTINE cross(v1, v2, v3, w1, w2, w3, u1, u2, u3, alpha, s)
        ! For vectors v and w, where
        ! v = (v1, v2, v3)
        ! w = (w1, w2, w3)
        ! the cross product v x w = x = (x1, x2, x3)
        ! output is u = (u1, u2, u3); where u is the unit
        ! vector in the direction of the vector x
        ! the angle between v and w is given as alpha;
        ! where alpha = arcsin(|v x w|/|v||w|)
        ! s is the distance along the arc between the two
        ! endpoints of vectors v and w, where s=r*theta
        ! alpha and s are only valid for angles <= 90 deg
        IMPLICIT NONE
        INTEGER, PARAMETER :: k=kind(0d0)
        REAL(k), PARAMETER :: pi=3.141592653589793_k
        REAL(k), INTENT(IN) :: v1, v2, v3, w1, w2, w3
        REAL(k), INTENT(OUT) :: u1, u2, u3, alpha, s
        REAL(k) :: magV, magW, magX, arg, rtd, theta
        REAL(k) :: x1, x2, x3

        rtd = 180_k/pi

        x1 = v2*w3 - v3*w2
        x2 = -v1*w3 + v3*w1
        x3 = v1*w2 - v2*w1

        magV = SQRT(v1**2 + v2**2 + v3**2)
        magW = SQRT(w1**2 + w2**2 + w3**2)
        magX = SQRT(x1**2 + x2**2 + x3**2)

        u1 = x1/magX; u2 = x2/magX; u3 = x3/magX

        arg = magX/(magV*magW)
        theta = ASIN(arg)
        s = magV*theta
        alpha = (ASIN(arg))*rtd 

        END SUBROUTINE cross
        
        !Dot Product between two vectors.  Extremely usefull for
        !determining the angle and distance between two points
        !on a shpere. 
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
        INTEGER, PARAMETER :: k=kind(0d0)
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

        !Euler Rotation Matrix
        !***************************************************    
        SUBROUTINE euler(xi,yi,zi,ex,ey,ez,alpha,xo,yo,zo)
        ! Input xi, yi, zi  are the cartesian coordinates
        ! of the point on the globe that you want to rotate.
        ! Input ex, ey, ez are the cartesian coordinates
        ! of the unit vector describing the euler rotation
        ! pole you wish to rotate around.
        ! Input alpha is the angle to rotate by (in deg)
        ! Output xo, yo, zo are the cartesean coordinates
        ! of the rotated point.
        ! rotations must be less than 90 deg
        IMPLICIT NONE
        INTEGER, PARAMETER :: k=kind(0d0)
        REAL(k), PARAMETER :: pi=3.141592653589793_k
        REAL(k), INTENT(IN) :: xi, yi, zi, ex, ey, ez, alpha
        REAL(k), INTENT(OUT) :: xo, yo, zo
        REAL(k), DIMENSION(3,3) :: R    
        REAL(k) :: dtr, angle

        dtr = pi/180_k
        angle = alpha*dtr
        
        R(1,1) = (ex**2)*(1 - COS(angle)) + COS(angle)
        R(1,2) = (ex*ey)*(1 - COS(angle)) - ez*SIN(angle)
        R(1,3) = (ex*ez)*(1 - COS(angle)) + ey*SIN(angle)
        R(2,1) = (ex*ey)*(1 - COS(angle)) + ez*SIN(angle)
        R(2,2) = (ey**2)*(1 - COS(angle)) + COS(angle)
        R(2,3) = (ey*ez)*(1 - COS(angle)) - ex*SIN(angle)
        R(3,1) = (ex*ez)*(1 - COS(angle)) - ey*SIN(angle)
        R(3,2) = (ey*ez)*(1 - COS(angle)) + ex*SIN(angle)
        R(3,3) = (ez**2)*(1 - COS(angle)) + COS(angle)

        xo = xi*R(1,1) + yi*R(1,2) + zi*R(1,3)
        yo = xi*R(2,1) + yi*R(2,2) + zi*R(2,3)
        zo = xi*R(3,1) + yi*R(3,2) + zi*R(3,3)
        
        END SUBROUTINE euler    
