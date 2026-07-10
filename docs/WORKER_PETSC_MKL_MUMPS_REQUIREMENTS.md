Worker update request for Large_scale_TO PETSc+MKL+MUMPS

Please make the worker match the master exactly:

1. User/path
   - user: votrongnam
   - project: /home/votrongnam/PROJECT/Large_scale_TO

2. MPICH
   - prefix: /home/votrongnam/opt/mpich-4.0-ch3-cxx
   - version: MPICH 4.0, CH3/nemesis, C++ enabled
   - check: /home/votrongnam/opt/mpich-4.0-ch3-cxx/bin/mpicxx -show

3. Intel MKL
   - path: /opt/intel/oneapi/mkl/2024.2
   - check: ls /opt/intel/oneapi/mkl/2024.2/lib

4. PETSc source
   - path: /home/votrongnam/src/petsc-3.19.6

5. Build new PETSc prefix
   - run from project root:
     bash scripts/build_petsc_mkl_mumps.sh
   - expected prefix:
     /home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl-mumps

6. Report back
   - output of:
     PKG_CONFIG_PATH=/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl-mumps/lib/pkgconfig pkg-config --modversion PETSc
     /home/votrongnam/opt/mpich-4.0-ch3-cxx/bin/mpiexec --version | head
   - confirm whether /opt/intel/oneapi/mkl/2024.2 exists.
