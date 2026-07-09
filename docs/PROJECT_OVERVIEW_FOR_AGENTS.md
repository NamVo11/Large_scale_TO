# Large_scale_TO - Project Overview For Agents

File nay dung de handoff cho agent moi. Khi bat dau lam viec trong repo nay, doc file nay truoc, sau do doc them docs/PORTING_RULES.md va cac file lien quan trong scripts, .vscode, cpp/src, cpp/include neu nhiem vu yeu cau.

## 1. Muc tieu du an

Large_scale_TO la repo C++/PETSc/MPI tach roi khoi project MATLAB goc. Muc tieu la port flow dynamic topology optimization sang backend co the chay mo hinh lon hon.

Trong pham vi hien tai, bo qua Stage 1. Tap trung vao Stage 2:

- dynamic forward linear 2D/3D
- FOM va ROM
- optimization loop
- sensitivity/objective/constraint theo flow MATLAB
- MMA
- plotting, live density viewer, MATLAB comparison
- local single-node va MPI multinode run

MATLAB goc thuong nam tai:

    D:\VoTrongNam\Local-E\My_TOptimization\TO_main

Repo C++/PETSc hien tai trong WSL:

    /home/votrongnam/PROJECT/Large_scale_TO

## 2. Nguyen tac porting

Cac rule chinh cung nam trong docs/PORTING_RULES.md.

- Uu tien large scale truoc convenience.
- Giu data split, ten struct, ten bien, ten ham, class, derived class va flow gan MATLAB nhat co the.
- Objective, constraint, sensitivity, AVM, Debug phai giu thanh cac flow-level function rieng de de so sanh voi MATLAB.
- Solver backend phai nam sau interface PETSc/FOM/ROM. Khong gan cung LU, CG, GAMG, MKL, MUMPS vao thuat toan optimization.
- Chi port code path can cho dynamic linear 2D/3D, FOM, MyStrategy, va optimization hien tai, tru khi user yeu cau mo rong.
- Khong sua source MATLAB goc. Neu can logic tu MATLAB thi port/copy vao repo nay.
- Khong revert thay doi khong do minh tao. Repo co the dang dirty vi user/agent khac dang chinh.

## 3. Cau truc thu muc chinh

    cpp/include/stage2/        Struct/data/function flow giong MATLAB cho dynamic forward + optimization
    cpp/src/stage2/            Source Stage 2, entry hien tai la main_stage2.cpp
    cpp/include/ROM/           ROM classes theo cau truc gan MATLAB src/ROM
    cpp/src/ROM/               ROM implementation
    cpp/third_party/mma/       MMA C++ cu
    cpp/third_party/mma_matlab_port/  MMA port gan MATLAB mmasub/subsolv
    cpp/third_party/mma_jdumas/       MMA tu jdumas/mma wrapper/adaptation
    scripts/                   Build/run/sync/watch/plot scripts
    .vscode/                   Tasks/launch/settings cho VS Code WSL
    results/                   Output chay C++
    docs/                      Tai lieu handoff/setup/rules

Executable chinh hien tai:

    stage2_dynamic_forward     Stage 2: forward, optimization, FOM/ROM mode

## 4. Build variants hien co

Cac build directory thuong gap:

    build-mpich-debug          Debug, PETSc-only MPICH
    build-mpich-release        Release, PETSc-only MPICH
    build-mpich-mkl-release    Release, PETSc + MKL/PARDISO

Stack thuong dung:

    MPICH:      /home/votrongnam/opt/mpich-4.0-ch3-cxx
    PETSc only: /home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx
    PETSc MKL:  /home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl
    MKL:        /opt/intel/oneapi/mkl/2024.2

Luu y:

- Khong dung /usr/bin/mpirun, /usr/bin/mpicxx, hoac PETSc cai bang apt cho repo nay.
- Khi chay VS Code, mo folder tu WSL bang code ., khong mo qua Windows mirror.
- User chinh nen la votrongnam, khong phai root.

## 5. Build Stage 2

CMakeLists.txt build executable stage2_dynamic_forward, link PETSc, MPI CXX, OpenMP, Stage2 source, ROM source, va cac MMA variants.

Khi doi PETSc variant, thuong doi PKG_CONFIG_PATH luc configure CMake, khong sua thuat toan.

Vi du configure PETSc+MKL release:

    cd /home/votrongnam/PROJECT/Large_scale_TO
    export PKG_CONFIG_PATH=/home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl/lib/pkgconfig
    cmake -S . -B build-mpich-mkl-release -DCMAKE_BUILD_TYPE=Release
    cmake --build build-mpich-mkl-release --target stage2_dynamic_forward -j 8

## 6. Solver policy hien tai

Local direct solver tren 1 node:

    NP=1
    KSP_TYPE=preonly
    PC_TYPE=lu
    PC_FACTOR_MAT_SOLVER_TYPE=mkl_pardiso
    OMP_NUM_THREADS=8
    MKL_NUM_THREADS=8

Muc dich: direct LU nhanh tren nhieu core cua mot may bang OpenMP/MKL.

Khong nen dung MKL/PARDISO LU voi nhieu MPI ranks cho cung mot ma tran distributed. PARDISO o day khong phai distributed LU.

Multinode iterative solver:

    NP=8 hoac hon
    KSP_TYPE=cg
    PC_TYPE=gamg
    OMP_NUM_THREADS=1
    MKL_NUM_THREADS=1

Co the chay bang PETSc-only hoac PETSc+MKL neu master/worker khop version va thu vien runtime. Khi chay iterative multinode, MKL chu yeu la BLAS/LAPACK backend, khong phai distributed direct solver.

Distributed direct solver:

Neu muon direct solver that su tren nhieu node, can build PETSc them MUMPS/SuperLU_DIST/STRUMPACK. Khuyen nghi thu MUMPS truoc. Nen build thanh prefix moi, vi du:

    /home/votrongnam/opt/petsc-3.19.6-mpich-ch3-cxx-mkl-mumps

Khong overwrite build PETSc dang chay on neu chua can.

## 7. Scripts quan trong

    scripts/mpiexec_mpich_ch3_cxx.sh

Launcher MPICH local. Script nay set port range va tranh loi hwloc/WSL bang HWLOC_COMPONENTS=-gl.

    scripts/run_stage2_case.sh

Script chay Stage 2 case bang environment variables. Hien dang thien ve PETSc+MKL direct local, mac dinh build dir la build-mpich-mkl-release.

    scripts/run_stage2_multinode.sh

Script chay multinode qua ssh worker. Can chu y vi co the con hardcode PETSc-only/build-mpich-release o mot so cho. Khi chuan hoa task, can tranh lech giua local va worker.

    scripts/watch_density_live.py

Live viewer density. Viewer nen doc CSV/latest density de khop MATLAB plot, khong chi dua vao PGM neu co sai khac.

    scripts/plot_density_history.py

Plot density history offline tu output C++.

## 8. VS Code tasks

Cac file lien quan:

    .vscode/tasks.json
    .vscode/task_111.json
    .vscode/launch.json
    .vscode/settings.json

task_111.json la file user tach ra de chuan hoa task theo nhom:

- configure/build/run
- debug/release
- PETSc-only hoac PETSc+MKL
- local 1 node hoac multinode

Huong chuan hoa dang can nhac:

- Gop configure + build neu task dung hang ngay can one-click.
- Dung mot kieu truyen tham so chinh: env vars + script run_stage2_case.sh.
- Tach model config khoi solver config:
  - Model/case file: LX, LY, NELX, NELY, NSTEP, MAX_ITER, load, volume, move...
  - Task env: solver, rank, hosts, output label, build dir.
- Mot script chay nen tu quyet dinh local/multinode dua vao HOSTS hoac mot bien ro rang, thay vi tao qua nhieu task trung nhau.

## 9. Output va so sanh MATLAB

Output C++ thuong nam trong:

    results/<LABEL>

Cac file quan trong thuong co:

    optimization_history.csv
    rank_history.csv hoac ROM/rank-related history
    residual history
    mode history
    results/<LABEL>/density/...

MATLAB comparison thuong dung scripts trong MATLAB repo, vi du:

    scripts/compare_density_mat_csv.m
    scripts/plot_Nam1111_density_compare.m
    post_processing/...

Khi ket luan C++ khac MATLAB, kiem tra truoc:

- C++ CSV xPhys dung iteration chua?
- MATLAB .mat dung iteration chua?
- Live watcher dang nhin dung results/<LABEL> chua?
- Colormap/display co giong MATLAB khong?
- Task dang chay dung label/output folder chua?

Gan day user xac nhan khi plot bang MATLAB script tu C++ CSV va MATLAB .mat, FOM tai iteration 200 co the match, trong khi watcher cu lam hieu nham la sai.

## 10. ROM status

ROM dang duoc chuan hoa lai de giong MATLAB src/ROM hon.

Cau truc C++ hien co:

    cpp/include/ROM/Error_Estimator/
    cpp/include/ROM/Reduced_Basis_Agorithm/
    cpp/include/ROM/ROM_method/
    cpp/include/ROM/Utils/
    cpp/src/ROM/Error_Estimator/
    cpp/src/ROM/Reduced_Basis_Agorithm/
    cpp/src/ROM/ROM_method/
    cpp/src/ROM/Utils/

Classes/methods quan trong hien co:

    RelativeResidualEstimator
    UpdateBase
    MyPOD_SVD1
    IncrementalSVD
    ROM_Base
    FOM_direct
    FOM_iterative
    MyStrategy
    ROM_factory

Yeu cau user da nhan manh:

- ROM phai uu tien large scale.
- ROM phai giong MATLAB cang nhieu cang tot: ten class, ten bien, cach chia thu muc, derived class, flow FOM/ROM mode.
- Khi kiem tra ROM, khong chi nhin anh density; can kiem tra FOM/ROM mode, residual, rank, objective/constraint/sensitivity timing.

## 11. MMA status

Hien co nhieu MMA variants:

    cpp/third_party/mma/              MMA C++ cu
    cpp/third_party/mma_matlab_port/  Port gan MATLAB mmasub/subsolv
    cpp/third_party/mma_jdumas/       Solver dua tren jdumas/mma

Khi debug sai khac MATLAB, uu tien dung ban mma_matlab_port de giam sai khac voi MATLAB.

Khi so sanh ket qua optimization, phai kiem tra dong thoi:

- beta continuation
- MMA move
- filter/projection
- objective scaling
- constraint sign/scaling
- sensitivity AVM/debug mode
- solver tolerance
- density output iteration

## 12. Multinode notes

Worker tung dung IP:

    192.168.11.21

Launcher dung ssh wrapper:

    scripts/ssh_worker_2222.sh

Cac loi tung gap:

- Master/worker lech binary hoac lech step/case config lam MPI dung.
- Lech PETSc/MPICH/MKL library path giua 2 may.
- VS Code permission loi khi build dir duoc tao boi root.
- run_stage2_multinode.sh sync binary nhung can chac worker co cung thu vien runtime va duong dan repo tuong ung.
- Voi PETSc+MKL multinode iterative, worker cung can PETSc+MKL va MKL runtime tuong ung neu binary link toi ban do.

## 13. Can than truoc khi sua

Truoc khi chinh file:

    cd /home/votrongnam/PROJECT/Large_scale_TO
    git status --short

Neu co file dirty khong do minh sua, khong revert. Doc va lam viec cung thay doi do.

Khong tu y xoa build/results lon neu user chua yeu cau. Neu can cleanup, hoi ro.

## 14. Checklist nhanh cho agent moi

1. Doc file nay.
2. Doc docs/PORTING_RULES.md.
3. Xac dinh nhiem vu thuoc nhom nao: build/setup, task/script, Stage2 algorithm, ROM, MMA, plotting, multinode.
4. Kiem tra git status --short.
5. Neu lien quan VS Code/task, doc .vscode/tasks.json, .vscode/task_111.json, .vscode/launch.json.
6. Neu lien quan run, doc scripts/run_stage2_case.sh, scripts/run_stage2_multinode.sh, scripts/mpiexec_mpich_ch3_cxx.sh.
7. Neu lien quan FOM/optimization, doc cpp/src/stage2/main_stage2.cpp, Optimization.cpp, DynamicElasticLinear.cpp, MatlabLikeData.cpp.
8. Neu lien quan ROM, doc cpp/include/ROM/ va cpp/src/ROM/, sau do doi chieu MATLAB src/ROM.
9. Chay test nho truoc khi ket luan.
10. Khi bao ket qua cho user, noi ro path/file/task/output label da dung.
