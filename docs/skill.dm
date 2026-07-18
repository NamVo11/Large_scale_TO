kill watcher đang chạy 
fuser -k 8765/tcp

full memory
dmesg -T | grep -Ei "out of memory|killed process|oom" | tail -50

kill all running

pkill -f 'stage2_dynamic_forward|mpiexec|hydra_pmi_proxy|hydra_nameserver' || true
sleep 2
pkill -9 -f 'stage2_dynamic_forward|mpiexec|hydra_pmi_proxy|hydra_nameserver' || true
pgrep -af 'stage2_dynamic_forward|mpiexec|hydra_pmi_proxy|hydra_nameserver' || true

timing_solve_wall        = toàn bộ ROM.main, có thể gồm factorization
info.timing_total        = toàn bộ ROM.main, có thể gồm factorization
info.timing_fom_solve    = chỉ KSPSolve
info.timing_fom_factorization = chỉ setup/factorization nếu có phát sinh