from pathlib import Path
old = Path('/home/votrongnam/PROJECT/Large_scale_TO/debug_logs/engine_bracket_memory_full_20260808_184856_memory_delta.tsv')
new = Path('/home/votrongnam/PROJECT/Large_scale_TO/debug_logs/engine_bracket_memory_full_20260809_171141_memory_delta.tsv')
keys = [
'iter_begin','forward_after_assemble_K_M','forward_after_prepare_Keff','forward_after_dirichlet_K_initial',
'fom_direct_after_KSPSetUp','forward_after_initial_solve_store_factor','forward_after_history_alloc','forward_after_step_100',
'forward_after_store_matrices','forward_after_destroy_temp','after_forward','after_objective_value','after_constraint_value',
'before_objective_sensitivity','petsc_sens_after_duplicate_K0','petsc_sens_after_duplicate_Keff','petsc_sens_after_get_K0','petsc_sens_after_get_Keff','petsc_sens_after_vec_create','petsc_sens_after_lambda_all_alloc'
]
def read(path):
    out=[]
    for line in path.read_text().splitlines()[1:]:
        idx, tag, avail, delta, consumed = line.split('\t')
        out.append((int(idx), tag, float(avail), float(delta), float(consumed)))
    return out
def first_by_tag(rows, tag):
    for r in rows:
        if r[1] == tag:
            return r
    return None
def all_by_tag(rows, tag):
    return [r for r in rows if r[1] == tag]
oldr=read(old); newr=read(new)
print('old_start', oldr[0][2], 'old_min_consumed', max(r[4] for r in oldr))
print('new_start', newr[0][2], 'new_min_consumed', max(r[4] for r in newr))
print('tag\told_consumed\tnew_consumed\tdiff_new_minus_old\told_avail\tnew_avail')
for k in keys:
    ro=first_by_tag(oldr,k); rn=first_by_tag(newr,k)
    if ro or rn:
        oc = ro[4] if ro else None; nc = rn[4] if rn else None
        oa = ro[2] if ro else None; na = rn[2] if rn else None
        diff = (nc-oc) if (ro and rn) else None
        print(f'{k}\t{oc if oc is not None else "NA"}\t{nc if nc is not None else "NA"}\t{diff if diff is not None else "NA"}\t{oa if oa is not None else "NA"}\t{na if na is not None else "NA"}')
print('\nKSPSetUp occurrences consumed:')
for i,(ro,rn) in enumerate(zip(all_by_tag(oldr,'fom_direct_after_KSPSetUp')[:6], all_by_tag(newr,'fom_direct_after_KSPSetUp')[:6])):
    print(i, 'old', ro[4], 'new', rn[4], 'diff', rn[4]-ro[4])
