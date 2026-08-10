import csv, math
from pathlib import Path
root=Path('/home/votrongnam/PROJECT/Large_scale_TO/case_study/Engine_bracket/mesh_size1.5/strategy_10_20/result')
fields=['accepted','fom_used','rom_used','rank','residual','timing_total','timing_petsc_to_eigen_mat','timing_petsc_to_eigen_vec','timing_projection','timing_reduced_rhs','timing_reduced_solve','timing_reconstruct','timing_residual','timing_fom_fallback','timing_fom_factorization','timing_fom_solve','timing_online_basis_update','timing_end_basis_update','timing_basis_update','timing_snapshot']

def read(name):
    p=root/name
    if not p.exists(): return []
    rows=[]
    with p.open() as f:
        for row in csv.reader(f):
            vals=[]
            for x in row:
                x=x.strip()
                if not x: continue
                try: vals.append(float(x))
                except: pass
            if vals: rows.append(vals)
    return rows

def flat(rows):
    return [x for r in rows for x in r]

def summarize(prefix):
    print('\n===',prefix,'===')
    data={f:flat(read(f'{prefix}_{f}.csv')) for f in fields}
    n=max((len(v) for v in data.values()), default=0)
    print('n_entries',n)
    if n==0: return
    for flag in ['rom_used','accepted','fom_used']:
        v=data.get(flag,[])
        print(flag,'count_nonzero',sum(1 for x in v if abs(x)>0.5),'/',len(v))
    rank=data.get('rank',[])
    if rank:
        print('rank min/max/mean',min(rank),max(rank),sum(rank)/len(rank))
    for f in fields:
        if not f.startswith('timing_'): continue
        v=[x for x in data.get(f,[]) if math.isfinite(x)]
        if not v: continue
        nz=[x for x in v if abs(x)>1e-12]
        print(f, 'sum',sum(v),'mean_all',sum(v)/len(v),'mean_nz', (sum(nz)/len(nz) if nz else 0), 'max', max(v), 'nz', len(nz))
    # accepted ROM only timing means
    acc=data.get('accepted',[]); fom=data.get('fom_used',[])
    for subset_name, pred in [('accepted_rom', lambda i: i<len(acc) and acc[i]>0.5),('fallback',lambda i: i<len(fom) and fom[i]>0.5)]:
        ids=[i for i in range(n) if pred(i)]
        print('subset',subset_name,'n',len(ids))
        for f in ['timing_total','timing_projection','timing_reduced_rhs','timing_reduced_solve','timing_reconstruct','timing_residual','timing_petsc_to_eigen_mat','timing_petsc_to_eigen_vec','timing_fom_solve','timing_fom_factorization','timing_online_basis_update','timing_end_basis_update']:
            v=data.get(f,[])
            vals=[v[i] for i in ids if i<len(v)]
            if vals:
                print(' ',f,'mean',sum(vals)/len(vals),'max',max(vals),'sum',sum(vals))

summarize('forward')
summarize('obj_sens')
