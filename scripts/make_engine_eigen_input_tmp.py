import json
from pathlib import Path
src = Path('case_study/Engine_bracket/mesh_size1.5/FOM/input.json')
dst = Path('debug_logs/engine_bracket_eigen_input_tmp.json')
data = json.loads(src.read_text())
data.setdefault('Run', {})['linear_backend'] = 'Eigen'
data['Run'].setdefault('mpi', {})['np'] = 1
dst.write_text(json.dumps(data, indent=2))
print(dst)
print('linear_backend=', data['Run'].get('linear_backend'))
