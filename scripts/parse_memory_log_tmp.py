import re
import sys
from pathlib import Path

root = Path('/home/votrongnam/PROJECT/Large_scale_TO/debug_logs')
logs = sorted([p for p in root.glob('engine_bracket_memory_full_*.log') if not any(x in p.name for x in ['monitor','memory_only','delta'])], key=lambda p: p.stat().st_mtime, reverse=True)
if not logs:
    raise SystemExit('no log found')
log = logs[0]
mem = log.with_name(log.stem + '_memory_only.log')
summary = log.with_name(log.stem + '_memory_delta.tsv')
pat = re.compile(r'memory\[(.*?)\]:.*?mem_available_GB=([0-9.eE+-]+)')
rows = []
with log.open(errors='ignore') as f:
    for line in f:
        m = pat.search(line)
        if m:
            rows.append((m.group(1), float(m.group(2)), line.strip()))
with mem.open('w') as f:
    for _, _, line in rows:
        f.write(line + '\n')
with summary.open('w') as f:
    f.write('idx\ttag\tmem_available_GB\tdelta_from_prev_GB\tconsumed_from_start_GB\n')
    if rows:
        start = rows[0][1]
        prev = rows[0][1]
        for i, (tag, avail, _) in enumerate(rows):
            f.write(f'{i}\t{tag}\t{avail:.6f}\t{avail-prev:.6f}\t{start-avail:.6f}\n')
            prev = avail
print(f'LOG={log}')
print(f'MEMORY_ONLY={mem}')
print(f'SUMMARY={summary}')
print(f'ROWS={len(rows)}')
if rows:
    print(f'START={rows[0][0]} {rows[0][1]:.6f}')
    tag, val, _ = min(rows, key=lambda r: r[1])
    print(f'MIN={tag} {val:.6f}')
