#!/usr/bin/env python3
"""Lightweight structural check of the reduced nodal data and accompanying experiments."""
from pathlib import Path
import csv

root = Path(__file__).resolve().parents[1]
for group in ('calibration', 'validation'):
    entries=list((root/'data'/group).rglob('RESULTS_*.csv'))
    experiments=list((root/'data'/group).rglob('*_R_*.txt'))
    assert entries and experiments, f'Empty {group} data'
    for p in entries:
        with p.open(encoding='utf-8', newline='') as f:
            rows=list(csv.reader(f))
        assert len(rows)==1 and len(rows[0])>=20, (p, len(rows))
    print(f'{group}: {len(entries)} FE hotspot rows and {len(experiments)} experimental files')
print('Dataset checks passed.')
