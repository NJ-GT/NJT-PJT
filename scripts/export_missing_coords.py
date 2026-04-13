# -*- coding: utf-8 -*-
import pandas as pd, os
BASE = os.path.join(os.path.dirname(__file__), '..')
path = os.path.join(BASE, '\uc18c\ubc29\uccad_\ud2b9\uc815\uc18c\ubc29\ub300\uc0c1\ubb3c\uc815\ubcf4\uc11c\ube44\uc2a4.csv')
df = pd.read_csv(path, encoding='utf-8-sig', skipinitialspace=True, dtype=str)
df.columns = df.columns.str.strip()
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
missing = df[df['X\uc88c\ud45c'].isin(['','nan']) | df['X\uc88c\ud45c'].isna()].copy()
print(f'미조회: {len(missing)}행')
out = os.path.join(BASE, 'data', '\uc18c\ubc29\uccad_\uc88c\ud45c\ubbf8\uc870\ud68c_87.csv')
missing.to_csv(out, index=False, encoding='utf-8-sig')
print(f'저장: {out}')
