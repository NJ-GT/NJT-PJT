# -*- coding: utf-8 -*-
import pandas as pd, os
BASE = os.path.join(os.path.dirname(__file__), '..')
path = os.path.join(BASE, '\uc18c\ubc29\uccad_\ud2b9\uc815\uc18c\ubc29\ub300\uc0c1\ubb3c\uc815\ubcf4\uc11c\ube44\uc2a4.csv')
df = pd.read_csv(path, encoding='utf-8-sig', skipinitialspace=True, dtype=str)
df.columns = df.columns.str.strip()
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
empty = df[df['X\uc88c\ud45c'].isin(['','nan']) | df['X\uc88c\ud45c'].isna()]
print(f'미조회: {len(empty)}행')
print(empty[['대상물명','기본주소','시군구명','건물명']].to_string())
