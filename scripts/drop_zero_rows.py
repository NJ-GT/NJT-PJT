# -*- coding: utf-8 -*-
import pandas as pd, os
BASE = os.path.join(os.path.dirname(__file__), '..')
path = os.path.join(BASE, 'data', '\ub4f1\uae30\ubd80\ub4f1\ubcf8_\uc219\ubc15\uc5c5_\ud575\uc2ec\ud53c\ucc98_\ubc95\uc815\uc0c1\ud55c.csv')
df = pd.read_csv(path, encoding='utf-8-sig', skipinitialspace=True)
df.columns = df.columns.str.strip()
df['용적률(%)'] = pd.to_numeric(df['용적률(%)'], errors='coerce')
before = len(df)
df = df[~(df['용적률(%)'].isna() | (df['용적률(%)'] == 0))].copy()
print(f'{before}행 → {len(df)}행 ({before - len(df)}개 삭제)')
df.to_csv(path, index=False, encoding='utf-8-sig')
print(f'저장: {path}')
