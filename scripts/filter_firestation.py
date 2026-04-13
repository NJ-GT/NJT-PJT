# -*- coding: utf-8 -*-
import pandas as pd, os
BASE = os.path.join(os.path.dirname(__file__), '..')
path = os.path.join(BASE, '\uc18c\ubc29\uccad_\ud2b9\uc815\uc18c\ubc29\ub300\uc0c1\ubb3c\uc815\ubcf4\uc11c\ube44\uc2a4.csv')
df = pd.read_csv(path, encoding='utf-8-sig', skipinitialspace=True, dtype=str)
df.columns = df.columns.str.strip()
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
print(f'전체: {len(df)}행')

has_coord = ~(df['X\uc88c\ud45c'].isin(['', 'nan']) | df['X\uc88c\ud45c'].isna())
has_check = ~(df['\uc790\uccb4\uc810\uac80\ub300\uc0c1\uc5ec\ubd80'].isin(['', 'nan']) | df['\uc790\uccb4\uc810\uac80\ub300\uc0c1\uc5ec\ubd80'].isna())

result = df[has_coord & has_check].copy()
print(f'좌표 있음: {has_coord.sum()}행')
print(f'자체점검대상여부 값 있음: {has_check.sum()}행')
print(f'둘 다 해당: {len(result)}행')

out = os.path.join(BASE, 'data', '\uc18c\ubc29\uccad_\uc790\uccb4\uc810\uac80\ub300\uc0c1_\uc88c\ud45c\ud3ec\ud568.csv')
result.to_csv(out, index=False, encoding='utf-8-sig')
print(f'저장: {out}')
