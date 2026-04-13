# -*- coding: utf-8 -*-
import pandas as pd, os
BASE = os.path.join(os.path.dirname(__file__), '..')
path = os.path.join(BASE, 'data', '\ub4f1\uae30\ubd80\ub4f1\ubcf8_\uc219\ubc15\uc5c5_\ud575\uc2ec\ud53c\ucc98_\ubc95\uc815\uc0c1\ud55c.csv')
df = pd.read_csv(path, encoding='utf-8-sig', skipinitialspace=True)
df.columns = df.columns.str.strip()

null_도로명 = df['도로명대지위치'].isna() | (df['도로명대지위치'].astype(str).str.strip() == '')
null_건폐율 = df['건폐율_법정상한(%)'].isna() | (df['건폐율_법정상한(%)'].astype(str).str.strip().isin(['','nan']))

print(f'전체: {len(df)}행')
print(f'도로명대지위치 null: {null_도로명.sum()}개')
print(f'건폐율_법정상한(%) null: {null_건폐율.sum()}개')
print(f'둘 중 하나라도 null: {(null_도로명 | null_건폐율).sum()}개')

both = df[null_도로명 | null_건폐율][['대지위치','도로명대지위치','용도지역','건폐율_법정상한(%)','용적률_법정상한(%)']].copy()
both['null항목'] = ''
both.loc[null_도로명 & ~null_건폐율, 'null항목'] = '도로명대지위치'
both.loc[~null_도로명 & null_건폐율, 'null항목'] = '건폐율_법정상한'
both.loc[null_도로명 & null_건폐율, 'null항목'] = '둘다'

print()
print(both.to_string())
