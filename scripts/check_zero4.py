# -*- coding: utf-8 -*-
import pandas as pd, os
BASE = os.path.join(os.path.dirname(__file__), '..')
path = os.path.join(BASE, 'data', '\ub4f1\uae30\ubd80\ub4f1\ubcf8_\uc219\ubc15\uc5c5_\ud575\uc2ec\ud53c\ucc98_\ubc95\uc815\uc0c1\ud55c.csv')
df = pd.read_csv(path, encoding='utf-8-sig', skipinitialspace=True)
df.columns = df.columns.str.strip()
for col in ['용적률(%)', '건폐율(%)']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
zero = df[df['용적률(%)'].isna() | (df['용적률(%)'] == 0)]
print(f'0/null 행: {len(zero)}개')
print(zero[['대지위치','용도지역','건폐율_법정상한(%)','용적률_법정상한(%)','용적률(%)','건폐율(%)']].to_string())
