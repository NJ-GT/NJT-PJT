import pandas as pd, os

BASE = r'C:\Users\USER\Documents\GitHub\기말공모전\NJT-PJT'
IN   = os.path.join(BASE, 'data', '화재출동', '화재출동_2021_2024.csv')
OUT  = os.path.join(BASE, 'data', '화재출동', '화재출동_사상자발생.csv')

df = pd.read_csv(IN, encoding='utf-8-sig', low_memory=False)
print(f'전체: {len(df)}행')

df['사망자수'] = pd.to_numeric(df['사망자수'], errors='coerce').fillna(0)
df['부상자수'] = pd.to_numeric(df['부상자수'], errors='coerce').fillna(0)

result = df[(df['사망자수'] >= 1) | (df['부상자수'] >= 1)].copy()
print(f'사상자 발생: {len(result)}행')
print(f'  - 사망자수 1명 이상: {(result["사망자수"] >= 1).sum()}건')
print(f'  - 부상자수 1명 이상: {(result["부상자수"] >= 1).sum()}건')

result.to_csv(OUT, index=False, encoding='utf-8-sig')
print(f'저장: {OUT}')
