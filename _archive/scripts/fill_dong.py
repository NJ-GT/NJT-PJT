# -*- coding: utf-8 -*-
import pandas as pd, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/통합숙박시설_최종안0421.csv', encoding='utf-8-sig')
print(f'처리 전 null: {df["동"].isna().sum()}행')

def extract_dong(addr):
    if pd.isna(addr):
        return None
    parts = str(addr).strip().split()
    # '서울특별시 종로구 종로2가 9번지' → index 2가 동
    for i, p in enumerate(parts):
        if p.endswith('구'):   # '시'는 서울특별시와 겹치므로 제외
            if i + 1 < len(parts):
                return parts[i + 1]
    return None

mask = df['동'].isna()
df.loc[mask, '동'] = df.loc[mask, '주소'].apply(extract_dong)

print(f'처리 후 null: {df["동"].isna().sum()}행')
print('\n[샘플 확인]')
print(df.loc[mask, ['구', '동', '주소']].head(15).to_string())

df.to_csv('data/통합숙박시설_최종안0421.csv', index=False, encoding='utf-8-sig')
print('\n저장 완료')
