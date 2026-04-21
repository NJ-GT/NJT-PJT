"""
[파일 설명]
국가공간정보(NSID) 건물 shapefile(AL_D010_11_20260409.shp)의 필드 메타데이터를 CSV로 출력하는 스크립트.

shapefile의 각 필드(A0~A28)에 대해 한글명, 영어명, 샘플 데이터를 정리하여
data/gis_building_info_metadata.csv로 저장한다. 다른 스크립트에서 참고용으로 사용.

입력: gis/AL_D010_11_20260409.shp  (서울시 건물 shapefile)
출력: data/gis_building_info_metadata.csv
"""

import pandas as pd
import shapefile

# 국가공간정보 건물 shapefile의 필드 코드(A0~A28)와 한글/영어 이름 매핑표
# 이 매핑을 참고하여 다른 스크립트에서 어떤 인덱스에 어떤 정보가 있는지 확인한다.
FIELD_MAP = {
    'A0': ('GIS유엔아이번호', 'GIS Unique Identifier'),
    'A1': ('고유번호', 'Building ID / Unique PNU'),
    'A2': ('건물고유번호', 'Building Unique ID'),
    'A3': ('법정동코드', 'Legal Dong Code'),
    'A4': ('법정동명', 'Legal Dong Name'),
    'A5': ('지번', 'Main Lot Number (Jibon)'),
    'A6': ('부번', 'Sub Lot Number (Bubeon)'),
    'A7': ('특수지구분명', 'Special Area Classification'),
    'A8': ('주용도코드', 'Primary Building Use Code'),
    'A9': ('주용도명', 'Primary Building Use Name'),
    'A10': ('건물구조코드', 'Building Structure Code'),
    'A11': ('기타구조명', 'Other Structure Name'),
    'A12': ('건축면적(㎡)', 'Building Area'),
    'A13': ('사용승인일자', 'Date of Approval for Use'),
    'A14': ('연면적(㎡)', 'Total Floor Area'),
    'A15': ('대지면면적(㎡)', 'Site Area'),
    'A16': ('높이(m)', 'Building Height'),
    'A17': ('건폐율(%)', 'Building Coverage Ratio'),
    'A18': ('용적률(%)', 'Floor Area Ratio'),
    'A19': ('도로명대지위치', 'Road Name Address (Base)'),
    'A20': ('도로명주소본번', 'Road Name Main Number'),
    'A21': ('도로명주소부번', 'Road Name Sub Number'),
    'A22': ('데이터기준일자', 'Data Reference Date'),
    'A23': ('시군구코드', 'SGG Code'),
    'A24': ('건물명', 'Building Name'),
    'A25': ('특수지코드', 'Special District Code'),
    'A26': ('지상층수', 'Above-ground Floor Count'),
    'A27': ('지하층수', 'Below-ground Floor Count'),
    'A28': ('작업일자', 'Last Modified Date')
}

def create_meta_csv():
    """shapefile 필드 구조를 읽어 메타데이터 CSV를 생성한다."""
    sf = shapefile.Reader('gis/AL_D010_11_20260409.shp', encoding='cp949')
    rec = sf.record(0)  # 0번 레코드의 샘플 데이터를 가져옴

    rows = []
    for i, field in enumerate(sf.fields[1:]):  # sf.fields[0]은 삭제마커(DeletionFlag)이므로 건너뜀
        alias = field[0]  # 필드 코드 (예: 'A8')
        name_kr, name_en = FIELD_MAP.get(alias, ('알수없음', 'Unknown'))
        sample_val = rec[i]  # 해당 필드의 첫 번째 레코드 값 (참고용)

        rows.append({
            'ID': alias,
            '한글 필드명': name_kr,
            'English Name': name_en,
            '샘플 데이터 (0번 레코드)': sample_val
        })

    df = pd.DataFrame(rows)
    df.to_csv('data/gis_building_info_metadata.csv', index=False, encoding='utf-8-sig')
    print("Metadata CSV created at data/gis_building_info_metadata.csv")

if __name__ == "__main__":
    create_meta_csv()
