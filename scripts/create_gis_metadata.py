import pandas as pd
import shapefile

# NSID (National Spatial Data Infrastructure) Building Info standard mapping
# Note: Field aliases A0-A28 vary slightly by export, but these are standard for NSID AL_D010
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
    sf = shapefile.Reader('gis/AL_D010_11_20260409.shp', encoding='cp949')
    rec = sf.record(0)
    
    rows = []
    for i, field in enumerate(sf.fields[1:]):
        alias = field[0]
        name_kr, name_en = FIELD_MAP.get(alias, ('알수없음', 'Unknown'))
        sample_val = rec[i]
        
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
