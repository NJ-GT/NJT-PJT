import sys, json, geopandas as gpd, os
sys.stdout.reconfigure(encoding='utf-8')

oa = gpd.read_file('data/bnd_oa_11_2025_2Q/bnd_oa_11_2025_2Q.shp').to_crs('EPSG:4326')
oa['geometry'] = oa['geometry'].simplify(0.0003, preserve_topology=True)

with open('data/oa_density.json', encoding='utf-8') as f:
    raw = json.load(f)
prop_map = {f['properties']['id']: f['properties'] for f in raw['features']}

def slim(p):
    return {
        'id':    p.get('id', ''),
        'gu':    p.get('gu_name', ''),
        'no':    p.get('oa_no', ''),
        'cnt':   p.get('count', 0),
        'fl':    round(p.get('avg_floors') or 0, 1),
        'age':   round(p.get('avg_age') or 0, 1),
        'rat':   round(p.get('ratio', 0), 1),
        'fire':  round(p.get('fire_score', 0), 1),
        'area':  round(p.get('area_ha', 0), 2),
    }

def round_geom(geom):
    t = geom['type']
    if t == 'Polygon':
        coords = [[[round(x,4), round(y,4)] for x,y in ring]
                  for ring in geom['coordinates']]
        return {'type': 'Polygon', 'coordinates': coords}
    elif t == 'MultiPolygon':
        mp = []
        for poly in geom['coordinates']:
            rings = [[[round(x,4), round(y,4)] for x,y in ring] for ring in poly]
            mp.append(rings)
        return {'type': 'MultiPolygon', 'coordinates': mp}
    return geom

features = []
for _, row in oa.iterrows():
    oid = row['TOT_OA_CD']
    props = prop_map.get(oid, {'id': oid, 'count': 0})
    geom = round_geom(row['geometry'].__geo_interface__)
    features.append({'type': 'Feature', 'geometry': geom, 'properties': slim(props)})

out = {'type': 'FeatureCollection', 'features': features}
with open('data/oa_3d.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, separators=(',', ':'))

sz = os.path.getsize('data/oa_3d.json')
print(f'저장 완료: {sz//1024} KB  집계구 {len(features)}개')
