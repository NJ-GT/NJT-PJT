import pandas as pd
import folium
from folium.plugins import HeatMap, MarkerCluster
from pyproj import Transformer
import os

# Paths
SUMMARY_PATH = 'data/XY_GIS_Analysis_Summary.csv'
OUTPUT_MAP = 'Map_Building_Density.html'

def get_risk_color(count):
    if count > 30:
        return 'darkred'   # High Risk
    elif count >= 15:
        return 'orange'    # Medium Risk
    else:
        return 'green'     # Low Risk

def generate_popup_html(row):
    total = row['반경_50m_건물수']
    if total == 0:
        return f"<b>인덱스:</b> {int(row['인덱스'])}<br>건물 없음"
    
    # Calculate percentages for the chart
    p_residential = (row['주택_수'] / total) * 100
    p_commercial = (row['상업_수'] / total) * 100
    p_accommodation = (row['숙박_수'] / total) * 100
    p_office = (row['사무_수'] / total) * 100
    p_other = (row['기타_수'] / total) * 100

    # CSS-based Donut/Pie Chart representation using conic-gradient
    # Order: Residential(Green), Commercial(Orange), Accommodation(Red), Office(Purple), Other(Gray)
    chart_style = f"""
    background: conic-gradient(
        #2ecc71 0% {p_residential}%, 
        #f39c12 {p_residential}% {p_residential + p_commercial}%, 
        #e74c3c {p_residential + p_commercial}% {p_residential + p_commercial + p_accommodation}%, 
        #9b59b6 {p_residential + p_commercial + p_accommodation}% {p_residential + p_commercial + p_accommodation + p_office}%, 
        #95a5a6 {p_residential + p_commercial + p_accommodation + p_office}% 100%
    );
    """

    html = f"""
    <div style="width: 220px; font-family: 'Malgun Gothic', sans-serif;">
        <h4 style="margin-bottom:10px;">ID: {int(row['인덱스'])} 상세 정보</h4>
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <div style="width: 80px; height: 80px; border-radius: 50%; {chart_style} margin-right: 15px; border: 2px solid #fff; box-shadow: 0 0 5px rgba(0,0,0,0.2);"></div>
            <div>
                <b style="font-size: 1.1em; color: #333;">총 건물 {int(total)}개</b><br>
                <span style="font-size: 0.9em; color: #666;">위험도: {row['밀집도']}</span>
            </div>
        </div>
        <table style="width: 100%; font-size: 0.85em; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #eee;">
                <td><span style="color:#2ecc71;">■</span> 주택</td><td>{int(row['주택_수'])} ({p_residential:.1f}%)</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td><span style="color:#f39c12;">■</span> 상업</td><td>{int(row['상업_수'])} ({p_commercial:.1f}%)</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td><span style="color:#e74c3c;">■</span> 숙박</td><td>{int(row['숙박_수'])} ({p_accommodation:.1f}%)</td>
            </tr>
            <tr style="border-bottom: 1px solid #eee;">
                <td><span style="color:#9b59b6;">■</span> 사무</td><td>{int(row['사무_수'])} ({p_office:.1f}%)</td>
            </tr>
            <tr>
                <td><span style="color:#95a5a6;">■</span> 기타</td><td>{int(row['기타_수'])} ({p_other:.1f}%)</td>
            </tr>
        </table>
    </div>
    """
    return html

def add_legend(m):
    legend_html = """
     <div style="
     position: fixed; 
     bottom: 50px; left: 50px; width: 160px; height: 160px; 
     background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
     padding: 10px; border-radius: 5px; opacity: 0.9;
     ">
     <b>화재 위험 밀집도</b><br>
     <i style="background: darkred; width: 18px; height: 18px; float: left; margin-right: 8px; border-radius: 50%;"></i> 위험 (> 30개)<br>
     <i style="background: orange; width: 18px; height: 18px; float: left; margin-right: 8px; border-radius: 50%;"></i> 주의 (15-30개)<br>
     <i style="background: green; width: 18px; height: 18px; float: left; margin-right: 8px; border-radius: 50%;"></i> 보통 (< 15개)<br>
     <hr style="margin: 5px 0;">
     <small>50m 반경 건물 수 기준</small>
     <br><br>
     <b>시설 구분</b><br>
     <small>주택: 초록 | 상업: 주황 | 숙박: 빨강</small>
     </div>
     """
    m.get_root().html.add_child(folium.Element(legend_html))

def create_map():
    print("Loading summary data for mapping...")
    if not os.path.exists(SUMMARY_PATH):
        print(f"Error: {SUMMARY_PATH} not found. Run analysis first.")
        return

    df = pd.read_csv(SUMMARY_PATH)
    df = df[df['반경_50m_건물수'] >= 0]
    
    transformer = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)
    
    lats, lons, heat_data = [], [], []
    for idx, row in df.iterrows():
        lon, lat = transformer.transform(row['보정_X'], row['보정_Y'])
        lats.append(lat)
        lons.append(lon)
        if row['반경_50m_건물수'] > 0:
            heat_data.append([lat, lon, float(row['반경_50m_건물수'])])
        
    df['lat'] = lats
    df['lon'] = lons

    # Initial Map
    center_lat, center_lon = sum(lats) / len(lats), sum(lons) / len(lons)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles='cartodbpositron')

    # 1. Heatmap Layer
    heat_group = folium.FeatureGroup(name="건물 밀집도 히트맵", show=True)
    HeatMap(heat_data, radius=15, blur=10).add_to(heat_group)
    heat_group.add_to(m)

    # 2. Marker Cluster Layer (Risk Analysis)
    marker_group = folium.FeatureGroup(name="개별 포인트 (위험도별 마커)", show=False)
    marker_cluster = MarkerCluster(name="Clustered Markers").add_to(marker_group)
    
    print("Generating markers with visual popups...")
    for idx, row in df.iterrows():
        color = get_risk_color(row['반경_50m_건물수'])
        popup_content = generate_popup_html(row)
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=6,
            popup=folium.Popup(popup_content, max_width=250),
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            weight=1
        ).add_to(marker_cluster)
        
    marker_group.add_to(m)

    # Add Layer Control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add Legend
    add_legend(m)
    
    print(f"Saving updated map to {OUTPUT_MAP}...")
    m.save(OUTPUT_MAP)
    print("Map successfully created.")

if __name__ == "__main__":
    create_map()
