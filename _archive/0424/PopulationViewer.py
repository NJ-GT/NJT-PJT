try:
    import geopandas as gpd
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "geopandas"])
    import geopandas as gpd

try:
    import folium
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "folium"])
    import folium

from folium import GeoJson
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import webbrowser
import os


class PopulationViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("서울시 생활인구 조회 시스템")
        
        # 창 크기 설정 (너비x높이+x좌표+y좌표)
        window_width = 850
        window_height = 1000
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = 0  # 화면 최상단에 위치
        self.root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.root.resizable(True, True)  # 사용자가 창 크기를 조절할 수 있도록 설정
        
        # 메인 캔버스와 스크롤바 생성
        self.canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # 마우스 휠 이벤트 바인딩
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # 스크롤바와 캔버스 배치
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        self.shp_path = None
        self.csv_path = None
        
        style = ttk.Style()
        style.configure('TButton', padding=5)
        style.configure('TLabel', padding=5)
        
        self.create_widgets()
        
    def create_widgets(self):
        # 제목
        title_label = ttk.Label(
            self.scrollable_frame, 
            text="서울시 생활인구 조회 시스템",
            font=('맑은 고딕', 36, 'bold')
        )
        title_label.pack(pady=20)
        
        # 데이터 선택 프레임
        data_frame = ttk.LabelFrame(self.scrollable_frame, text="데이터 파일 선택", padding=10)
        data_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Button(
            data_frame, 
            text="지도 데이터 선택 (집계구.shp)",
            command=self.load_shp
        ).pack(fill='x', pady=5)
        
        ttk.Button(
            data_frame, 
            text="생활인구 데이터 선택 (csv)",
            command=self.load_csv
        ).pack(fill='x', pady=5)
        
        # 지도 열기 버튼 - scrollable_frame으로 변경
        ttk.Button(
            self.scrollable_frame,  # self.root에서 변경
            text="지도 열기 (마우스를 올려서 집계구 코드 확인)",
            command=self.show_map
        ).pack(pady=10, padx=20, fill='x')  # fill='x' 추가
        
        # 집계구 코드 입력 프레임
        code_frame = ttk.LabelFrame(self.scrollable_frame, text="집계구 코드 입력", padding=10)
        code_frame.pack(fill='x', padx=20, pady=10)
        
        ttk.Label(
            code_frame,
            text="지도에서 확인한 집계구 코드를 입력해주세요\n(여러 개인 경우 쉼표로 구분)",
            font=('맑은 고딕', 28)
        ).pack(pady=5)
        
        self.code_entry = ttk.Entry(
            code_frame,
            width=30,
            font=('맑은 고딕', 28)
        )
        self.code_entry.pack(pady=10)
        
        # 조회 버튼 - scrollable_frame으로 변경
        ttk.Button(
            self.scrollable_frame,  # self.root에서 변경
            text="조회하기",
            command=self.check_population,
            style='Query.TButton'
        ).pack(pady=15, padx=20, fill='x')  # fill='x' 추가
        
        # 결과 프레임
        result_frame = ttk.LabelFrame(self.scrollable_frame, text="조회 결과", padding=10)
        result_frame.pack(fill='x', padx=20, pady=10)
        
        self.result_label = ttk.Label(
            result_frame, 
            text="집계구 코드를 입력한 후\n조회 버튼을 클릭해주세요",
            font=('맑은 고딕', 28),
            justify='center'
        )
        self.result_label.pack(pady=10)

    def show_map(self):
        if not self.shp_path:
            messagebox.showerror("오류", "지도 데이터를 먼저 선택해주세요.")
            return
            
        try:
            geo = gpd.read_file(self.shp_path)
            geo = geo[['TOT_REG_CD', 'geometry']]
            if geo.crs is None:
                geo = geo.set_crs(epsg=5179)
            geo = geo.to_crs(epsg=4326)
            geo['geometry'] = geo['geometry'].simplify(tolerance=0.00001)

            m = folium.Map(
                location=[37.5665, 126.9780],
                zoom_start=11,
                tiles='cartodbpositron',
                prefer_canvas=True
            )

            GeoJson(
                geo,
                style_function=lambda x: {
                    'fillColor': 'white',
                    'color': '#666666',
                    'weight': 0.3,
                    'fillOpacity': 0.1
                },
                highlight_function=lambda x: {
                    'color': '#ff0000',
                    'weight': 1,
                    'fillOpacity': 0.3
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['TOT_REG_CD'],
                    aliases=['집계구코드:'],
                    style="font-size: 25px; padding: 5px;"
                )
            ).add_to(m)

            output_file = 'seoul_map_simple.html'
            m.save(output_file)
            webbrowser.open('file://' + os.path.realpath(output_file))
            
        except Exception as e:
            messagebox.showerror("오류", f"지도 생성 중 오류가 발생했습니다.\n{str(e)}")

    def check_population(self):
        if not self.csv_path:
            messagebox.showerror("오류", "생활인구 데이터를 먼저 선택해주세요.")
            return
        
        try:
            # 입력된 코드들을 쉼표로 구분하여 리스트로 변환
            target_codes = [code.strip() for code in self.code_entry.get().split(',')]
            if not target_codes or all(not code for code in target_codes):
                messagebox.showwarning("경고", "집계구 코드를 입력해주세요.")
                return
            
            data = pd.read_csv(self.csv_path, encoding='cp949')
            
            result_text = "[조회 완료]\n\n"
            
            for hour in range(24):
                result = data[
                    (data['집계구코드'].astype(str).isin(target_codes)) & 
                    (data['시간대구분'] == hour)
                ]
                
                if len(result) > 0:
                    total_population = result['총생활인구수'].sum()
                    result_text += f"{hour:02d}시: 총 생활인구 {total_population:.1f}명\n"
                else:
                    result_text += f"{hour:02d}시: 데이터 없음\n"
            
            self.result_label.config(text=result_text)
                
        except Exception as e:
            messagebox.showerror("오류", f"조회 중 오류가 발생했습니다.\n{str(e)}")

    def load_shp(self):
        self.shp_path = filedialog.askopenfilename(
            title="지도 데이터 선택",
            filetypes=[("Shape 파일", "*.shp")]
        )
        if self.shp_path:
            messagebox.showinfo("알림", "지도 데이터가 선택되었습니다.")

    def load_csv(self):
        self.csv_path = filedialog.askopenfilename(
            title="생활인구 데이터 선택",
            filetypes=[("CSV 파일", "*.csv")]
        )
        if self.csv_path:
            messagebox.showinfo("알림", "생활인구 데이터가 선택되었습니다.")

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = PopulationViewer()
    app.run()
