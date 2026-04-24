# -*- coding: utf-8 -*-
import pandas as pd, numpy as np, sys
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data/서울10구_숙소_소방거리_유클리드.csv', encoding='utf-8-sig')

VARS = ['소방위험도_점수', '노후도_점수', '반경_50m_건물수', '집중도(%)']
X = df[VARS].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 1. 최적 군집 수 탐색 (엘보우 + 실루엣) ───────────────────────────
inertias, silhouettes = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(K_range, inertias, 'o-', color='#2c7bb6')
axes[0].set_xlabel('군집 수 (K)'); axes[0].set_ylabel('Inertia')
axes[0].set_title('엘보우 차트'); axes[0].grid(alpha=0.3)

axes[1].plot(K_range, silhouettes, 'o-', color='#d7191c')
axes[1].set_xlabel('군집 수 (K)'); axes[1].set_ylabel('실루엣 점수')
axes[1].set_title('실루엣 점수'); axes[1].grid(alpha=0.3)
best_k = K_range[np.argmax(silhouettes)]
axes[1].axvline(best_k, linestyle='--', color='gray', alpha=0.7, label=f'최적 K={best_k}')
axes[1].legend()

plt.tight_layout()
plt.savefig('c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data/clustering_elbow.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'최적 K: {best_k} (실루엣 {max(silhouettes):.3f})')

# ── 2. 최적 K로 군집화 ───────────────────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['군집'] = km_final.fit_predict(X_scaled)

# ── 3. 군집별 특성 ───────────────────────────────────────────────────
print(f'\n=== 군집별 특성 (K={best_k}) ===')
summary = df.groupby('군집')[VARS + ['위험점수_AHP']].mean().round(3)
summary['숙소수'] = df.groupby('군집').size()
print(summary.to_string())

# ── 4. 군집 시각화 (산점도) ─────────────────────────────────────────
colors = ['#27ae60','#e67e22','#c0392b','#2980b9','#8e44ad','#f39c12','#1abc9c']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for cluster in range(best_k):
    mask = df['군집'] == cluster
    cnt = mask.sum()
    avg_risk = df[mask]['위험점수_AHP'].mean()
    axes[0].scatter(df[mask]['소방위험도_점수'], df[mask]['노후도_점수'],
                    c=colors[cluster], s=20, alpha=0.6,
                    label=f'군집{cluster} ({cnt}개, 위험점수{avg_risk:.0f})')
    axes[1].scatter(df[mask]['반경_50m_건물수'], df[mask]['집중도(%)'],
                    c=colors[cluster], s=20, alpha=0.6,
                    label=f'군집{cluster}')

axes[0].set_xlabel('소방위험도_점수'); axes[0].set_ylabel('노후도_점수')
axes[0].set_title('소방위험도 vs 노후도'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
axes[1].set_xlabel('반경_50m_건물수'); axes[1].set_ylabel('집중도(%)')
axes[1].set_title('밀집도 vs 집중도'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data/clustering_scatter.png', dpi=150, bbox_inches='tight')
plt.close()

# ── 5. 저장 ─────────────────────────────────────────────────────────
df.to_csv('c:/Users/USER/Documents/GitHub/기말공모전/NJT-PJT/data/서울10구_숙소_소방거리_유클리드.csv', index=False, encoding='utf-8-sig')
print('\n[저장 완료]')
print('\n구별 군집 분포:')
print(df.groupby(['구','군집']).size().unstack(fill_value=0).to_string())
