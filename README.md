# eruption-cloud-jhistorical

過去の日本における大規模噴火の噴煙解析コード。  
Woodhouse et al. (2013) の 1D 噴煙モデルと再解析気象データ（ERA5 / 20CRv3）を組み合わせて、
各噴火時の噴煙高度から質量放出率（質量フラックス Q）を推定する。

## 解析対象噴火

| 噴火 | 年月日 | 気象データ | 観測噴煙高 |
|---|---|---|---|
| 桜島大正噴火 (Sakurajima1914) | 1914-01-12 | 20CRv3 | 18,000 m |
| 北海道駒ヶ岳 (Komagatake1929) | 1929-06-17 | 20CRv3 | 13,900 m |
| 十勝岳 (Tokachi1962) | 1962-06-29 | ERA5 | 12,000 m |
| 有珠山 (Usu1977) | 1977-08-07 | ERA5 | 12,000 m |

## モデル

Woodhouse, M. J., Hogg, A. J., Phillips, J. C., & Sparks, R. S. J. (2013).
Interaction between volcanic plumes and wind during the 2010 Eyjafjallajökull eruption, Iceland.
*Journal of Geophysical Research: Solid Earth*, 118(1), 92–109.
https://doi.org/10.1029/2012JB009592

状態ベクトル `y = [y1, y2, y3, θ]`（スケール済み質量・運動量・エネルギーフラックス、仰角）を
RK4 で弧長方向に積分する。

## リポジトリ構成

```
eruption-cloud-jhistorical/
├── src/
│   ├── plume_model.py       # Woodhouse 2013 ODE、RK4 積分、Q 逆算
│   ├── atmos_profile.py     # 気象プロファイル準備（z=0 アンカー挿入・補間）
│   └── loader/
│       ├── era5.py          # ERA5 NetCDF 読み込み / CDS API ダウンロード
│       └── cr20.py          # 20CRv3 THREDDS/OPeNDAP 読み込み
├── scripts/
│   ├── forward.py           # 順問題：複数 (r0, u0) のプロファイル計算・図示
│   ├── qdet.py              # 逆問題：観測噴煙高から Q を格子探索
│   ├── qh_sensitivity.py    # Q–H 感度解析：r0 sweep、n0/u0 パラメータ別に図示
│   └── fetch_data.py        # 気象データ取得テスト（OPeNDAP / CDS API）
├── eruptions/
│   └── catalog.yaml         # 噴火メタデータ（座標・標高・日時・パラメータ格子）
├── data/                    # 気象データ（CSV は git 管理、NC は gitignore）
│   ├── Sakurajima1914/      # 20CRv3 プロファイル CSV
│   ├── Komagatake1929/      # 20CRv3 プロファイル CSV
│   ├── Tokachi1962/         # ERA5 NetCDF（gitignore）
│   └── Usu1977/             # ERA5 NetCDF（gitignore）
├── ipynb/                   # 旧ノートブック（参考用）
├── output/                  # 実行結果（gitignore）
│   ├── <EruptionKey>/
│   │   ├── forward/         # PNG
│   │   └── qdet/            # CSV
│   └── qh_sensitivity/      # Q–H 感度解析 PNG（n0・u0 別）
└── pyproject.toml
```

## セットアップ

[uv](https://docs.astral.sh/uv/) を使用する。

```bash
# 依存パッケージのインストール
uv sync

# ERA5 ダウンロード機能も使う場合（~/.cdsapirc が必要）
uv sync --extra cds
```

ERA5 の認証情報は [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/) でアカウントを作成し、
`~/.cdsapirc` に以下の形式で設定する。

```
url: https://cds.climate.copernicus.eu/api
key: <your-api-key>
```

## 使い方

### 気象データ取得テスト

各噴火の気象データ取得をテストする。20CRv3 は OPeNDAP、ERA5 はローカル NC ファイルを確認する。

```bash
uv run python scripts/fetch_data.py

# ERA5 の CDS API ダウンロードも含めてテストする場合（数分かかる）
uv run python scripts/fetch_data.py --test-download
```

### 順問題（forward simulation）

複数の (r0, u0) 組み合わせで噴煙プロファイルを計算し、PNG を出力する。

```bash
uv run python scripts/forward.py --eruption Tokachi1962
uv run python scripts/forward.py --eruption all        # 全噴火を一括実行
```

出力: `output/<EruptionKey>/forward/plume_forward_<EruptionKey>.png`

### 逆問題（Q determination）

観測された噴煙高 `z_target` に一致する質量フラックス Q を、
(r0, T0, n0) の格子で探索して CSV に保存する。

```bash
uv run python scripts/qdet.py --eruption Tokachi1962
uv run python scripts/qdet.py --eruption all
```

出力: `output/<EruptionKey>/qdet/<eruption>_qdet_z<ztarget>m_<datetime>.csv`

### Q–H 感度解析

r0 を 10–250 m で sweep し、Q と噴煙高の関係を n0・u0 パラメータ別に図示する。
4 火山を 1 図にまとめ、観測噴煙高を黒点で示す。n0 × u0 の組み合わせで 18 枚生成する。

```bash
uv run python scripts/qh_sensitivity.py
```

出力: `output/qh_sensitivity/qh_sensitivity_n0{003|004|005}_u0{100|120|...|200}.png`

### 新規噴火の追加

`eruptions/catalog.yaml` に噴火情報を追記するだけで全スクリプトが対応する。
ERA5 の場合、NC ファイルが存在しなければ `forward.py` / `qdet.py` 実行時に自動ダウンロードされる。
20CRv3 の場合、CSV が存在しなければ OPeNDAP から取得して `data/` に保存される。

```yaml
eruptions:
  NewEruption:
    name_ja: 有珠山 2000年噴火
    lat: 42.544
    lon: 140.839
    vent_height_m: 733.0
    analysis_utc: "2000-03-31 03:00"
    reanalysis: era5
    era5_ncfile: data/NewEruption/era5_usu_20000331_03UTC_pl.nc
    era5_download:
      area: [43.2, 139.8, 41.8, 141.8]
    forward:
      r0_list: [10, 30, 50, 100]
      u0_list: [50, 100, 150]
    qdet:
      z_target_m: 12000.0
      r0_grid: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
      T0_grid_K: [1073, 1173, 1273, 1373]
      n0_grid: [0.03, 0.04, 0.05]
```

## 今後の開発

- [ ] **不確かさ評価**  
  格子探索の結果から Q 推定値の信頼区間を可視化する。

- [ ] **新規噴火の追加**  
  対象噴火を拡大する（例：磐梯山 1888、有珠山 2000、霧島新燃岳 2011 など）。  
  20CRv3 カバー範囲外の古い噴火については代替再解析データの検討が必要。

- [ ] **結果の集約・比較プロット**  
  複数噴火の Q 推定値を並べた比較図・テーブルを生成するスクリプトを追加する。

- [ ] **モデル拡張の検討**  
  相変化（水蒸気凝結）や噴火柱崩壊条件の判定など、
  Woodhouse (2013) の拡張版モデルへの対応。
