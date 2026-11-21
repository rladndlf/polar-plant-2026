import io, os, numpy as np, pandas as pd, plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="극지식물 생육 대시보드", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

def read_csv(file):
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception:
            pass
    return pd.DataFrame()

def std_cols(df):
    # 업로드 파일 객체가 그대로 넘어온 경우 대비: bytes 추출 → DataFrame 변환
    if not isinstance(df, pd.DataFrame):
        if hasattr(df, "getvalue"):
            df = pd.read_csv(io.BytesIO(df.getvalue()))
        else:
            return pd.DataFrame()
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "")
                .str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = pd.to_datetime(parts[0], format="%Y%m%d") + pd.to_timedelta(
                parts[1].astype(int), unit="h"
            )
        df["time"] = parsed
    return df

def trendline(x, y, color="red"):
    xs = x.dropna()
    ys = y.loc[xs.index].dropna()
    if len(xs) < 2:
        return None
    m, b = np.polyfit(xs, ys, 1)
    return go.Scatter(
        x=[xs.min(), xs.max()],
        y=[m * xs.min() + b, m * xs.max() + b],
        mode="lines",
        line=dict(color=color, width=2, dash="dash"),
        name=f"추세선 (r={np.corrcoef(xs, ys)[0,1]:.2f})",
    )

def daily_line(df, col, label, color):
    if col not in df.columns:
        return None
    s = df[[ "time", col ]].dropna()
    if s.empty:
        return None
    daily = s.copy()
    daily["date"] = daily["time"].dt.date
    g = daily.groupby("date")[col].mean().reset_index()
    if len(g) < 2:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=g["date"],
            y=g[col],
            mode="lines+markers",
            name=label,
            line=dict(color=color, width=2),
            marker=dict(size=5),
        )
    )
    fig.update_layout(
        title=f"{label} (일별 평균)",
        xaxis_title="날짜",
        yaxis_title=label,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="x unified",
    )
    return fig

def bar_chart(df, col, label, color):
    if col not in df.columns:
        return None
    s = df[[ "개체번호", col ]].dropna()
    if s.empty or len(s) < 2:
        return None
    s = s.sort_values("개체번호")
    mean_val = s[col].mean()
    q75 = s[col].quantile(0.75)
    q25 = s[col].quantile(0.25)
    colors = []
    for v in s[col]:
        if v >= q75:
            colors.append("#2e8b57")
        elif v <= q25:
            colors.append("#c0392b")
        else:
            colors.append(color)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=s["개체번호"].astype(str),
            y=s[col],
            marker_color=colors,
            hovertemplate="%{x}호: %{y:.1f} " + label + "<extra></extra>",
        )
    )
    fig.add_hline(y=mean_val, line_dash="dot", line_color="gray", annotation_text="평균")
    fig.update_layout(
        title=f"{label} (개체별, 상위 25%·하위 25% 강조)",
        xaxis_title="개체",
        yaxis_title=label,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig

def boxplot_fig(df, col, label):
    if col not in df.columns:
        return None
    s = df[[ "개체번호", col ]].dropna()
    if s.empty or len(s) < 3:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=s[col],
            name=label,
            fillcolor="rgba(52,152,219,0.55)",
            line_color="rgba(52,152,219,0.9)",
            boxmean=True,
            hovertemplate="%{y:.1f} " + label + "<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{label} 분포 (개체별)",
        yaxis_title=label,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig

def scatter_fig(df, xcol, ycol, xlabel, ylabel):
    if xcol not in df.columns or ycol not in df.columns:
        return None
    s = df[[ xcol, ycol ]].dropna()
    if len(s) < 3:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=s[xcol],
            y=s[ycol],
            mode="markers",
            marker=dict(size=9, color="rgba(46,139,87,0.7)", line=dict(width=1, color="white")),
            hovertemplate="{xlabel}: %{x:.1f}<br>{ylabel}: %{y:.1f}<extra></extra>",
        )
    )
    tl = trendline(s[xcol], s[ycol])
    if tl is not None:
        fig.add_trace(tl)
    r = np.corrcoef(s[xcol], s[ycol])[0, 1]
    fig.update_layout(
        title=f"{xlabel} vs {ylabel} (r = {r:.3f})",
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        hovermode="closest",
    )
    return fig

# ===== 사이드바 =====
st.sidebar.title("🌱 극지식물 생육 대시보드")

env_file = st.sidebar.file_uploader("환경 CSV 업로드", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 CSV 업로드", type="csv")

growth_options = ["생중량(g)", "잎 수(장)", "지상부 길이(mm)", "지하부 길이(mm)"]
growth_item = st.sidebar.selectbox("생육 측정 항목 선택", growth_options)

env_df = pd.DataFrame()
growth_df = pd.DataFrame()

if env_file is not None:
    with st.spinner("환경 데이터를 읽고 있어요..."):
        env_df = std_cols(read_csv(env_file))
if growth_file is not None:
    with st.spinner("생육 데이터를 읽고 있어요..."):
        growth_df = std_cols(growth_file)

school_name = "우리 학교"
for f in (env_file, growth_file):
    if f is not None:
        stem = os.path.splitext(os.path.basename(f.name))[0]
        school_name = stem.replace("_", " ")
        break

st.sidebar.markdown(f"**{school_name}** 극지식물 분석")
st.sidebar.caption(
    "CSV를 올리면 자동으로 열 이름을 표준화해 어떤 학교 데이터든 에러 없이 표시해요."
)
st.sidebar.caption(
    "표준 열: time·temperature·humidity·ph·ec / 개체번호·잎 수·지상부 길이·지하부 길이·생중량"
)

# ===== 탭 =====
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

with tab1:
    try:
        st.header("연구 배경 및 목적")
        st.markdown(
            """
- **배경:** 극지방과 고산 지대처럼 춥고 건조한 극한 환경에서도 살아가는 식물(극지식물)이 있어요. 
  이런 식물들은 온도·습도·pH 같은 환경 조건이 조금만 달라도 생육 결과에 큰 차이를 보일 수 있어요.
- **목적:** 우리 학교 극지식물 실험에서 **개체별로 생육이 얼마나 다른지**를 숫자로 확인하고, 
  환경 조건과의 관계도 함께 살펴봐요.
- **분석 항목:** 평균·표준편차·변동계수(CV), 상관계수(r), 상위·하위 25% 비교 등 고등학교 통계 과정에서 배우는 개념을 사용해요.
"""
        )
        st.markdown("> 💡 **통계 용어 잠깐!**  \n> • **평균**: 모든 값을 더해 개수로 나눈 대표값  \n> • **표준편차**: 값들이 평균에서 얼마나 흩어져 있는지 나타내는 수치 (클수록 들쭉날쭉)  \n> • **변동계수(CV)**: 표준편차를 평균으로 나눈 값 → 단위가 다른 대상끼리 흩어짐을 비교할 때 유용  \n> • **상관계수(r)**: 두 값이 함께 움직이는 정도 (-1 ~ 1, 0에 가까울수록 관계 약함)", )

        st.subheader("데이터 요약")

        rows = []
        if env_df is not None and not env_df.empty and "time" in env_df.columns:
            rows.append(["측정 기간 (환경)", f"{env_df['time'].min().strftime('%Y-%m-%d')} ~ {env_df['time'].max().strftime('%Y-%m-%d')}"])
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            rows.append(["환경 측정 항목", ", ".join(env_cols) if env_cols else "없음"])
        if growth_df is not None and not growth_df.empty and "개체번호" in growth_df.columns:
            rows.append(["총 개체 수", str(len(growth_df["개체번호"].dropna()))])
            growth_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이", "생중량"] if c in growth_df.columns]
            rows.append(["생육 측정 항목", ", ".join(growth_cols) if growth_cols else "없음"])
        if not rows:
            rows.append(["데이터", "아직 CSV가 업로드되지 않았어요."])
        st.table(pd.DataFrame(rows, columns=["항목", "내용"]))

        st.subheader("주요 지표")
        c1, c2, c3, c4 = st.columns(4)

        total = 0
        if growth_df is not None and not growth_df.empty and "개체번호" in growth_df.columns:
            total = int(growth_df["개체번호"].dropna().count())
        c1.metric("총 개체 수", total)

        avg_temp = None
        if env_df is not None and not env_df.empty and "temperature" in env_df.columns:
            v = env_df["temperature"].dropna()
            if len(v) >= 2:
                avg_temp = float(v.mean())
        c2.metric("평균 온도 (°C)", round(avg_temp, 2) if avg_temp is not None else "―")

        avg_hum = None
        if env_df is not None and not env_df.empty and "humidity" in env_df.columns:
            v = env_df["humidity"].dropna()
            if len(v) >= 2:
                avg_hum = float(v.mean())
        c3.metric("평균 습도 (%)", round(avg_hum, 2) if avg_hum is not None else "―")

        avg_bio = None
        if growth_df is not None and not growth_df.empty and "생중량" in growth_df.columns:
            v = growth_df["생중량"].dropna()
            if len(v) >= 2:
                avg_bio = float(v.mean())
        c4.metric("평균 생중량 (g)", round(avg_bio, 3) if avg_bio is not None else "―")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab2:
    try:
        if env_df is None or env_df.empty:
            st.info("🌡️ 환경 데이터 탭이에요. 왼쪽에 환경 CSV를 올려주세요.")
        else:
            st.header("🌡️ 환경 데이터")

            st.subheader("환경 요약 (평균 · 최소 · 최대)")
            env_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if env_cols:
                summary_rows = []
                for c in env_cols:
                    v = env_df[c].dropna()
                    if len(v) >= 1:
                        summary_rows.append([c, round(float(v.mean()), 3), round(float(v.min()), 3), round(float(v.max()), 3)])
                if summary_rows:
                    labels = {"temperature": "온도 (°C)", "humidity": "습도 (%)", "ph": "pH", "ec": "EC (mS/cm)"}
                    st.table(
                        pd.DataFrame(
                            [[labels.get(r[0], r[0]), r[1], r[2], r[3]] for r in summary_rows],
                            columns=["측정 항목", "평균", "최소", "최대"],
                        )
                    )
                else:
                    st.info("환경 수치를 계산할 데이터가 충분하지 않아요.")
            else:
                st.info("환경 데이터에서 temperature·humidity·ph·ec 중 어떤 열도 찾지 못했어요. CSV 열 이름을 확인해주세요.")

            st.subheader("날짜별 시계열 (일별 평균)")
            ts_cols = [c for c in ["temperature", "humidity", "ph", "ec"] if c in env_df.columns]
            if ts_cols:
                fig_ts = make_subplots(rows=len(ts_cols), cols=1, shared_xaxes=True, vertical_spacing=0.08)
                colors_map = {"temperature": "#e74c3c", "humidity": "#3498db", "ph": "#8e44ad", "ec": "#16a085"}
                for i, c in enumerate(ts_cols):
                    d = daily_line(env_df, c, labels.get(c, c), colors_map.get(c, "#333"))
                    if d is not None:
                        for tr in d.data:
                            fig_ts.add_trace(tr, row=i + 1, col=1)
                fig_ts.update_layout(
                    height=200 + 180 * len(ts_cols),
                    font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
                    margin=dict(l=40, r=20, t=40, b=40),
                    showlegend=False,
                )
                if len(ts_cols) > 1:
                    fig_ts.update_xaxes(title_text="날짜", row=len(ts_cols), col=1)
                st.plotly_chart(fig_ts, width="stretch")
            else:
                st.info("표시할 환경 시계열 열이 없어요.")

            with st.expander("📄 환경 데이터 원본 보기 + CSV 다운로드"):
                if not env_df.empty:
                    st.dataframe(env_df, width="stretch")
                    csv_bytes = env_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "CSV 다운로드",
                        csv_bytes,
                        file_name="환경데이터_original.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("표시할 원본 데이터가 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")

with tab3:
    try:
        if growth_df is None or growth_df.empty:
            st.info("📊 생육 결과 탭이에요. 왼쪽에 생육 결과 CSV를 올려주세요.")
        else:
            st.header("📊 생육 결과")

            st.subheader("🥇 핵심 결과")
            kc1, kc2, kc3 = st.columns(3)
            if "생중량" in growth_df.columns:
                v = growth_df["생중량"].dropna()
                if len(v) >= 2:
                    mean_val = float(v.mean())
                    max_idx = int(v.idxmax())
                    min_idx = int(v.idxmin())
                    kc1.metric("평균 생중량 (g)", round(mean_val, 3))
                    kc2.metric("최대 생중량 개체", f"개체 {max_idx}호")
                    kc3.metric("최소 생중량 개체", f"개체 {min_idx}호")
                else:
                    kc1.metric("평균 생중량 (g)", "―")
                    kc2.metric("최대 생중량 개체", "―")
                    kc3.metric("최소 생중량 개체", "―")
            else:
                st.info("생중량 열이 없어 핵심 결과 카드를 표시할 수 없어요.")

            st.subheader("개체별 생육 비교 (막대 그래프)")
            bio_item_display = {
                "생중량(g)": "생중량 (g)",
                "잎 수(장)": "잎 수 (장)",
                "지상부 길이(mm)": "지상부 길이 (mm)",
                "지하부 길이(mm)": "지하부 길이 (mm)",
            }
            bc = bar_chart(growth_df, growth_item, bio_item_display.get(growth_item, growth_item), "#3498db")
            if bc is not None:
                st.plotly_chart(bc, width="stretch")
            else:
                st.info(f"선택한 '{growth_item}' 열이 생육 데이터에 없거나 값이 충분하지 않아요.")

            st.subheader("생중량 분포")
            bx = boxplot_fig(growth_df, "생중량", "생중량 (g)")
            if bx is not None:
                st.plotly_chart(bx, width="stretch")
            else:
                st.info("생중량 분포가 박스플롯을 그릴 만큼 데이터가 충분하지 않아요.")

            st.subheader("상관관계 분석 (산점도 + 추세선)")
            sc_cols = [c for c in ["잎 수", "지상부 길이"] if c in growth_df.columns]
            sc_cols = [c for c in sc_cols if "생중량" in growth_df.columns]
            if sc_cols:
                for xc in sc_cols:
                    lbl_map = {"잎 수": "잎 수 (장)", "지상부 길이": "지상부 길이 (mm)"}
                    sf = scatter_fig(growth_df, xc, "생중량", lbl_map.get(xc, xc), "생중량 (g)")
                    if sf is not None:
                        st.plotly_chart(sf, width="stretch")
                    else:
                        st.info(f"{lbl_map.get(xc, xc)} vs 생중량 관계를 표시할 데이터가 부족해요.")
            else:
                st.info("잎 수·지상부 길이·생중량 중 표시할 수 있는 조합이 없어요.")

            with st.expander("📄 생육 데이터 원본 보기 + CSV 다운로드"):
                if not growth_df.empty:
                    gd = growth_df.copy()
                    if "개체번호" in gd.columns and pd.api.types.is_numeric_dtype(gd["개체번호"]):
                        gd["개체번호"] = gd["개체번호"].astype(int)
                    st.dataframe(gd, width="stretch")
                    csv_bytes = gd.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "CSV 다운로드",
                        csv_bytes,
                        file_name="생육결과_original.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("표시할 원본 데이터가 없어요.")
    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
