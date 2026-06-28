import math

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from terrain_nav import (
    load_dem_from_bytes,
    generate_synthetic_dem,
    clean_dem,
    estimate_pixel_size_m,
    simulate_flight,
    sample_profile,
    search_by_correlation,
    confidence_label,
    parse_nmea_profile_with_dt,
    terrain_from_radio,
    kalman_1d,
    trajectory_error_metrics,
    circular_angle_error_deg,
    route_to_csv,
    route_to_geojson,
    make_report_json,
    pixel_to_map,
    pixel_to_lonlat,
    velocity_components,
    estimate_accuracy_radius_m,
    nmea_quality_report,
    parse_plain_heights_text,
    heights_to_terrain_profile,
    build_azimuth_candidates,
    map_to_pixel,
)


st.set_page_config(
    page_title="Хакатон №4. Команда: СКБ 'ФИЗ-ТЕХ' 5",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.35rem;
        font-weight: 850;
        line-height: 1.08;
        margin-bottom: 0.15rem;
    }
    .subtitle {
        font-size: 1.02rem;
        opacity: 0.78;
        margin-bottom: 1.0rem;
    }
    .hero-box {
        padding: 1rem 1.15rem;
        border-radius: 18px;
        border: 1px solid rgba(130, 130, 130, 0.22);
        background: linear-gradient(135deg, rgba(80, 120, 255, 0.10), rgba(0, 190, 140, 0.10));
        margin-bottom: 0.75rem;
    }
    .solution-card {
        padding: 0.85rem 0.9rem;
        border-radius: 16px;
        border: 1px solid rgba(130, 130, 130, 0.20);
        background: rgba(120, 120, 120, 0.07);
        min-height: 105px;
    }
    .solution-card .label {
        font-size: 0.78rem;
        opacity: 0.68;
        text-transform: uppercase;
        letter-spacing: 0.04rem;
    }
    .solution-card .value {
        font-size: 1.45rem;
        font-weight: 780;
        margin-top: 0.25rem;
    }
    .solution-card .hint {
        font-size: 0.82rem;
        opacity: 0.70;
        margin-top: 0.25rem;
    }
    .status-good { color: #0fa968; font-weight: 800; }
    .status-mid { color: #d49b00; font-weight: 800; }
    .status-low { color: #d64545; font-weight: 800; }
    </style>
    """,
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, hint: str = ""):
    st.markdown(
        f"""
        <div class="solution-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def confidence_class(label: str) -> str:
    if label.startswith("HIGH"):
        return "status-good"
    if label.startswith("MEDIUM"):
        return "status-mid"
    return "status-low"


def make_2d_map(dem, best_points, truth_points=None, center=None, current_idx=None, title="DEM + траектория"):
    fig, ax = plt.subplots(figsize=(8.5, 7.4))
    img = ax.imshow(dem, cmap="terrain")
    plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="Высота, м")

    if truth_points is not None:
        ax.plot(truth_points[:, 0], truth_points[:, 1], linewidth=2.5, label="Контрольная траектория")
    ax.plot(best_points[:, 0], best_points[:, 1], linewidth=2.8, linestyle="--", label="Найденная траектория")
    ax.scatter([best_points[-1, 0]], [best_points[-1, 1]], s=95, marker="o", label="Текущая позиция")
    if current_idx is not None:
        ax.scatter([best_points[current_idx, 0]], [best_points[current_idx, 1]], s=110, marker="D", label="Позиция окна")
    if center is not None:
        ax.scatter([center[0]], [center[1]], marker="x", s=85, label="Центр поиска")

    ax.set_xlabel("X, пиксели")
    ax.set_ylabel("Y, пиксели")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.15)
    return fig


def _nearest_dem_values(dem, x_values, y_values):
    h, w = dem.shape
    xs = np.clip(np.rint(np.asarray(x_values, dtype=float)).astype(int), 0, w - 1)
    ys = np.clip(np.rint(np.asarray(y_values, dtype=float)).astype(int), 0, h - 1)
    return dem[ys, xs].astype(float)


def _corr_rgba(value, alpha=0.75):
    v = float(np.clip((value + 1.0) / 2.0, 0.0, 1.0))
    r = int(40 + 215 * v)
    g = int(85 + 140 * v)
    b = int(210 - 160 * v)
    return f"rgba({r},{g},{b},{alpha})"


def _select_top_heatmap_candidates(heatmap, azimuths, shifts, top_n=5, min_az_sep_deg=4.0, min_shift_sep_px=6.0):
    valid = np.argwhere(np.isfinite(heatmap))
    if len(valid) == 0:
        return []

    values = heatmap[valid[:, 0], valid[:, 1]]
    order = np.argsort(values)[::-1]
    selected = []

    for idx in order:
        ai, si = valid[idx]
        corr = float(heatmap[ai, si])
        az = float(azimuths[ai])
        shift = float(shifts[si])
        if corr < -0.99:
            continue

        too_close = False
        for item in selected:
            da = abs((az - item["azimuth_deg"] + 180.0) % 360.0 - 180.0)
            ds = abs(shift - item["shift_px"])
            if da < min_az_sep_deg and ds < min_shift_sep_px:
                too_close = True
                break
        if too_close:
            continue

        selected.append({"azimuth_deg": az, "shift_px": shift, "corr": corr})
        if len(selected) >= top_n:
            break

    return selected


def make_3d_terrain_matching_scene(
    dem,
    best,
    heatmap,
    azimuths,
    shifts,
    observed_profile,
    pixel_size_m,
    accuracy=None,
    x0=None,
    y0=None,
    truth_points=None,
    truth_profile=None,
    max_grid=135,
):
    h, w = dem.shape
    step = max(1, int(max(h, w) / max_grid))
    dem_small = dem[::step, ::step]
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)

    relief_range = float(np.nanmax(dem) - np.nanmin(dem))
    z_offset = max(22.0, 0.055 * relief_range)
    n_points = len(best.ref_profile) if best.ref_profile is not None else len(observed_profile)
    start_x = float(best.start_x_px if best.start_x_px is not None else (x0 if x0 is not None else w / 2.0))
    start_y = float(best.start_y_px if best.start_y_px is not None else (y0 if y0 is not None else h / 2.0))
    ds_px = float(best.ds_px if best.ds_px is not None else 1.0)
    best_shift = float(best.shift_px if best.shift_px is not None else 0.0)
    best_az = float(best.azimuth_deg if best.azimuth_deg is not None else 0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=xs,
            y=ys,
            z=dem_small,
            colorscale="Earth",
            opacity=0.86,
            showscale=True,
            colorbar=dict(title="Высота, м"),
            name="DEM",
        )
    )


    az_corr = np.max(heatmap, axis=1) if heatmap is not None and len(heatmap) else np.asarray([])
    finite = az_corr[np.isfinite(az_corr)]
    corr_min = float(np.min(finite)) if len(finite) else -1.0
    corr_max = float(np.max(finite)) if len(finite) else 1.0
    fan_stride = max(1, int(math.ceil(max(len(azimuths), 1) / 40)))
    fan_radius = min(0.48 * min(h, w), max(best_shift + max(n_points - 1, 1) * ds_px, 35.0))

    for idx in range(0, len(azimuths), fan_stride):
        az = float(azimuths[idx])
        corr = float(az_corr[idx]) if len(az_corr) else -1.0
        quality = 0.0 if corr_max <= corr_min else (corr - corr_min) / (corr_max - corr_min)
        theta = math.radians(az)
        rr = np.linspace(0.0, fan_radius, 70)
        xx = np.clip(start_x + rr * math.sin(theta), 0, w - 1)
        yy = np.clip(start_y - rr * math.cos(theta), 0, h - 1)
        zz = _nearest_dem_values(dem, xx, yy) + z_offset * (0.85 + 0.25 * quality)
        fig.add_trace(
            go.Scatter3d(
                x=xx,
                y=yy,
                z=zz,
                mode="lines",
                line=dict(width=1.2 + 4.2 * quality, color=_corr_rgba(corr, alpha=0.42 + 0.38 * quality)),
                opacity=0.55 + 0.35 * quality,
                name="Веер направлений" if idx == 0 else None,
                showlegend=(idx == 0),
                hovertemplate=f"Азимут: {az:.0f}°<br>Max corr: {corr:.4f}<extra></extra>",
            )
        )


    top_candidates = _select_top_heatmap_candidates(
        heatmap,
        azimuths,
        shifts,
        top_n=6,
        min_az_sep_deg=max(3.0, float(np.mean(np.diff(azimuths))) * 1.5 if len(azimuths) > 1 else 3.0),
        min_shift_sep_px=max(5.0, float(np.mean(np.diff(shifts))) * 1.5 if len(shifts) > 1 else 5.0),
    )

    for rank, cand in enumerate(reversed(top_candidates[1:]), start=2):
        prof, pts = sample_profile(
            dem=dem,
            x0=start_x,
            y0=start_y,
            azimuth_deg=cand["azimuth_deg"],
            ds_px=ds_px,
            n_points=n_points,
            shift_px=cand["shift_px"],
        )
        if prof is None or pts is None:
            continue
        fig.add_trace(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=np.asarray(prof) + z_offset * 1.35,
                mode="lines",
                line=dict(width=3, color=_corr_rgba(cand["corr"], alpha=0.48)),
                opacity=0.52,
                name="Альтернативные кандидаты" if rank == 2 else None,
                showlegend=(rank == 2),
                hovertemplate=(
                    f"Кандидат<br>az: {cand['azimuth_deg']:.0f}°"
                    f"<br>shift: {cand['shift_px']:.0f} px"
                    f"<br>corr: {cand['corr']:.4f}<extra></extra>"
                ),
            )
        )


    fig.add_trace(
        go.Scatter3d(
            x=best.points[:, 0],
            y=best.points[:, 1],
            z=np.asarray(best.ref_profile) + z_offset * 1.75,
            mode="lines+markers",
            line=dict(width=8, color="rgba(255,235,90,0.98)"),
            marker=dict(size=3, color="rgba(255,255,210,0.95)"),
            name="Лучшее совпадение",
            hovertemplate="Best match<br>x: %{x:.1f}<br>y: %{y:.1f}<br>terrain: %{z:.1f}<extra></extra>",
        )
    )

    if truth_points is not None and truth_profile is not None:
        fig.add_trace(
            go.Scatter3d(
                x=truth_points[:, 0],
                y=truth_points[:, 1],
                z=np.asarray(truth_profile) + z_offset * 2.05,
                mode="lines",
                line=dict(width=4, dash="dash", color="rgba(230,230,230,0.72)"),
                name="Контрольная траектория",
            )
        )


    current_x = float(best.points[-1, 0])
    current_y = float(best.points[-1, 1])
    current_z = float(best.ref_profile[-1]) + z_offset * 2.25
    fig.add_trace(
        go.Scatter3d(
            x=[current_x],
            y=[current_y],
            z=[current_z],
            mode="markers+text",
            marker=dict(size=8, color="rgba(255,255,255,1.0)"),
            text=["CURRENT POSITION"],
            textposition="top center",
            name="Текущая позиция",
        )
    )


    if accuracy and pixel_size_m > 0:
        radius_px = float(accuracy.get("estimated_radius_m", 0.0)) / float(pixel_size_m)
        if np.isfinite(radius_px) and radius_px > 0:
            radius_px = min(radius_px, 0.32 * min(h, w))
            tt = np.linspace(0, 2 * np.pi, 160)
            ring_x = np.clip(current_x + radius_px * np.cos(tt), 0, w - 1)
            ring_y = np.clip(current_y + radius_px * np.sin(tt), 0, h - 1)
            ring_z = np.full_like(ring_x, current_z - z_offset * 0.22)
            fig.add_trace(
                go.Scatter3d(
                    x=ring_x,
                    y=ring_y,
                    z=ring_z,
                    mode="lines",
                    line=dict(width=5, color="rgba(255,255,255,0.58)"),
                    name=f"Зона ошибки R≈{accuracy.get('estimated_radius_m', 0.0):.0f} м",
                )
            )


    az_rad = math.radians(best_az)
    vector_scale = max(8.0, 0.045 * min(h, w))
    fig.add_trace(
        go.Cone(
            x=[current_x],
            y=[current_y],
            z=[current_z + z_offset * 0.12],
            u=[math.sin(az_rad) * vector_scale],
            v=[-math.cos(az_rad) * vector_scale],
            w=[0.0],
            sizemode="absolute",
            sizeref=max(8.0, vector_scale * 0.9),
            anchor="tail",
            colorscale=[[0, "rgb(255,210,80)"], [1, "rgb(255,245,170)"]],
            showscale=False,
            name="Вектор движения",
        )
    )


    if observed_profile is not None and best.ref_profile is not None:
        lateral = max(4.0, 0.018 * min(h, w))
        normal_x = math.cos(az_rad)
        normal_y = math.sin(az_rad)
        obs = np.asarray(observed_profile, dtype=float)
        ref = np.asarray(best.ref_profile, dtype=float)
        z_base = np.nanmin(dem) + relief_range * 0.10

        ref_z = ref + z_offset * 2.65
        obs_z = obs + z_offset * 2.92
        fp_x = np.clip(best.points[:, 0] + normal_x * lateral, 0, w - 1)
        fp_y = np.clip(best.points[:, 1] + normal_y * lateral, 0, h - 1)
        fp2_x = np.clip(best.points[:, 0] + normal_x * lateral * 2.0, 0, w - 1)
        fp2_y = np.clip(best.points[:, 1] + normal_y * lateral * 2.0, 0, h - 1)
        fig.add_trace(
            go.Scatter3d(
                x=fp_x,
                y=fp_y,
                z=ref_z,
                mode="lines",
                line=dict(width=5, color="rgba(90,210,255,0.92)"),
                name="DEM fingerprint",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=fp2_x,
                y=fp2_y,
                z=obs_z,
                mode="lines",
                line=dict(width=5, color="rgba(255,120,210,0.88)", dash="dash"),
                name="Radio fingerprint",
            )
        )

    title = (
        f"3D: corr={best.corr:.4f}, az={best_az:.0f}°, "
        " "
        f"speed={float(best.speed_mps or 0.0):.1f} м/с"
    )
    fig.update_layout(
        height=690,
        margin=dict(l=0, r=0, t=45, b=0),
        title=title,
        scene=dict(
            xaxis_title="X, px",
            yaxis_title="Y, px",
            zaxis_title="Высота, м",
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=max(0.55, h / max(w, 1)), z=0.36),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_3d_search_preview(dem, x0, y0, observed_profile, pixel_size_m, max_shift_px, max_grid=105):
    h, w = dem.shape
    step = max(1, int(max(h, w) / max_grid))
    dem_small = dem[::step, ::step]
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)

    relief_range = float(np.nanmax(dem) - np.nanmin(dem))
    z_offset = max(18.0, 0.04 * relief_range)
    center_x = float(np.clip(x0, 0, w - 1))
    center_y = float(np.clip(y0, 0, h - 1))
    center_z = float(dem[int(round(center_y)), int(round(center_x))]) + z_offset * 1.4

    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=xs,
            y=ys,
            z=dem_small,
            colorscale="Earth",
            opacity=0.88,
            showscale=True,
            colorbar=dict(title="Высота, м"),
            name="DEM",
        )
    )

    profile_len_px = len(observed_profile) * max(float(pixel_size_m), 1.0) / max(float(pixel_size_m), 1.0)
    search_radius = float(min(max_shift_px + profile_len_px, 0.46 * min(h, w)))
    for az in range(0, 360, 15):
        theta = math.radians(az)
        rr = np.linspace(0.0, search_radius, 80)
        xx = np.clip(center_x + rr * math.sin(theta), 0, w - 1)
        yy = np.clip(center_y - rr * math.cos(theta), 0, h - 1)
        zi = _nearest_dem_values(dem, xx, yy) + z_offset
        opacity = 0.23 if az % 45 else 0.48
        width = 2 if az % 45 else 4
        fig.add_trace(
            go.Scatter3d(
                x=xx,
                y=yy,
                z=zi,
                mode="lines",
                line=dict(width=width, color=f"rgba(130,190,255,{opacity})"),
                name="Проверяемые направления" if az == 0 else None,
                showlegend=(az == 0),
            )
        )


    rr = np.linspace(0.0, min(profile_len_px, search_radius), 80)
    xx = np.clip(center_x + rr, 0, w - 1)
    yy = np.full_like(xx, center_y)
    zz = _nearest_dem_values(dem, xx, yy) + z_offset * 1.45
    fig.add_trace(
        go.Scatter3d(
            x=xx,
            y=yy,
            z=zz,
            mode="lines",
            line=dict(width=6, color="rgba(255,235,120,0.86)"),
            name="Длина профиля",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[center_x],
            y=[center_y],
            z=[center_z],
            mode="markers+text",
            marker=dict(size=7, color="rgba(255,255,255,0.95)"),
            text=["центр поиска"],
            textposition="top center",
            name="Центр поиска",
        )
    )

    fig.update_layout(
        height=560,
        margin=dict(l=0, r=0, t=35, b=0),
        title="3D Terrain Matching: область и направления корреляционного поиска",
        scene=dict(
            xaxis_title="X, px",
            yaxis_title="Y, px",
            zaxis_title="Высота, м",
            aspectmode="manual",
            aspectratio=dict(x=1.0, y=max(0.55, h / max(w, 1)), z=0.34),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def run_search(
    dem,
    observed_profile,
    x0,
    y0,
    dt,
    pixel_size_m,
    azimuth_step,
    max_shift,
    shift_step,
    speed_candidates,
    start_radius,
    start_step,
    azimuth_candidates=None,
):
    estimated_iterations = (
        len(speed_candidates)
        * (len(azimuth_candidates) if azimuth_candidates is not None else len(np.arange(0, 360, int(azimuth_step))))
        * len(np.arange(0, int(max_shift) + 1, int(shift_step)))
    )
    if start_radius > 0:
        approx_start_cells = max(1, int((2 * start_radius / start_step + 1) ** 2))
        estimated_iterations *= approx_start_cells

    if estimated_iterations > 900_000:
        st.warning(
            f"Поиск тяжёлый: примерно {estimated_iterations:,} кандидатов. "
            "Для ускорения расчёта можно увеличить шаг скорости/shift или отключить расширенный поиск стартовой точки.".replace(",", " ")
        )

    progress = st.progress(0.0, text="Идёт корреляционный поиск по DEM...")
    try:
        return search_by_correlation(
            dem=dem,
            observed_profile=observed_profile,
            x0=x0,
            y0=y0,
            dt=dt,
            pixel_size_m=pixel_size_m,
            azimuth_step_deg=int(azimuth_step),
            max_shift_px=int(max_shift),
            shift_step_px=int(shift_step),
            speed_candidates_mps=speed_candidates,
            start_search_radius_px=int(start_radius),
            start_search_step_px=int(start_step),
            azimuth_candidates_deg=azimuth_candidates,
            progress_callback=lambda value: progress.progress(value, text="Идёт корреляционный поиск по DEM..."),
        )
    finally:
        progress.empty()


st.markdown(
    """
    <div class="main-title">BlindFlight TerrainNav</div>
    <div class='title'>Команда: СКБ "ФИЗ-ТЕХ" 5</div>
    <div class="subtitle">Участники команды: Сафронов Евгений, Порозов Вячеслав, Задоя Маргарита, Шишков Дмитрий</div>
    """,
    unsafe_allow_html=True,
)


st.sidebar.header("1. Режим")
preset = st.sidebar.selectbox(
    "Пресет поиска",
    ["Проверка", "Быстрый режим", "Азимут 1°", "Точный режим"],
    index=0,
    help=(
        "Готовый пресет для проведения проверки на третьем чек-поинте. "
    ),
)
checkpoint_preset = preset == "Проверка"

work_mode = st.sidebar.radio(
    "Что запускаем?",
    ["Контрольный профиль", "Анализ NMEA"],
    index=1 if checkpoint_preset else 0,
    disabled=checkpoint_preset,
    help=(
        "Симуляция используется для проверки ошибки на известной траектории. "
        "В пресете «Проверка» включается анализ внешнего файла высот для чекпоинта."
    ),
)

if preset == "Проверка":
    default_az_step, default_shift_step, default_speed_step = 1, 1, 1.0
elif preset == "Азимут 1°":
    default_az_step, default_shift_step, default_speed_step = 1, 5, 5.0
elif preset == "Точный режим":
    default_az_step, default_shift_step, default_speed_step = 1, 2, 2.0
else:
    default_az_step, default_shift_step, default_speed_step = 2, 4, 5.0

st.sidebar.header("2. Карта рельефа")
source_mode = st.sidebar.radio(
    "Источник DEM",
    ["Синтетическая DEM", "Загрузить GeoTIFF"],
    index=1 if checkpoint_preset else 0,
)

uploaded_dem = None
synthetic_pixel_size = 30.0

if source_mode == "Загрузить GeoTIFF":
    uploaded_dem = st.sidebar.file_uploader("DEM GeoTIFF (.tif/.tiff)", type=["tif", "tiff"])
    dem_h, dem_w, dem_seed = 500, 500, 42
else:
    dem_h = st.sidebar.slider("Высота DEM, пиксели", 250, 1200, 500, 50)
    dem_w = st.sidebar.slider("Ширина DEM, пиксели", 250, 1200, 500, 50)
    dem_seed = st.sidebar.number_input("Seed DEM", min_value=0, max_value=9999, value=42, step=1)
    synthetic_pixel_size = st.sidebar.number_input(
        "Размер пикселя синтетической DEM, м/пиксель",
        min_value=1.0,
        max_value=200.0,
        value=30.0,
        step=1.0,
    )

st.sidebar.header("3. Датчик")
h_abs = st.sidebar.slider("Барометрическая высота H_abs, м", 500.0, 3000.0, 1500.0, 10.0)
manual_dt = st.sidebar.slider("Период измерений dt, с", 0.1, 2.0, 0.5, 0.1)
use_kalman = st.sidebar.checkbox("Сгладить профиль фильтром Калмана", value=False)
kalman_measurement_var = st.sidebar.slider("Дисперсия измерений для Калмана", 0.5, 100.0, 9.0, 0.5)

if work_mode == "Контрольный профиль":
    st.sidebar.header("4. Контрольная траектория")
    truth_speed_mps = st.sidebar.slider("Контрольная скорость, м/с", 10.0, 120.0, 40.0, 1.0)
    n_points = st.sidebar.slider("Длина окна профиля, точек", 30, 350, 120, 10)
    truth_azimuth = st.sidebar.slider("Контрольный азимут, °", 0, 359, 38, 1)
    truth_shift = st.sidebar.slider("Контрольное начальное смещение, пиксели", 0, 500, 40, 1)
    noise_std = st.sidebar.slider("Шум радиовысотомера, м", 0.0, 25.0, 2.0, 0.5)
    sim_seed = st.sidebar.number_input("Seed шума", min_value=0, max_value=9999, value=1, step=1)
    uploaded_nmea = None
    use_auto_dt = False
else:
    st.sidebar.header("4. Входные высоты")
    input_format = st.sidebar.selectbox(
        "Формат файла",
        ["NMEA GGA", "Простой TXT с высотами"],
        index=1 if checkpoint_preset else 0,
        help="Для третьего чекпоинта удобнее выбрать простой TXT: одно значение высоты на строку или через заданный разделитель.",
    )
    uploaded_nmea = st.sidebar.file_uploader("Файл с высотами / radio.nmea / .txt", type=["nmea", "txt", "log", "csv"])
    use_auto_dt = st.sidebar.checkbox("Пробовать взять dt из NMEA-времени", value=True, disabled=(input_format != "NMEA GGA"))
    text_delimiter = st.sidebar.text_input("Разделитель TXT", value="auto", help="auto, \n, ;, пробел, запятая или любой строковый разделитель")
    heights_kind = st.sidebar.selectbox(
        "Что лежит в файле",
        ["Радиовысота H_radio", "Высота рельефа H_terrain"],
        help="Если организаторы дают именно данные радиовысотомера, оставьте H_radio. Тогда применяется H_terrain = H_abs - H_radio.",
    )
    n_points = None
    truth_speed_mps = None
    truth_azimuth = None
    truth_shift = None
    noise_std = None
    sim_seed = None

st.sidebar.header("5. Поиск")
search_speed = st.sidebar.checkbox("Искать скорость автоматически", value=True)
if search_speed:
    sp_min = st.sidebar.number_input("Минимальная скорость, м/с", min_value=1.0, max_value=300.0, value=20.0, step=1.0)
    sp_max = st.sidebar.number_input("Максимальная скорость, м/с", min_value=1.0, max_value=300.0, value=80.0, step=1.0)
    sp_step = st.sidebar.number_input("Шаг скорости, м/с", min_value=1.0, max_value=50.0, value=float(default_speed_step), step=1.0)
else:
    fixed_speed = st.sidebar.slider("Фиксированная скорость, м/с", 5.0, 150.0, 40.0, 1.0)
    sp_step = 0.0

azimuth_step = st.sidebar.slider("Шаг азимута, °", 1, 20, int(default_az_step), 1)
default_max_shift = 0 if (checkpoint_preset or work_mode == "Анализ NMEA") else 220
max_shift = st.sidebar.slider("Максимальное смещение, пиксели", 0, 600, default_max_shift, 5)
shift_step = st.sidebar.slider("Шаг смещения, пиксели", 1, 30, int(default_shift_step), 1)

with st.sidebar.expander("Расширенный поиск стартовой точки"):
    start_radius = st.slider(
        "Радиус вокруг центра, пиксели",
        0,
        200,
        0,
        10,
        help="0 = стартовая точка привязана к центру карты. Больше 0 = перебор стартовых точек вокруг центра, но расчёт тяжелее.",
    )
    start_step = st.slider("Шаг сетки стартовой точки, пиксели", 10, 100, 40, 10)

st.sidebar.header("6. Данные чекпоинта")
start_mode = st.sidebar.radio(
    "Стартовая точка",
    ["Центр карты", "Задать x/y"],
    index=1 if checkpoint_preset else 0,
    help="На третьем чекпоинте организаторы дают начальную точку x,y — выберите «Задать x/y».",
)
coord_mode = st.sidebar.selectbox("Система координат x/y", ["Пиксели DEM", "Координаты карты GeoTIFF"], disabled=(start_mode == "Центр карты"))
input_start_x = st.sidebar.number_input("start x", value=250.0, step=1.0, disabled=(start_mode == "Центр карты"))
input_start_y = st.sidebar.number_input("start y", value=250.0, step=1.0, disabled=(start_mode == "Центр карты"))
use_known_azimuth = st.sidebar.checkbox(
    "Использовать известное направление",
    value=(checkpoint_preset or work_mode == "Анализ NMEA"),
    help="Если организаторы дали направление относительно севера, включите этот режим.",
)
known_azimuth = st.sidebar.number_input("Направление, ° от севера", min_value=0.0, max_value=359.999, value=0.0, step=1.0, disabled=not use_known_azimuth)
azimuth_tolerance = st.sidebar.number_input("Допуск по направлению, ±°", min_value=0.0, max_value=180.0, value=0.0, step=1.0, disabled=not use_known_azimuth)

use_auto_pixel_size = st.sidebar.checkbox(
    "Авторазмер пикселя из GeoTIFF",
    value=checkpoint_preset,
    disabled=(source_mode != "Загрузить GeoTIFF"),
    help="В пресете «Проверка» лучше брать размер пикселя из GeoTIFF. Если карта в градусах, оценка будет грубой — лучше UTM.",
)
manual_pixel_size = st.sidebar.number_input(
    "Размер пикселя для расчёта, м/пиксель",
    min_value=0.01,
    max_value=2000.0,
    value=30.0,
    step=1.0,
    disabled=use_auto_pixel_size and source_mode == "Загрузить GeoTIFF",
    help="Для реальной DEM лучше перепроецировать карту в UTM, тогда пиксели будут в метрах.",
)

if checkpoint_preset:
    run_label = "Запустить проверку чекпоинта"
else:
    run_label = "Сформировать профиль и найти траекторию" if work_mode == "Контрольный профиль" else "Прочитать NMEA и найти траекторию"
run = st.sidebar.button(run_label, type="primary", use_container_width=True)

if not run:
    if checkpoint_preset:
        st.info(
            "Включен пресет, для проверки данных, полученных на чек-поинте."
        )
    else:
        st.info("Задайте параметры слева и нажмите кнопку запуска.")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("1", "DEM", "карта рельефа GeoTIFF или синтетика")
    with c2:
        metric_card("2", "NMEA", "радиовысотомер 1–10 Гц")
    with c3:
        metric_card("3", "NAV SOLUTION", "x/y, скорость, азимут, доверие")
    st.stop()


try:
    if source_mode == "Загрузить GeoTIFF":
        if uploaded_dem is None:
            st.warning("Загрузите DEM GeoTIFF.")
            st.stop()
        dem, transform, crs = load_dem_from_bytes(uploaded_dem.read())
    else:
        dem, transform, crs = generate_synthetic_dem(
            height=int(dem_h), width=int(dem_w), seed=int(dem_seed), pixel_size_m=float(synthetic_pixel_size)
        )
    dem = clean_dem(dem)
except Exception as e:
    st.error(f"Ошибка загрузки DEM: {e}")
    st.stop()

height, width = dem.shape
if start_mode == "Задать x/y":
    if coord_mode == "Координаты карты GeoTIFF":
        x0, y0 = map_to_pixel(float(input_start_x), float(input_start_y), transform)
    else:
        x0, y0 = float(input_start_x), float(input_start_y)
else:
    x0 = width / 2.0
    y0 = height / 2.0
auto_px = estimate_pixel_size_m(transform, crs)
if use_auto_pixel_size and source_mode == "Загрузить GeoTIFF":
    pixel_size_m = float(auto_px)
else:
    pixel_size_m = float(manual_pixel_size)

if not (0 <= x0 < width and 0 <= y0 < height):
    st.error(f"Стартовая точка вне DEM: x={x0:.2f}, y={y0:.2f}, размер карты {width}×{height} px.")
    st.stop()

if source_mode == "Синтетическая DEM" and abs(pixel_size_m - synthetic_pixel_size) > 1e-6:
    st.warning("Размер пикселя для расчёта отличается от размера пикселя синтетической DEM. Для честного теста лучше оставить их одинаковыми.")

if source_mode == "Загрузить GeoTIFF" and crs is not None and getattr(crs, "is_geographic", False):
    st.warning("GeoTIFF в градусах. Для точной скорости и ошибок лучше перепроецировать DEM в UTM и экспортировать GeoTIFF.")

if checkpoint_preset:
    st.caption(
        f"Пресет «Проверка»: старт ({x0:.1f}, {y0:.1f}) px · "
        f"направление {'задано' if use_known_azimuth else 'не задано'} · "
        f"размер пикселя {pixel_size_m:.3f} м/px."
    )

sim = None
truth_pts = None
nmea_text = ""
raw_observed_profile = None
radio_profile = None
dt = float(manual_dt)
dt_est = None

try:
    if work_mode == "Контрольный профиль":
        sim = simulate_flight(
            dem=dem,
            x0=x0,
            y0=y0,
            azimuth_deg=float(truth_azimuth),
            speed_mps=float(truth_speed_mps),
            dt=dt,
            n_points=int(n_points),
            shift_px=float(truth_shift),
            pixel_size_m=pixel_size_m,
            h_abs=float(h_abs),
            noise_std_m=float(noise_std),
            seed=int(sim_seed),
        )
        raw_observed_profile = sim["terrain_measured"]
        radio_profile = sim["radio_profile"]
        truth_pts = sim["truth_points"]
        nmea_text = "\n".join(sim["nmea_lines"])
    else:
        if uploaded_nmea is None:
            st.warning("Загрузите файл с высотами или NMEA.")
            st.stop()
        nmea_text = uploaded_nmea.read().decode("utf-8", errors="ignore")
        if input_format == "NMEA GGA":
            radio_profile, dt_est = parse_nmea_profile_with_dt(nmea_text)
            if len(radio_profile) == 0:
                st.error("В NMEA не удалось найти GGA-строки с высотой в поле altitude.")
                st.stop()
            if use_auto_dt and dt_est is not None:
                dt = float(dt_est)
            raw_observed_profile = terrain_from_radio(float(h_abs), radio_profile)
        else:
            plain_heights = parse_plain_heights_text(nmea_text, delimiter=text_delimiter)
            if len(plain_heights) == 0:
                st.error("В TXT не удалось найти числовые высоты. Проверьте разделитель и формат файла.")
                st.stop()
            height_kind = "radio" if heights_kind.startswith("Радиовысота") else "terrain"
            raw_observed_profile = heights_to_terrain_profile(plain_heights, h_abs=float(h_abs), height_kind=height_kind)
            radio_profile = plain_heights if height_kind == "radio" else None
        n_points = len(raw_observed_profile)
except Exception as e:
    st.error(str(e))
    st.stop()

observed_profile = (
    kalman_1d(raw_observed_profile, process_var=1.0, measurement_var=float(kalman_measurement_var))
    if use_kalman
    else raw_observed_profile
)

if search_speed:
    if sp_max < sp_min:
        st.error("Максимальная скорость меньше минимальной.")
        st.stop()
    speed_candidates = np.arange(float(sp_min), float(sp_max) + float(sp_step) * 0.5, float(sp_step), dtype=np.float32)
else:
    speed_candidates = np.asarray([float(fixed_speed)], dtype=np.float32)

azimuth_candidates = None
if use_known_azimuth:
    azimuth_candidates = build_azimuth_candidates(float(known_azimuth), float(azimuth_tolerance), float(azimuth_step))

search_preview = st.empty()
with search_preview.container():
    st.markdown("### 3D")
    st.caption("Во время расчёта отображается геометрия поиска: веер направлений, длина профиля и область сопоставления с DEM.")
    pv_left, pv_right = st.columns([1.45, 0.75])
    with pv_left:
        st.plotly_chart(
            make_3d_search_preview(
                dem,
                x0,
                y0,
                observed_profile,
                pixel_size_m,
                max_shift_px=int(max_shift),
            ),
            use_container_width=True,
            key="active_search_3d_terrain",
        )
    with pv_right:
        metric_card("DEM", f"{width}×{height} px", f"{pixel_size_m:.1f} м/px")
        metric_card("Профиль", f"{len(observed_profile)} точек", f"dt {dt:.2f} с")
        metric_card("Перебор", f"{len(azimuth_candidates) if azimuth_candidates is not None else int(360 / azimuth_step)} az / {shift_step} px", "азимуты / смещение")
        metric_card("Скорости", f"{float(speed_candidates[0]):.0f}–{float(speed_candidates[-1]):.0f} м/с", f"{len(speed_candidates)} кандидатов")

best, heatmap, azimuths, shifts = run_search(
    dem,
    observed_profile,
    x0,
    y0,
    dt,
    pixel_size_m,
    azimuth_step,
    max_shift,
    shift_step,
    speed_candidates,
    start_radius,
    start_step,
    azimuth_candidates=azimuth_candidates,
)
search_preview.empty()

if best.points is None:
    st.error("Маршрут не найден. Увеличьте max_shift, уменьшите длину окна/скорость или проверьте размер пикселя.")
    st.stop()

label = confidence_label(best, observed_profile)
vel = velocity_components(best.speed_mps, best.azimuth_deg)
accuracy = estimate_accuracy_radius_m(
    best,
    observed_profile,
    pixel_size_m,
    azimuth_step_deg=float(azimuth_step),
    shift_step_px=float(shift_step),
    speed_step_mps=float(sp_step if search_speed else 0.0),
    dt=float(dt),
)
nmea_stats = nmea_quality_report(nmea_text)

start_pt = best.points[0:1]
end_pt = best.points[-1:]
start_map_x, start_map_y = pixel_to_map(start_pt, transform)
end_map_x, end_map_y = pixel_to_map(end_pt, transform)
start_lon, start_lat = pixel_to_lonlat(start_pt, transform, crs)
end_lon, end_lat = pixel_to_lonlat(end_pt, transform, crs)

err = {}
az_err = None
sp_err = None
if work_mode == "Контрольный профиль":
    err = trajectory_error_metrics(truth_pts, best.points, pixel_size_m)
    az_err = circular_angle_error_deg(float(truth_azimuth), float(best.azimuth_deg))
    sp_err = abs(float(truth_speed_mps) - float(best.speed_mps))


status_html = f'<span class="{confidence_class(label)}">{label}</span>'
st.markdown(f"### Навигационное решение: {status_html}", unsafe_allow_html=True)

r1, r2, r3, r4 = st.columns(4)
with r1:
    metric_card("Текущая позиция X/Y", f"{best.points[-1,0]:.1f} / {best.points[-1,1]:.1f} px", )
with r2:
    metric_card("Координаты карты", f"{end_map_x[0]:.1f} / {end_map_y[0]:.1f}")
with r3:
    metric_card("Скорость / азимут", f"{best.speed_mps:.1f} м/с · {best.azimuth_deg:.0f}°", )
with r4:
    metric_card("Оценка ошибки", f"R≈{accuracy['estimated_radius_m']:.0f} м", )

r5, r6, r7, r8 = st.columns(4)
with r5:
    metric_card("Вектор скорости", f"E {vel['v_east_mps']:.1f} / N {vel['v_north_mps']:.1f}", )
with r6:
    metric_card("Корреляция", f"{best.corr:.4f}", f"gap: {best.confidence_gap:.5f}")
with r7:
    metric_card("Профиль", f"{len(observed_profile)} точек", f"перепад: {accuracy['relief_range_m']:.1f} м")
with r8:
    metric_card("NMEA", f"{nmea_stats['altitude_values']} высот", f"checksum bad: {nmea_stats['checksum_bad']}")

if work_mode == "Контрольный профиль":
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Ошибка азимута", f"{az_err:.1f}°", f"контроль {truth_azimuth}°")
    e2.metric("Ошибка скорости", f"{sp_err:.1f} м/с", f"контроль {truth_speed_mps:.1f} м/с")
    e3.metric("Средняя ошибка", f"{err.get('mean_error_m', 0):.1f} м")
    e4.metric("Ошибка конца", f"{err.get('end_error_m', 0):.1f} м")


tab_map, tab_3d, tab_corr, tab_nmea, tab_biz = st.tabs(
    ["Карта", "3D", "Корреляция", "NMEA / экспорт", "Применение"]
)

with tab_map:
    left, right = st.columns([1.15, 1.0])
    with left:
        st.subheader("DEM + найденная траектория")
        st.pyplot(make_2d_map(dem, best.points, truth_points=truth_pts, center=(x0, y0)))
    with right:
        st.subheader("Профили высот")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        if sim is not None:
            ax.plot(sim["terrain_true"], label="Контрольный профиль DEM")
        ax.plot(raw_observed_profile, label="Профиль из радиовысотомера", alpha=0.75)
        if use_kalman:
            ax.plot(observed_profile, label="После фильтра Калмана")
        ax.plot(best.ref_profile, linestyle="--", label="Лучший профиль из DEM")
        ax.set_xlabel("Номер измерения")
        ax.set_ylabel("Высота рельефа, м")
        ax.set_title("Сопоставление профилей")
        ax.grid(alpha=0.3)
        ax.legend()
        st.pyplot(fig)

        with st.expander("Скользящее окно обработки"):
            k = st.slider("Сколько измерений уже получено", 5, len(best.points), len(best.points), 1)
            st.caption("Логика обработки: по мере накопления NMEA-точек текущая позиция обновляется по последнему устойчивому окну профиля.")
            st.pyplot(make_2d_map(dem, best.points[:k], truth_points=truth_pts[:k] if truth_pts is not None else None, center=(x0, y0), current_idx=k - 1, title="Скользящее окно"))

with tab_3d:
    st.plotly_chart(
        make_3d_terrain_matching_scene(
            dem=dem,
            best=best,
            heatmap=heatmap,
            azimuths=azimuths,
            shifts=shifts,
            observed_profile=observed_profile,
            pixel_size_m=pixel_size_m,
            accuracy=accuracy,
            x0=x0,
            y0=y0,
            truth_points=truth_pts,
            truth_profile=sim["terrain_true"] if sim is not None else None,
        ),
        use_container_width=True,
    )
    st.caption("3D-сцена показывает процесс terrain matching: веер проверенных направлений, альтернативные кандидаты, лучший профиль, текущую позицию и радиус ошибки.")

with tab_corr:
    st.subheader("Корреляционная карта")
    fig, ax = plt.subplots(figsize=(11, 5.5))
    valid_heat = heatmap[np.isfinite(heatmap)]
    vmin = max(-1, float(np.min(valid_heat))) if len(valid_heat) else -1
    vmax = float(np.max(valid_heat)) if len(valid_heat) else 1
    im = ax.imshow(
        heatmap,
        aspect="auto",
        origin="lower",
        extent=[float(shifts[0]), float(shifts[-1]), float(azimuths[0]), float(azimuths[-1])],
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label="Коэффициент корреляции")
    ax.scatter([best.shift_px], [best.azimuth_deg], s=90, marker="x", label="Лучший кандидат")
    ax.set_xlabel("Смещение, пиксели")
    ax.set_ylabel("Азимут, °")
    ax.set_title("Heatmap: максимум корреляции по азимуту и смещению")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Максимальная корреляция по азимуту")
    az_corr = np.max(heatmap, axis=1)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(azimuths, az_corr)
    if truth_azimuth is not None:
        ax.axvline(truth_azimuth, linestyle=":", label="Контрольный азимут")
    ax.axvline(best.azimuth_deg, linestyle="--", label="Найденный азимут")
    ax.set_xlabel("Азимут, °")
    ax.set_ylabel("Max corr")
    ax.set_title("Пик графика показывает найденное направление")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)

    with st.expander("Из чего состоит оценка ошибки"):
        st.json(accuracy)

with tab_nmea:
    st.subheader("Качество NMEA-0183")
    q1, q2, q3, q4, q5 = st.columns(5)
    q1.metric("Всего строк", nmea_stats["total_lines"])
    q2.metric("GGA", nmea_stats["gga_lines"])
    q3.metric("Checksum OK", nmea_stats["checksum_ok"])
    q4.metric("Checksum BAD", nmea_stats["checksum_bad"])
    q5.metric("Частота", "—" if nmea_stats["freq_hz"] is None else f"{nmea_stats['freq_hz']:.2f} Гц")

    if nmea_text:
        st.code("\n".join(nmea_text.splitlines()[:16]), language="text")

    estimated_step_m = float(best.speed_mps) * float(dt)
    truth_step_m = float(truth_speed_mps) * float(dt) if truth_speed_mps is not None else estimated_step_m
    estimated_csv = route_to_csv(best.points, estimated_step_m, transform, crs)
    truth_csv = route_to_csv(truth_pts, truth_step_m, transform, crs) if truth_pts is not None else ""
    geojson = route_to_geojson(best.points, transform, crs)

    extra_report = {
        "mode": work_mode,
        "preset": preset,
        "velocity_vector": vel,
        "current_position_px": {"x": float(best.points[-1, 0]), "y": float(best.points[-1, 1])},
        "current_position_map": {"x": float(end_map_x[0]), "y": float(end_map_y[0])},
        "accuracy_estimate": accuracy,
        "nmea_quality": nmea_stats,
        "confidence_gap": float(best.confidence_gap),
        "profile_relief_range_m": float(np.max(observed_profile) - np.min(observed_profile)),
        "profile_variance": float(np.var(observed_profile)),
        "input_start_px": {"x": float(x0), "y": float(y0)},
        "known_azimuth_used": bool(use_known_azimuth),
        "checkpoint_preset": bool(checkpoint_preset),
        "pixel_size_source": "geotiff_transform" if (use_auto_pixel_size and source_mode == "Загрузить GeoTIFF") else "manual",
    }
    if end_lon is not None and end_lat is not None:
        extra_report["current_position_lonlat"] = {"lon": float(end_lon[0]), "lat": float(end_lat[0])}
    if work_mode == "Контрольный профиль":
        extra_report.update(
            {
                "truth_azimuth_deg": float(truth_azimuth),
                "truth_speed_mps": float(truth_speed_mps),
                "truth_shift_px": float(truth_shift),
                "azimuth_error_deg": float(az_err),
                "speed_error_mps": float(sp_err),
                "trajectory_errors": err,
            }
        )
    if dt_est is not None:
        extra_report["dt_estimated_from_nmea_s"] = float(dt_est)

    report_json = make_report_json(best, label, pixel_size_m, dt, len(observed_profile), extra=extra_report)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("Скачать маршрут CSV", estimated_csv, file_name="estimated_route.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("Скачать GeoJSON", geojson, file_name="estimated_route.geojson", mime="application/geo+json", use_container_width=True)
    with c3:
        st.download_button("Скачать отчёт JSON", report_json, file_name="navigation_report.json", mime="application/json", use_container_width=True)
    with c4:
        st.download_button("Скачать radio.nmea", nmea_text, file_name="radio.nmea", mime="text/plain", use_container_width=True)

    if truth_csv:
        st.download_button("Скачать контрольный маршрут CSV", truth_csv, file_name="truth_route.csv", mime="text/csv")

with tab_biz:
    st.subheader("Сценарии применения")
    st.markdown(
        """
        **Назначение:** резервный гражданский навигационный канал для участков, где GNSS деградирует или недоступен.
        Система не требует камеры, связи с землёй и внешней инфраструктуры: нужен радиовысотомер, барометрическая высота и заранее загруженная DEM.

        **Сценарии:** доставка медикаментов и оборудования в таёжные посёлки, маршруты к арктическим метеостанциям, снабжение геологоразведочных лагерей, восстановление фактического маршрута после миссии.

        **Ограничения:** плоский рельеф снижает информативность, качество DEM влияет на точность, а для real-time режима алгоритм запускается на скользящем окне последних измерений.
        """
    )
    st.info("При отказе GNSS система продолжает оценивать местоположение борта, направление движения, скорость и доверие к найденному решению.")

st.success("Готово: координаты, скорость, азимут, current position, 3D DEM, heatmap, NMEA-quality и экспорт построены.")
