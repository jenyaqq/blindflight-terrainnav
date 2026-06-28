from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from terrain_nav import (
    build_azimuth_candidates,
    clean_dem,
    estimate_accuracy_radius_m,
    estimate_pixel_size_m,
    heights_to_terrain_profile,
    load_dem_from_bytes,
    map_to_pixel,
    parse_plain_heights_text,
    pixel_to_lonlat,
    pixel_to_map,
    route_to_csv,
    route_to_geojson,
    search_by_correlation,
    velocity_components,
)


def _plot_route(dem: np.ndarray, points: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    img = ax.imshow(dem, cmap="terrain")
    fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04, label="Высота DEM, м")
    ax.plot(points[:, 0], points[:, 1], linewidth=2.5, linestyle="--", label="Найденная траектория")
    ax.scatter(points[0, 0], points[0, 1], s=90, marker="o", label="Старт")
    ax.scatter(points[-1, 0], points[-1, 1], s=100, marker="x", label="Текущая позиция")
    ax.set_xlabel("X, пиксели")
    ax.set_ylabel("Y, пиксели")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_heatmap(heatmap: np.ndarray, azimuths: np.ndarray, shifts: np.ndarray, best, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y0 = float(np.min(azimuths)) if len(azimuths) else 0.0
    y1 = float(np.max(azimuths)) if len(azimuths) else 1.0
    if abs(y1 - y0) < 1e-9:
        y0 -= 0.5
        y1 += 0.5
    x0 = float(shifts[0]) if len(shifts) else 0.0
    x1 = float(shifts[-1]) if len(shifts) else 1.0
    if abs(x1 - x0) < 1e-9:
        x0 -= 0.5
        x1 += 0.5
    im = ax.imshow(heatmap, aspect="auto", origin="lower", extent=[x0, x1, y0, y1], cmap="viridis")
    fig.colorbar(im, ax=ax, label="Коэффициент корреляции")
    ax.scatter([best.shift_px], [best.azimuth_deg], s=90, marker="x", label="Лучший кандидат")
    ax.set_xlabel("Смещение, пиксели")
    ax.set_ylabel("Азимут, °")
    ax.set_title("Heatmap корреляционного поиска")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("значение должно быть > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TerrainNav checkpoint runner")
    parser.add_argument("--dem", required=True, help="Путь к GeoTIFF DEM")
    parser.add_argument("--heights", required=True, help="TXT/CSV с высотами в метрах")
    parser.add_argument("--start-x", required=True, type=float, help="Начальная координата x")
    parser.add_argument("--start-y", required=True, type=float, help="Начальная координата y")
    parser.add_argument("--azimuth", required=True, type=float, help="Направление движения, градусы от севера")
    parser.add_argument("--coord-system", choices=["px", "map"], default="px", help="px: x/y в пикселях DEM; map: x/y в координатах GeoTIFF")
    parser.add_argument("--height-kind", choices=["radio", "terrain"], default="radio", help="radio: радиовысота; terrain: уже высота рельефа")
    parser.add_argument("--delimiter", default="auto", help="Разделитель высот: auto, \\n, ;, ',', пробел или строка")
    parser.add_argument("--h-abs", type=float, default=1500.0, help="Абсолютная барометрическая высота полёта, м")
    parser.add_argument("--dt", type=_positive_float, default=0.5, help="Период измерений, с. Нужен для скорости м/с")
    parser.add_argument("--pixel-size", type=_positive_float, default=None, help="Размер пикселя, м. Если не указан, берётся из GeoTIFF")
    parser.add_argument("--speed-min", type=_positive_float, default=20.0, help="Минимальная скорость для перебора, м/с")
    parser.add_argument("--speed-max", type=_positive_float, default=80.0, help="Максимальная скорость для перебора, м/с")
    parser.add_argument("--speed-step", type=_positive_float, default=1.0, help="Шаг перебора скорости, м/с")
    parser.add_argument("--azimuth-tolerance", type=float, default=0.0, help="Допуск вокруг заданного азимута, ±°")
    parser.add_argument("--azimuth-step", type=_positive_float, default=1.0, help="Шаг азимута внутри допуска, °")
    parser.add_argument("--max-shift", type=int, default=0, help="Максимальное смещение от заданной стартовой точки, px. Для точного старта оставьте 0")
    parser.add_argument("--shift-step", type=int, default=1, help="Шаг смещения, px")
    parser.add_argument("--out", default="checkpoint_output", help="Папка результата")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dem_path = Path(args.dem)
    heights_path = Path(args.heights)
    if not dem_path.exists():
        raise FileNotFoundError(dem_path)
    if not heights_path.exists():
        raise FileNotFoundError(heights_path)

    dem, transform, crs = load_dem_from_bytes(dem_path.read_bytes())
    dem = clean_dem(dem)
    height, width = dem.shape

    if args.coord_system == "map":
        start_x_px, start_y_px = map_to_pixel(args.start_x, args.start_y, transform)
    else:
        start_x_px, start_y_px = float(args.start_x), float(args.start_y)

    if not (0 <= start_x_px < width and 0 <= start_y_px < height):
        raise ValueError(f"Стартовая точка вне DEM: x={start_x_px:.2f}, y={start_y_px:.2f}, размер {width}×{height} px")

    pixel_size_m = float(args.pixel_size) if args.pixel_size is not None else estimate_pixel_size_m(transform, crs)
    heights_text = heights_path.read_text(encoding="utf-8", errors="ignore")
    heights_m = parse_plain_heights_text(heights_text, delimiter=args.delimiter)
    if len(heights_m) < 5:
        raise ValueError("Слишком мало высот: нужно минимум 5 числовых значений")

    observed_profile = heights_to_terrain_profile(heights_m, h_abs=args.h_abs, height_kind=args.height_kind)
    speed_candidates = np.arange(args.speed_min, args.speed_max + args.speed_step * 0.5, args.speed_step, dtype=np.float32)
    azimuth_candidates = build_azimuth_candidates(args.azimuth, args.azimuth_tolerance, args.azimuth_step)

    best, heatmap, azimuths, shifts = search_by_correlation(
        dem=dem,
        observed_profile=observed_profile,
        x0=start_x_px,
        y0=start_y_px,
        dt=args.dt,
        pixel_size_m=pixel_size_m,
        max_shift_px=max(0, int(args.max_shift)),
        shift_step_px=max(1, int(args.shift_step)),
        speed_candidates_mps=speed_candidates,
        azimuth_candidates_deg=azimuth_candidates,
    )
    if not best.ok:
        raise RuntimeError("Маршрут не найден. Проверьте старт, направление, dt, диапазон скоростей и размер пикселя")

    velocity = velocity_components(best.speed_mps, best.azimuth_deg)
    accuracy = estimate_accuracy_radius_m(
        best,
        observed_profile,
        pixel_size_m=pixel_size_m,
        azimuth_step_deg=args.azimuth_step if args.azimuth_tolerance > 0 else 0.0,
        shift_step_px=max(1, int(args.shift_step)),
        speed_step_mps=args.speed_step,
        dt=args.dt,
    )

    map_x, map_y = pixel_to_map(best.points, transform)
    lon, lat = pixel_to_lonlat(best.points, transform, crs)
    current = {
        "x_px": float(best.points[-1, 0]),
        "y_px": float(best.points[-1, 1]),
        "map_x": float(map_x[-1]),
        "map_y": float(map_y[-1]),
    }
    if lon is not None and lat is not None:
        current.update({"lon": float(lon[-1]), "lat": float(lat[-1])})

    report = {
        "input": {
            "dem": str(dem_path),
            "heights": str(heights_path),
            "coord_system": args.coord_system,
            "start_x_px": float(start_x_px),
            "start_y_px": float(start_y_px),
            "given_azimuth_deg": float(args.azimuth),
            "height_kind": args.height_kind,
            "h_abs_m": float(args.h_abs),
            "dt_s": float(args.dt),
            "pixel_size_m": float(pixel_size_m),
        },
        "result": best.to_public_dict(),
        "velocity_vector": velocity,
        "current_position": current,
        "accuracy_estimate": accuracy,
        "profile": {
            "n_points": int(len(observed_profile)),
            "relief_range_m": float(np.max(observed_profile) - np.min(observed_profile)),
            "variance": float(np.var(observed_profile)),
        },
    }

    (out_dir / "estimated_route.csv").write_text(route_to_csv(best.points, float(best.speed_mps) * float(args.dt), transform, crs), encoding="utf-8")
    (out_dir / "estimated_route.geojson").write_text(route_to_geojson(best.points, transform, crs), encoding="utf-8")
    (out_dir / "navigation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_route(dem, best.points, out_dir / "trajectory_on_dem.png", "Найденная траектория на DEM")
    _plot_heatmap(heatmap, azimuths, shifts, best, out_dir / "correlation_heatmap.png")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nФайлы сохранены в: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
