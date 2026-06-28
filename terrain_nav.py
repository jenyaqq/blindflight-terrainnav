from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform


@dataclass
class SearchResult:

    corr: float
    azimuth_deg: Optional[float]
    shift_px: Optional[float]
    ref_profile: Optional[np.ndarray]
    points: Optional[np.ndarray]
    second_corr: float = -1.0
    speed_mps: Optional[float] = None
    ds_px: Optional[float] = None
    start_x_px: Optional[float] = None
    start_y_px: Optional[float] = None
    start_dx_px: float = 0.0
    start_dy_px: float = 0.0

    @property
    def confidence_gap(self) -> float:
        return float(self.corr - self.second_corr)

    @property
    def ok(self) -> bool:
        return self.points is not None and self.ref_profile is not None

    def to_public_dict(self) -> dict:
        return {
            "corr": float(self.corr),
            "second_corr": float(self.second_corr),
            "confidence_gap": float(self.confidence_gap),
            "azimuth_deg": none_or_float(self.azimuth_deg),
            "shift_px": none_or_float(self.shift_px),
            "speed_mps": none_or_float(self.speed_mps),
            "ds_px": none_or_float(self.ds_px),
            "start_x_px": none_or_float(self.start_x_px),
            "start_y_px": none_or_float(self.start_y_px),
            "start_dx_px": float(self.start_dx_px),
            "start_dy_px": float(self.start_dy_px),
        }


def none_or_float(value):
    return None if value is None else float(value)


def load_dem_from_bytes(file_bytes: bytes):
    with MemoryFile(file_bytes) as memfile:
        with memfile.open() as src:
            dem = src.read(1).astype(np.float32)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)

    return dem, transform, crs


def generate_synthetic_dem(height: int = 500, width: int = 500, seed: int = 42, pixel_size_m: float = 30.0):
    rng = np.random.default_rng(seed)

    y = np.linspace(-1, 1, height)
    x = np.linspace(-1, 1, width)
    xx, yy = np.meshgrid(x, y)

    dem = (
        300
        + 120 * np.exp(-((xx + 0.45) ** 2 + (yy + 0.25) ** 2) / 0.05)
        + 180 * np.exp(-((xx - 0.25) ** 2 + (yy - 0.15) ** 2) / 0.08)
        + 90 * np.exp(-((xx + 0.10) ** 2 + (yy - 0.55) ** 2) / 0.03)
        - 100 * np.exp(-((xx - 0.55) ** 2 + (yy + 0.45) ** 2) / 0.06)
        + 35 * np.sin(5 * np.pi * xx) * np.cos(4 * np.pi * yy)
        + 18 * np.sin(11 * xx + 4 * yy)
        + 6 * rng.normal(size=(height, width))
    ).astype(np.float32)

    transform = from_origin(0, height * pixel_size_m, pixel_size_m, pixel_size_m)
    crs = None
    return dem, transform, crs


def clean_dem(dem: np.ndarray) -> np.ndarray:
    dem = dem.astype(np.float32)
    if np.all(np.isnan(dem)):
        raise ValueError("DEM состоит только из NaN.")
    return np.nan_to_num(dem, nan=float(np.nanmean(dem)))


def estimate_pixel_size_m(transform, crs=None) -> float:
    if transform is None:
        return 30.0

    a = abs(float(transform.a))
    e = abs(float(transform.e))
    px = (a + e) / 2.0 if (a > 0 and e > 0) else 30.0

    try:
        if crs is not None and getattr(crs, "is_geographic", False):

            return float(px * 111_320.0)
    except Exception:
        pass

    if px <= 0:
        px = 30.0
    return float(px)


def bilinear_sample(dem: np.ndarray, x: float, y: float) -> float:
    h, w = dem.shape

    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return np.nan

    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1

    dx = x - x0
    dy = y - y0

    q11 = float(dem[y0, x0])
    q21 = float(dem[y0, x1])
    q12 = float(dem[y1, x0])
    q22 = float(dem[y1, x1])

    if np.any(np.isnan([q11, q21, q12, q22])):
        return np.nan

    return (
        q11 * (1 - dx) * (1 - dy)
        + q21 * dx * (1 - dy)
        + q12 * (1 - dx) * dy
        + q22 * dx * dy
    )


def sample_profile(
    dem: np.ndarray,
    x0: float,
    y0: float,
    azimuth_deg: float,
    ds_px: float,
    n_points: int,
    shift_px: float = 0.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    h, w = dem.shape
    az = np.deg2rad(azimuth_deg)

    s = shift_px + np.arange(n_points, dtype=np.float32) * float(ds_px)
    xs = x0 + s * np.sin(az)
    ys = y0 - s * np.cos(az)


    if np.any(xs < 0) or np.any(xs >= w - 1) or np.any(ys < 0) or np.any(ys >= h - 1):
        return None, None

    x0i = np.floor(xs).astype(np.int32)
    y0i = np.floor(ys).astype(np.int32)
    x1i = x0i + 1
    y1i = y0i + 1

    dx = xs - x0i
    dy = ys - y0i

    q11 = dem[y0i, x0i].astype(np.float32)
    q21 = dem[y0i, x1i].astype(np.float32)
    q12 = dem[y1i, x0i].astype(np.float32)
    q22 = dem[y1i, x1i].astype(np.float32)

    if np.any(np.isnan(q11)) or np.any(np.isnan(q21)) or np.any(np.isnan(q12)) or np.any(np.isnan(q22)):
        return None, None

    profile = (
        q11 * (1 - dx) * (1 - dy)
        + q21 * dx * (1 - dy)
        + q12 * (1 - dx) * dy
        + q22 * dx * dy
    ).astype(np.float32)

    points = np.column_stack([xs, ys]).astype(np.float32)
    return profile, points


def terrain_from_radio(h_abs: float, radio_profile: np.ndarray) -> np.ndarray:
    return h_abs - np.asarray(radio_profile, dtype=np.float32)


def normalized_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if len(a) != len(b) or len(a) == 0:
        return -1.0

    a = a - np.mean(a)
    b = b - np.mean(b)

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na < 1e-9 or nb < 1e-9:
        return -1.0

    return float(np.dot(a, b) / (na * nb))


def kalman_1d(measurements: np.ndarray, process_var: float = 1.0, measurement_var: float = 4.0) -> np.ndarray:
    z = np.asarray(measurements, dtype=np.float32)
    if len(z) == 0:
        return z

    x = float(z[0])
    p = 1.0
    out = np.zeros_like(z, dtype=np.float32)

    for i, value in enumerate(z):

        p = p + process_var


        k = p / (p + measurement_var)
        x = x + k * (float(value) - x)
        p = (1 - k) * p
        out[i] = x

    return out


def nmea_checksum(payload: str) -> str:
    cs = 0
    for ch in payload:
        cs ^= ord(ch)
    return f"{cs:02X}"


def make_nmea_sentence(payload: str) -> str:
    return f"${payload}*{nmea_checksum(payload)}"


def validate_nmea_checksum(sentence: str) -> bool:
    line = sentence.strip()
    if not line.startswith("$") or "*" not in line:
        return False
    payload, expected = line[1:].split("*", 1)
    expected = expected[:2].upper()
    return nmea_checksum(payload) == expected


def radio_to_nmea(radio_profile: np.ndarray, dt: float) -> List[str]:
    lines = []
    for i, h_radio in enumerate(radio_profile):
        t = i * dt
        hh = int(t // 3600) % 24
        mm = int((t % 3600) // 60)
        ss = t % 60
        stamp = f"{hh:02d}{mm:02d}{ss:06.3f}"
        fields = [
            "GPGGA",
            stamp,
            "", "", "", "", "", "", "",
            f"{float(h_radio):.2f}",
            "M",
            "46.9",
            "M",
            "",
            "",
        ]
        lines.append(make_nmea_sentence(",".join(fields)))
    return lines


def _strip_checksum(sentence: str) -> str:
    line = sentence.strip()
    if line.startswith("$"):
        line = line[1:]
    if "*" in line:
        line = line.split("*", 1)[0]
    return line


def parse_gga_time_to_seconds(value: str) -> Optional[float]:
    if not value or len(value) < 6:
        return None
    try:
        hh = int(value[0:2])
        mm = int(value[2:4])
        ss = float(value[4:])
        return hh * 3600 + mm * 60 + ss
    except ValueError:
        return None


def parse_nmea_altitude(sentence: str) -> Optional[float]:
    payload = _strip_checksum(sentence)
    parts = payload.split(",")
    if len(parts) < 10:
        return None
    if not parts[0].endswith("GGA"):
        return None
    try:
        return float(parts[9])
    except ValueError:
        return None


def parse_nmea_profile(nmea_text: str) -> np.ndarray:
    profile, _ = parse_nmea_profile_with_dt(nmea_text)
    return profile


def parse_nmea_profile_with_dt(nmea_text: str) -> Tuple[np.ndarray, Optional[float]]:
    values = []
    times = []

    for line in nmea_text.splitlines():
        if not line.strip():
            continue

        payload = _strip_checksum(line)
        parts = payload.split(",")
        if len(parts) < 10 or not parts[0].endswith("GGA"):
            continue

        try:
            values.append(float(parts[9]))
            t = parse_gga_time_to_seconds(parts[1])
            if t is not None:
                times.append(t)
        except ValueError:
            continue

    dt_est = None
    if len(times) >= 2:
        diffs = np.diff(np.asarray(times, dtype=np.float64))
        diffs = diffs[diffs > 0]
        if len(diffs) > 0:
            dt_est = float(np.median(diffs))

    return np.asarray(values, dtype=np.float32), dt_est


def parse_plain_heights_text(text: str, delimiter: str = "auto") -> np.ndarray:
    if text is None:
        return np.asarray([], dtype=np.float32)

    payload = str(text).replace("\ufeff", "").strip()
    if not payload:
        return np.asarray([], dtype=np.float32)

    delimiter_normalized = (delimiter or "auto").strip()
    if delimiter_normalized in {"\\n", "newline", "line", "строка", "строки"}:
        tokens = payload.splitlines()
    elif delimiter_normalized and delimiter_normalized.lower() != "auto":
        tokens = payload.split(delimiter_normalized)
    else:


        tokens = re.findall(r"[-+]?\d+(?:[\.,]\d+)?(?:[eE][-+]?\d+)?", payload)

    values = []
    for token in tokens:
        t = str(token).strip()
        if not t:
            continue

        match = re.search(r"[-+]?\d+(?:[\.,]\d+)?(?:[eE][-+]?\d+)?", t)
        if not match:
            continue
        try:
            values.append(float(match.group(0).replace(",", ".")))
        except ValueError:
            continue

    return np.asarray(values, dtype=np.float32)


def heights_to_terrain_profile(heights_m: np.ndarray, h_abs: float = 1500.0, height_kind: str = "radio") -> np.ndarray:
    heights = np.asarray(heights_m, dtype=np.float32)
    mode = (height_kind or "radio").strip().lower()
    if mode in {"terrain", "relief", "dem", "ground", "h_terrain", "рельеф"}:
        return heights
    return terrain_from_radio(float(h_abs), heights)


def build_azimuth_candidates(center_deg: float, tolerance_deg: float = 0.0, step_deg: float = 1.0) -> np.ndarray:
    center = float(center_deg) % 360.0
    tolerance = max(0.0, float(tolerance_deg))
    step = max(1e-6, float(step_deg))
    if tolerance <= 0:
        return np.asarray([center], dtype=np.float32)

    offsets = np.arange(-tolerance, tolerance + step * 0.5, step, dtype=np.float32)
    values = (center + offsets) % 360.0

    out = []
    seen = set()
    for value in values:
        key = round(float(value), 6)
        if key not in seen:
            out.append(float(value))
            seen.add(key)
    return np.asarray(out, dtype=np.float32)


def simulate_flight(
    dem: np.ndarray,
    x0: float,
    y0: float,
    azimuth_deg: float,
    speed_mps: float,
    dt: float,
    n_points: int,
    shift_px: float,
    pixel_size_m: float = 30.0,
    h_abs: float = 1500.0,
    noise_std_m: float = 2.0,
    seed: int = 1,
):
    rng = np.random.default_rng(seed)

    ds_m = speed_mps * dt
    ds_px = ds_m / pixel_size_m

    terrain_true, truth_points = sample_profile(
        dem=dem,
        x0=x0,
        y0=y0,
        azimuth_deg=azimuth_deg,
        ds_px=ds_px,
        n_points=n_points,
        shift_px=shift_px,
    )

    if terrain_true is None or truth_points is None:
        raise ValueError(
            "Истинная траектория вышла за границы DEM. "
            "Измени азимут, shift, скорость, число точек или размер пикселя."
        )

    noise = rng.normal(0, noise_std_m, size=n_points).astype(np.float32)
    radio_profile = h_abs - terrain_true + noise
    terrain_measured = terrain_from_radio(h_abs, radio_profile)

    nmea_lines = radio_to_nmea(radio_profile, dt)

    return {
        "terrain_true": terrain_true,
        "terrain_measured": terrain_measured,
        "radio_profile": radio_profile,
        "truth_points": truth_points,
        "nmea_lines": nmea_lines,
        "ds_px": ds_px,
    }


def _build_start_offsets(radius_px: int, step_px: int) -> List[Tuple[float, float]]:
    if radius_px <= 0:
        return [(0.0, 0.0)]
    step_px = max(1, int(step_px))
    values = np.arange(-radius_px, radius_px + 1, step_px, dtype=np.float32)
    offsets = []
    for dx in values:
        for dy in values:
            if dx * dx + dy * dy <= radius_px * radius_px:
                offsets.append((float(dx), float(dy)))
    if (0.0, 0.0) not in offsets:
        offsets.append((0.0, 0.0))
    return offsets


def search_by_correlation(
    dem: np.ndarray,
    observed_profile: np.ndarray,
    x0: float,
    y0: float,
    speed_mps: Optional[float] = None,
    dt: float = 0.5,
    pixel_size_m: float = 30.0,
    azimuth_step_deg: int = 2,
    max_shift_px: int = 250,
    shift_step_px: int = 2,
    speed_candidates_mps: Optional[Iterable[float]] = None,
    start_search_radius_px: int = 0,
    start_search_step_px: int = 30,
    azimuth_candidates_deg: Optional[Iterable[float]] = None,
    progress_callback: Optional[Callable[[float], None]] = None,
) -> Tuple[SearchResult, np.ndarray, np.ndarray, np.ndarray]:
    observed_profile = np.asarray(observed_profile, dtype=np.float32)
    n_points = len(observed_profile)
    if n_points < 5:
        raise ValueError("Слишком короткий профиль. Нужно хотя бы 5 измерений.")

    if speed_candidates_mps is None:
        if speed_mps is None:
            raise ValueError("Нужно указать speed_mps или speed_candidates_mps.")
        speeds = np.asarray([speed_mps], dtype=np.float32)
    else:
        speeds = np.asarray(list(speed_candidates_mps), dtype=np.float32)
        speeds = speeds[np.isfinite(speeds) & (speeds > 0)]
        if len(speeds) == 0:
            raise ValueError("Список скоростей пустой.")

    if azimuth_candidates_deg is None:
        azimuths = np.arange(0, 360, max(1, int(azimuth_step_deg)), dtype=np.float32)
    else:
        azimuth_values = []
        seen_azimuths = set()
        for value in azimuth_candidates_deg:
            if value is None or not np.isfinite(value):
                continue
            normalized = float(value) % 360.0
            key = round(normalized, 6)
            if key not in seen_azimuths:
                azimuth_values.append(normalized)
                seen_azimuths.add(key)
        if len(azimuth_values) == 0:
            raise ValueError("Список азимутов пустой.")
        azimuths = np.asarray(azimuth_values, dtype=np.float32)

    shifts = np.arange(0, max_shift_px + 1, max(1, int(shift_step_px)), dtype=np.float32)
    start_offsets = _build_start_offsets(int(start_search_radius_px), int(start_search_step_px))

    heatmap = np.full((len(azimuths), len(shifts)), -1.0, dtype=np.float32)

    best = SearchResult(
        corr=-1.0,
        azimuth_deg=None,
        shift_px=None,
        ref_profile=None,
        points=None,
        second_corr=-1.0,
    )
    second_corr = -1.0

    total = len(speeds) * len(start_offsets) * len(azimuths) * len(shifts)
    done = 0

    for speed in speeds:
        ds_m = float(speed) * dt
        ds_px = ds_m / pixel_size_m

        if ds_px <= 0:
            continue

        for dx0, dy0 in start_offsets:
            sx = x0 + dx0
            sy = y0 + dy0

            for ai, az in enumerate(azimuths):
                for si, shift in enumerate(shifts):
                    ref_profile, ref_points = sample_profile(
                        dem=dem,
                        x0=sx,
                        y0=sy,
                        azimuth_deg=float(az),
                        ds_px=ds_px,
                        n_points=n_points,
                        shift_px=float(shift),
                    )

                    done += 1
                    if progress_callback and total > 0 and done % 1000 == 0:
                        progress_callback(min(done / total, 1.0))

                    if ref_profile is None:
                        continue

                    corr = normalized_corr(observed_profile, ref_profile)

                    if corr > heatmap[ai, si]:
                        heatmap[ai, si] = corr

                    if corr > best.corr:
                        second_corr = best.corr
                        best = SearchResult(
                            corr=corr,
                            azimuth_deg=float(az),
                            shift_px=float(shift),
                            ref_profile=ref_profile,
                            points=ref_points,
                            second_corr=second_corr,
                            speed_mps=float(speed),
                            ds_px=float(ds_px),
                            start_x_px=float(sx),
                            start_y_px=float(sy),
                            start_dx_px=float(dx0),
                            start_dy_px=float(dy0),
                        )
                    elif corr > second_corr and corr < best.corr:
                        second_corr = corr

    best.second_corr = float(second_corr)
    if progress_callback:
        progress_callback(1.0)

    return best, heatmap, azimuths, shifts


def confidence_label(best: SearchResult, observed_profile: np.ndarray) -> str:
    variance = float(np.var(observed_profile))
    relief_range = float(np.max(observed_profile) - np.min(observed_profile)) if len(observed_profile) else 0.0
    gap = best.confidence_gap

    if relief_range < 15 or variance < 5:
        return "LOW: плоский/слабоинформативный рельеф"
    if best.corr > 0.92 and gap > 0.015:
        return "HIGH"
    if best.corr > 0.80:
        return "MEDIUM"
    return "LOW"


def circular_angle_error_deg(a: float, b: float) -> float:
    return float(abs((a - b + 180) % 360 - 180))


def trajectory_error_metrics(truth_points: np.ndarray, estimated_points: np.ndarray, pixel_size_m: float) -> dict:
    if truth_points is None or estimated_points is None:
        return {}

    n = min(len(truth_points), len(estimated_points))
    if n == 0:
        return {}

    diff_px = np.asarray(truth_points[:n], dtype=np.float32) - np.asarray(estimated_points[:n], dtype=np.float32)
    dist_m = np.linalg.norm(diff_px, axis=1) * pixel_size_m

    return {
        "start_error_m": float(dist_m[0]),
        "end_error_m": float(dist_m[-1]),
        "mean_error_m": float(np.mean(dist_m)),
        "max_error_m": float(np.max(dist_m)),
        "rmse_error_m": float(np.sqrt(np.mean(dist_m ** 2))),
    }


def pixel_to_map(points: np.ndarray, transform) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if transform is None:
        return pts[:, 0], pts[:, 1]

    xs = transform.c + pts[:, 0] * transform.a + pts[:, 1] * transform.b
    ys = transform.f + pts[:, 0] * transform.d + pts[:, 1] * transform.e
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def map_to_pixel(x_map: float, y_map: float, transform) -> Tuple[float, float]:
    if transform is None:
        return float(x_map), float(y_map)
    inv = ~transform
    x_px, y_px = inv * (float(x_map), float(y_map))
    return float(x_px), float(y_px)


def pixel_to_lonlat(points: np.ndarray, transform, crs) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if transform is None or crs is None:
        return None, None

    xs, ys = pixel_to_map(points, transform)

    try:
        if getattr(crs, "is_geographic", False):
            return xs, ys
        lon, lat = warp_transform(crs, "EPSG:4326", xs.tolist(), ys.tolist())
        return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
    except Exception:
        return None, None


def route_to_csv(points: np.ndarray, step_m: float, transform=None, crs=None) -> str:
    pts = np.asarray(points, dtype=np.float64)
    map_x, map_y = pixel_to_map(pts, transform)
    lon, lat = pixel_to_lonlat(pts, transform, crs)

    lines = ["idx,x_px,y_px,distance_m,map_x,map_y,lon,lat"]
    for i, (x, y) in enumerate(pts):
        distance_m = i * float(step_m)
        lon_s = "" if lon is None else f"{lon[i]:.8f}"
        lat_s = "" if lat is None else f"{lat[i]:.8f}"
        lines.append(
            f"{i},{x:.3f},{y:.3f},{distance_m:.3f},{map_x[i]:.3f},{map_y[i]:.3f},{lon_s},{lat_s}"
        )
    return "\n".join(lines)


def velocity_components(speed_mps: float, azimuth_deg: float) -> dict:
    az = math.radians(float(azimuth_deg))
    speed = float(speed_mps)
    return {
        "v_east_mps": float(speed * math.sin(az)),
        "v_north_mps": float(speed * math.cos(az)),
        "ground_speed_mps": speed,
        "azimuth_deg": float(azimuth_deg),
    }


def estimate_accuracy_radius_m(
    best: SearchResult,
    observed_profile: np.ndarray,
    pixel_size_m: float,
    azimuth_step_deg: float,
    shift_step_px: float,
    speed_step_mps: float,
    dt: float,
) -> dict:
    n = int(len(observed_profile))
    route_len_m = max(1.0, float(best.speed_mps or 0.0) * float(dt) * max(n - 1, 1))
    grid_error_m = max(0.0, float(pixel_size_m) * float(shift_step_px) / 2.0)
    angle_error_m = route_len_m * math.sin(math.radians(max(0.0, float(azimuth_step_deg)) / 2.0))
    speed_error_m = max(0.0, max(n - 1, 1) * float(dt) * float(speed_step_mps) / 2.0)

    corr_penalty_m = max(0.0, (1.0 - max(-1.0, min(1.0, float(best.corr)))) * route_len_m * 0.25)
    gap = max(0.0, float(best.confidence_gap))
    ambiguity_penalty_m = 0.0 if gap >= 0.02 else (0.02 - gap) / 0.02 * max(30.0, 2.0 * pixel_size_m)

    relief_range = float(np.max(observed_profile) - np.min(observed_profile)) if n else 0.0
    relief_penalty_m = 0.0
    if relief_range < 15.0:
        relief_penalty_m = 5.0 * pixel_size_m
    elif relief_range < 35.0:
        relief_penalty_m = 2.0 * pixel_size_m

    radius_m = math.sqrt(
        grid_error_m ** 2
        + angle_error_m ** 2
        + speed_error_m ** 2
        + corr_penalty_m ** 2
        + ambiguity_penalty_m ** 2
        + relief_penalty_m ** 2
    )

    return {
        "estimated_radius_m": float(radius_m),
        "grid_error_m": float(grid_error_m),
        "angle_error_m": float(angle_error_m),
        "speed_error_m": float(speed_error_m),
        "corr_penalty_m": float(corr_penalty_m),
        "ambiguity_penalty_m": float(ambiguity_penalty_m),
        "relief_penalty_m": float(relief_penalty_m),
        "route_length_m": float(route_len_m),
        "relief_range_m": float(relief_range),
    }


def nmea_quality_report(nmea_text: str) -> dict:
    total = 0
    gga = 0
    checksum_ok = 0
    checksum_bad = 0
    altitude_count = 0
    malformed = 0
    times = []

    for raw in nmea_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        total += 1
        has_checksum = line.startswith("$") and "*" in line
        if has_checksum:
            if validate_nmea_checksum(line):
                checksum_ok += 1
            else:
                checksum_bad += 1

        payload = _strip_checksum(line)
        parts = payload.split(",")
        if len(parts) < 10 or not parts[0].endswith("GGA"):
            malformed += 1
            continue
        gga += 1
        try:
            float(parts[9])
            altitude_count += 1
        except ValueError:
            malformed += 1
        t = parse_gga_time_to_seconds(parts[1]) if len(parts) > 1 else None
        if t is not None:
            times.append(t)

    dt_est = None
    freq_hz = None
    if len(times) >= 2:
        diffs = np.diff(np.asarray(times, dtype=np.float64))
        diffs = diffs[diffs > 0]
        if len(diffs) > 0:
            dt_est = float(np.median(diffs))
            if dt_est > 0:
                freq_hz = float(1.0 / dt_est)

    return {
        "total_lines": int(total),
        "gga_lines": int(gga),
        "checksum_ok": int(checksum_ok),
        "checksum_bad": int(checksum_bad),
        "altitude_values": int(altitude_count),
        "malformed_or_ignored": int(malformed),
        "dt_est_s": dt_est,
        "freq_hz": freq_hz,
    }


def route_to_geojson(points: np.ndarray, transform=None, crs=None) -> str:
    pts = np.asarray(points, dtype=np.float64)
    lon, lat = pixel_to_lonlat(pts, transform, crs)
    if lon is not None and lat is not None:
        coords = [[float(lon[i]), float(lat[i])] for i in range(len(pts))]
        coord_type = "lonlat"
    else:
        map_x, map_y = pixel_to_map(pts, transform)
        coords = [[float(map_x[i]), float(map_y[i])] for i in range(len(pts))]
        coord_type = "map_xy_or_pixels"

    feature = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "estimated_terrain_nav_route", "coordinate_type": coord_type},
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        ],
    }
    return json.dumps(feature, ensure_ascii=False, indent=2)

def make_report_json(
    best: SearchResult,
    confidence: str,
    pixel_size_m: float,
    dt: float,
    n_points: int,
    extra: Optional[dict] = None,
) -> str:
    report = {
        "best_result": best.to_public_dict(),
        "confidence": confidence,
        "pixel_size_m": float(pixel_size_m),
        "dt_s": float(dt),
        "n_points": int(n_points),
    }
    if extra:
        report.update(extra)
    return json.dumps(report, ensure_ascii=False, indent=2)
