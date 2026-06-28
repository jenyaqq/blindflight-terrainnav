import numpy as np

from terrain_nav import (
    clean_dem,
    confidence_label,
    generate_synthetic_dem,
    search_by_correlation,
    simulate_flight,
    trajectory_error_metrics,
)


def main():
    dem, transform, crs = generate_synthetic_dem(500, 500, seed=42, pixel_size_m=30.0)
    dem = clean_dem(dem)

    x0 = dem.shape[1] / 2
    y0 = dem.shape[0] / 2

    sim = simulate_flight(
        dem=dem,
        x0=x0,
        y0=y0,
        azimuth_deg=38,
        speed_mps=40,
        dt=0.5,
        n_points=120,
        shift_px=40,
        pixel_size_m=30,
        h_abs=1500,
        noise_std_m=2,
        seed=1,
    )

    best, _, _, _ = search_by_correlation(
        dem=dem,
        observed_profile=sim["terrain_measured"],
        x0=x0,
        y0=y0,
        dt=0.5,
        pixel_size_m=30,
        azimuth_step_deg=2,
        max_shift_px=220,
        shift_step_px=4,
        speed_candidates_mps=np.arange(20, 81, 5),
    )

    errors = trajectory_error_metrics(sim["truth_points"], best.points, 30)
    print("Best:", best.to_public_dict())
    print("Confidence:", confidence_label(best, sim["terrain_measured"]))
    print("Errors:", errors)

    assert abs(best.azimuth_deg - 38) <= 2
    assert abs(best.speed_mps - 40) <= 5
    assert errors["mean_error_m"] < 100
    print("OK")


if __name__ == "__main__":
    main()
