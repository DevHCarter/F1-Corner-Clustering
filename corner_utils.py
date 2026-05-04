import warnings
import numpy as np
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d


# -- Reference corner counts (official / commonly cited) ----------------------
# Keys must match the GRAND_PRIX string used in FastF1 (e.g. 'Monza', 'Monaco').
# FastF1 accepts both GP name ('Italian') and circuit/city name ('Monza').
# Use whichever form you pass to fastf1.get_session().
KNOWN_CORNER_COUNTS = {
    # Keys must match FastF1 EventName exactly (schedule['EventName'])
    'Bahrain Grand Prix':        15,
    'Saudi Arabian Grand Prix':  27,
    'Australian Grand Prix':     16,
    'Japanese Grand Prix':       18,
    'Chinese Grand Prix':        16,
    'Miami Grand Prix':          19,
    'Emilia Romagna Grand Prix': 19,
    'Monaco Grand Prix':         19,
    'Canadian Grand Prix':       13,
    'Spanish Grand Prix':        14,
    'Austrian Grand Prix':       10,
    'British Grand Prix':        18,
    'Hungarian Grand Prix':      14,
    'Belgian Grand Prix':        19,
    'Dutch Grand Prix':          14,
    'Italian Grand Prix':        11,
    'Azerbaijan Grand Prix':     20,
    'Singapore Grand Prix':      19,
    'United States Grand Prix':  20,
    'Mexico City Grand Prix':    17,
    'S\u00e3o Paulo Grand Prix': 15,  # \u00e3 is the a-tilde; unicode escape keeps this file ASCII
    'Las Vegas Grand Prix':      17,
    'Qatar Grand Prix':          16,
    'Abu Dhabi Grand Prix':      16,
}


# Normalize raw steering to [0, 1] so it can be compared across circuits and laps.
def _normalize_steering(steering_raw, smooth_sigma=3):
    abs_steer = np.abs(steering_raw.astype(float))
    smoothed  = gaussian_filter1d(abs_steer, sigma=smooth_sigma)
    peak      = smoothed.max()
    if peak < 1e-6:
        return np.zeros_like(smoothed)
    return smoothed / peak


# Signed curvature formula: kappa = (x' * y'' - y' * x'') / (x'^2 + y'^2)^1.5
# Positive kappa = left-hander, negative = right-hander.
# Savitzky-Golay smoothing is applied first to reduce GPS noise before differentiating.
def compute_curvature(x, y, smooth_window=11, smooth_poly=3):

    x_s = savgol_filter(x, smooth_window, smooth_poly)
    y_s = savgol_filter(y, smooth_window, smooth_poly)

    dx  = np.gradient(x_s)
    dy  = np.gradient(y_s)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    denom = (dx**2 + dy**2) ** 1.5
    denom = np.where(denom < 1e-8, 1e-8, denom)
    kappa = (dx * ddy - dy * ddx) / denom
    return kappa, x_s, y_s


def detect_corners(
    abs_kappa,
    dist,
    speed,
    kappa_signed,
    target_count=None,
    min_dist_m=60,
    window_m=100,
    steering_arr=None,
    steer_dist=None,
    steer_prominence=0.35,
):
    
    # Convert meter thresholds to sample-index gaps based on median sample spacing.
    avg_spacing = np.median(np.diff(dist))
    min_idx_gap = max(1, int(min_dist_m / avg_spacing))
    half_win    = max(1, int(window_m  / avg_spacing))

    nonzero = abs_kappa[abs_kappa > 0]

    # Binary search: find the prominence threshold that yields exactly target_count peaks.
    if target_count is not None and target_count > 0:
        lo = np.percentile(nonzero, 40)
        hi = np.percentile(abs_kappa, 100)
        best_peaks = np.array([], dtype=int)
        best_diff  = float('inf')

        for _ in range(60):
            mid   = (lo + hi) / 2.0
            peaks, _ = find_peaks(abs_kappa, prominence=mid,
                                  distance=min_idx_gap)
            diff  = len(peaks) - target_count

            if abs(diff) < best_diff:
                best_diff  = abs(diff)
                best_peaks = peaks.copy()

            if diff == 0:
                break
            elif diff > 0:   # too many -> raise threshold
                lo = mid
            else:            # too few  -> lower threshold
                hi = mid

        if best_diff > 0:
            warnings.warn(
                f"Corner calibration: target={target_count}, "
                f"achieved={len(best_peaks)} (closest possible)"
            )
        peaks = best_peaks

    else:
        prominence = np.percentile(nonzero, 60)
        peaks, _   = find_peaks(abs_kappa, prominence=prominence,
                                distance=min_idx_gap)

    corners = []
    for rank, idx in enumerate(peaks):
        idx   = int(idx)
        start = max(0, idx - half_win)
        end   = min(len(dist) - 1, idx + half_win)
        direction = 'L' if kappa_signed[idx] > 0 else 'R'
        corners.append({
            'corner_num':  rank + 1,
            'apex_idx':    idx,
            'apex_dist':   dist[idx],
            'start_idx':   start,
            'end_idx':     end,
            'direction':   direction,
            'peak_kappa':  abs_kappa[idx],
        })
    if steering_arr is not None and steer_dist is not None:
        steer_norm     = _normalize_steering(steering_arr)
        steer_on_grid  = np.interp(dist, steer_dist, steer_norm)
        steer_peaks, _ = find_peaks(steer_on_grid, prominence=steer_prominence,
                                    distance=min_idx_gap)
        claimed_dists  = {c['apex_dist'] for c in corners}
        for idx in steer_peaks:
            idx = int(idx)
            if all(abs(dist[idx] - cd) > min_dist_m for cd in claimed_dists):
                start = max(0, idx - half_win)
                end   = min(len(dist) - 1, idx + half_win)
                corners.append({
                    'corner_num':  None,
                    'apex_idx':    idx,
                    'apex_dist':   dist[idx],
                    'start_idx':   start,
                    'end_idx':     end,
                    'direction':   'L' if kappa_signed[idx] > 0 else 'R',
                    'peak_kappa':  abs_kappa[idx],
                    'detected_by': 'steering',
                })
                claimed_dists.add(dist[idx])
        corners.sort(key=lambda c: c['apex_dist'])
        for rank, c in enumerate(corners):
            c['corner_num'] = rank + 1
            c.setdefault('detected_by', 'curvature')

    return corners


# Classify a corner by how much braking occurs in its window.
# brake_frac is the fraction of samples where the brake channel is non-zero.
def _classify_corner_type(seg_brake, seg_throttle):
    brake_frac = float(np.mean(seg_brake > 0))
    if brake_frac < 0.05:
        return 'flat_out'
    elif brake_frac < 0.25:
        return 'light_braking'
    else:
        return 'heavy_braking'


# Extract a flat feature dict for one corner using a +/- window_m meter window around the apex.
# Returns None if the required telemetry columns are missing or the window is too short.
def extract_features(corner, telemetry_df, abs_kappa, window_m=100):

    required_cols = {'Speed', 'Brake', 'Throttle', 'nGear', 'Distance'}
    if not required_cols.issubset(telemetry_df.columns):
        return None

    apex_dist = float(corner['apex_dist'])
    dist_arr  = telemetry_df['Distance'].values.astype(float)

    # Distance-based window -- works for any telemetry length
    mask = (dist_arr >= apex_dist - window_m) & (dist_arr <= apex_dist + window_m)
    seg  = telemetry_df[mask]

    if len(seg) < 5:
        return None

    seg_speed    = seg['Speed'].values.astype(float)
    seg_brake    = seg['Brake'].values.astype(float)
    seg_throttle = np.clip(seg['Throttle'].values.astype(float), 0.0, 100.0)
    seg_gear     = seg['nGear'].values.astype(float)
    seg_dist     = seg['Distance'].values.astype(float)

    has_steering = 'Steering' in telemetry_df.columns
    seg_steer    = seg['Steering'].values.astype(float) if has_steering else None
    corner_type  = _classify_corner_type(seg_brake, seg_throttle)

    # Apex index within this driver's window (closest sample to apex_dist)
    apex_local = int(np.argmin(np.abs(seg_dist - apex_dist)))

    # Curvature features from fastest-lap geometry slice
    k_slice = abs_kappa[corner['start_idx']:corner['end_idx']]
    if len(k_slice) == 0:
        kappa_max  = float(corner['peak_kappa'])
        kappa_mean = float(corner['peak_kappa'])
    else:
        kappa_max  = float(k_slice.max())
        kappa_mean = float(k_slice.mean())

    # Speed features
    speed_min   = float(seg_speed.min())
    speed_mean  = float(seg_speed.mean())
    speed_entry = float(seg_speed[0])
    speed_exit  = float(seg_speed[-1])
    speed_drop  = speed_entry - speed_min

    # Lateral G proxy: (v_ms^2 * kappa_max) / 9.81
    # kappa_max is a scalar from fastest-lap geometry -- consistent across all drivers
    v_ms      = seg_speed / 3.6
    lat_g_max = float(((v_ms ** 2) * kappa_max / 9.81).max())

    if has_steering:
        steer_abs  = np.abs(seg_steer)
        steer_max  = float(steer_abs.max())
        steer_mean = float(steer_abs.mean())
        steer_apex = float(abs(seg_steer[apex_local]))
    else:
        steer_max = steer_mean = steer_apex = float('nan')

    return {
        'speed_min':     speed_min,
        'speed_mean':    speed_mean,
        'speed_entry':   speed_entry,
        'speed_exit':    speed_exit,
        'speed_drop':    speed_drop,
        'brake_frac':    float(seg_brake.mean()),
        'throttle_mean': float(seg_throttle.mean()),
        'kappa_max':     kappa_max,
        'kappa_mean':    kappa_mean,
        'lat_g_max':     lat_g_max,
        'gear_apex':     float(seg_gear[apex_local]),
        'dir_encoded':   1 if corner['direction'] == 'L' else -1,
        'peak_kappa':    float(corner['peak_kappa']),
        'steer_max':     steer_max,
        'steer_mean':    steer_mean,
        'steer_apex':    steer_apex,
        'corner_type':   corner_type,
        'detected_by':   corner.get('detected_by', 'curvature'),
    }
