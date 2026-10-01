"""Synthetic 4-TI x 3-plane IR-FSE brain phantom for comparing reconstructions.

The object is defined analytically in the scanner frame (X = S/I, Y = L/R, Z = A/P, mm,
origin at isocentre), so all three planes see exactly the same head. For each scan in
``acq_spec/acq_spec.json`` it is sampled on a grid finer than the reconstruction grid
(``FACTOR``), weighted by synthetic coil sensitivities, and taken to k-space with BART's
NUFFT on that scan's real sampling coordinates. Simulating on a finer grid than the one
reconstructed avoids the "inverse crime" of generating and reconstructing on one grid.

Output layout matches the real-data pipeline (``T1map_3plane_SR_V1.ipynb``):
``<out>/CFL/<PLANE>_TI<ms>_synth_{data,traj}.cfl`` with BART dims
data (1, read, pe, coil, 1, echo) and traj (3, read, pe, 1, 1, echo), plus ``_info.json``.
Ground truth goes to ``<out>/truth_<PLANE>.npz`` (reconstruction grid) and
``<out>/truth_iso.npz`` (1.8 mm isotropic grid). ``truth_<PLANE>.npz`` also holds a
resolution-limited reference: ``S_ideal`` (the true signal at each TI with k-space
truncated to the encoded extent, fully sampled, noise-free) and ``T1_ideal_ms`` fitted from
it. Bias relative to ``T1_ideal_ms`` is attributable to the reconstruction alone; bias of
``T1_ideal_ms`` relative to ``T1_ms`` is the unavoidable partial-volume effect.

Tissue T1 (ms) at 64 mT, varying smoothly in space within these ranges:
WM 250-300, GM 310-370, CSF 3400-3700 (literature ranges supplied by the user; within each
tissue the smooth field gives a bell-shaped distribution centred mid-range, SD ~15 % of the
range, with its tails at the bounds).
Fat (180 ms) and bone are placeholders. Lesions: spheres of 2, 4, 6 and 10 mm diameter
in white matter with T1 = 1.3 x and 0.7 x the mid-range WM T1 (357.5 and 192.5 ms), M0 equal
to WM, so they differ from WM in T1 only.

Signal model (the one the reconstructions fit):
    S(TI) = M0 * (1 - 2 exp(-TI/T1) + exp(-TR/T1)),  TR from the header (TR = TI + 1.2 s)
Phase: a smooth background phase common to all scans, a per-scan global phase offset
(``TI_PHASE_SD_DEG``, measured 4-9 deg between TI scans on the real data), and an extra
smooth phase on the second echo group (FSE even/odd echoes).
Noise: complex Gaussian, SD = ``NOISE_RATIO`` x RMS of that plane's TI800 k-space (0.18,
measured on the real data from repeated phase-encode samples).

Usage:  python phantom.py <out_dir> [--factor 2 2 4] [--seed 0] [--noise 0.18] [--motion]

``--motion`` applies known rigid head motion per scan (``MOTION_PRESET``: drifts within each
plane, a 2 mm step before axial TI800, and larger offsets between planes). The poses are
saved in ``phantom_config.json``; each plane's truth is at its TI800 pose, ``truth_iso.npz``
is the unmoved head. Without the flag the head is static.
Requires BART (BART_TOOLBOX_PATH set, or /usr/local/bart, with its python/ wrapper).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AXIS = {'X': 0, 'Y': 1, 'Z': 2}

BART_PATH = os.environ.get('BART_TOOLBOX_PATH', '/usr/local/bart')
os.environ['BART_TOOLBOX_PATH'] = BART_PATH
sys.path.append(os.path.join(BART_PATH, 'python'))
from bart import bart, cfl  # noqa: E402
from pipeline_utils import fit_t1_grid  # noqa: E402

FACTOR = (2, 2, 4)            # fine-grid factor per encoding axis (read, pe1, pe2)
NOISE_RATIO = 0.18
TI_PHASE_SD_DEG = 5.0

# ---- tissues: label -> (name, T1 range ms, M0) --------------------------------------
TISSUE = {
    0: ('background', (1.0, 1.0), 0.0),
    1: ('fat', (180.0, 180.0), 0.90),     # placeholder T1
    2: ('bone', (300.0, 300.0), 0.05),    # placeholder, little signal
    3: ('CSF', (3400.0, 3700.0), 1.00),
    4: ('GM', (310.0, 370.0), 0.80),
    5: ('WM', (250.0, 300.0), 0.65),
    6: ('lesion_long', (357.5, 357.5), 0.65),    # 1.3 x 275 ms (WM mid-range)
    7: ('lesion_short', (192.5, 192.5), 0.65),   # 0.7 x 275 ms
}
# Nested head ellipsoids, semi-axes in (L/R, A/P, S/I) mm; each shell is (outer, label).
SHELLS = [((76, 96, 88), 1),   # scalp fat
          ((71, 91, 83), 2),   # skull
          ((66, 86, 78), 3),   # CSF
          ((63, 83, 75), 4),   # cortical GM
          ((58, 78, 70), 5)]   # WM
# Extra structures: (label, centre (LR, AP, SI), semi-axes (LR, AP, SI), rotation about S/I deg)
STRUCTS = [(3, (-10, 5, 10), (6, 22, 10), 15), (3, (10, 5, 10), (6, 22, 10), -15),   # ventricles
           (4, (-20, -5, -2), (9, 14, 10), 0), (4, (20, -5, -2), (9, 14, 10), 0)]    # deep GM
# Lesions: (label, centre (LR, AP, SI), diameter mm)
LESIONS = [(6, (-30, 20, 35), 10), (6, (-30, -15, 35), 6), (6, (-36, 35, 15), 4), (6, (-36, -40, 15), 2),
           (7, (30, 20, 35), 10), (7, (30, -15, 35), 6), (7, (36, 35, 15), 4), (7, (36, -40, 15), 2)]


# Known rigid head motion per scan (object pose in the scanner frame), used only with
# --motion. rot_deg = rotations about scanner X (S/I), Y (L/R), Z (A/P); trans_mm = shift
# along X, Y, Z. Within each plane: small drifts and, for the axial plane, a 2 mm step before
# TI800 (as measured on the real data); between planes: larger offsets. Coils stay fixed to
# the scanner (the head moves inside the coil), as do the background and echo phases.
MOTION_PRESET = {
    'AX_TI31':   dict(rot_deg=(0.3, 0.0, 0.2),  trans_mm=(0.4, -0.3, 0.2)),
    'AX_TI150':  dict(rot_deg=(0.5, -0.2, 0.3), trans_mm=(0.6, -0.5, 0.4)),
    'AX_TI400':  dict(rot_deg=(0.7, -0.3, 0.4), trans_mm=(0.8, -0.6, 0.5)),
    'AX_TI800':  dict(rot_deg=(0.5, -0.6, 1.2), trans_mm=(2.0, 0.8, -1.4)),
    'COR_TI31':  dict(rot_deg=(1.5, 0.4, -0.5), trans_mm=(-1.8, 1.9, 1.0)),
    'COR_TI150': dict(rot_deg=(1.6, 0.4, -0.6), trans_mm=(-1.9, 2.1, 1.1)),
    'COR_TI400': dict(rot_deg=(1.6, 0.5, -0.6), trans_mm=(-2.0, 2.2, 1.2)),
    'COR_TI800': dict(rot_deg=(1.7, 0.5, -0.7), trans_mm=(-2.1, 2.3, 1.2)),
    'SAG_TI31':  dict(rot_deg=(-1.0, 1.8, 0.6), trans_mm=(1.2, -2.4, -2.8)),
    'SAG_TI150': dict(rot_deg=(-1.1, 2.0, 0.6), trans_mm=(1.3, -2.6, -3.0)),
    'SAG_TI400': dict(rot_deg=(-1.2, 2.1, 0.7), trans_mm=(1.4, -2.7, -3.1)),
    'SAG_TI800': dict(rot_deg=(-1.3, 2.2, 0.7), trans_mm=(1.5, -2.8, -3.2)),
}


def rigid_matrix(rot_deg, trans_mm):
    """4x4 object-to-scanner transform: p_scanner = R p_object + t, R = Rz Ry Rx
    (rotations about scanner X = S/I, Y = L/R, Z = A/P)."""
    ax, ay, az = np.deg2rad(rot_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    M = np.eye(4); M[:3, :3] = Rz @ Ry @ Rx; M[:3, 3] = trans_mm
    return M


def to_object(X, Y, Z, M):
    """Scanner-frame coordinates -> object-frame coordinates of a head posed by M."""
    if M is None:
        return X, Y, Z
    Ri = M[:3, :3].T; t = M[:3, 3]
    P = np.stack([X - t[0], Y - t[1], Z - t[2]], 0).reshape(3, -1)
    Q = (Ri @ P).reshape((3,) + X.shape)
    return Q[0].astype(np.float32), Q[1].astype(np.float32), Q[2].astype(np.float32)


def load_spec():
    return json.load(open(HERE / 'acq_spec' / 'acq_spec.json'))


def bart_traj(coord, matrix, n_read, echo):
    """(3, read, pe) trajectory in BART units, identical to the real-data converter."""
    p1, p2 = matrix[1], matrix[2]
    t = np.zeros((3, n_read, coord.shape[1]), np.float32)
    t[0] = (np.arange(n_read) - n_read // 2)[:, None]
    t[1] = p1 * coord[0, :, echo][None]
    t[2] = p2 * coord[1, :, echo][None]
    return t.astype(np.complex64)


def grid_coords(matrix, fov_mm, geom, factor=(1, 1, 1), centred=False):
    """Scanner-frame coordinates (X, Y, Z in mm) of every voxel of a (possibly refined)
    encoding grid. Voxel n//2 sits at isocentre, as in BART's image convention.
    centred=True shifts a refined grid by -(f-1)/2 fine samples per axis, so that each block of
    f fine samples is centred on its reconstruction voxel (for block-mean truth maps)."""
    n = [m * f for m, f in zip(matrix, factor)]
    sp = [fv / nn for fv, nn in zip(fov_mm, n)]
    sh = [-(f - 1) / 2 * s if centred else 0.0 for f, s in zip(factor, sp)]
    ax = [(np.arange(nn) - nn // 2) * s + o for nn, s, o in zip(n, sp, sh)]
    A = np.meshgrid(*ax, indexing='ij')
    xyz = [None, None, None]
    for j, a in enumerate(geom):
        xyz[AXIS[a]] = A[j]
    return xyz  # X (S/I), Y (L/R), Z (A/P)


def _smooth_field(X, Y, Z, seed, n=6, period_mm=45.0):
    """Grid-independent smooth field in [0, 1]: a fixed sum of random 3D cosines."""
    rng = np.random.default_rng(seed)
    f = np.zeros_like(X, dtype=np.float32)
    amp = rng.uniform(0.5, 1.0, n)
    for k in range(n):
        d = rng.normal(size=3); d /= np.linalg.norm(d)
        w = 2 * np.pi / (period_mm * rng.uniform(0.7, 1.5))
        f += amp[k] * np.cos(w * (d[0] * X + d[1] * Y + d[2] * Z) + rng.uniform(0, 2 * np.pi))
    return 0.5 + 0.5 * f / amp.sum()


def tissue_maps(X, Y, Z, seed=0):
    """Label, T1 (ms), M0 and a lesion id on the given coordinates."""
    lr, ap, si = Y, Z, X
    lab = np.zeros(X.shape, np.int8)
    for (a, b, c), l in SHELLS:
        lab[(lr / a) ** 2 + (ap / b) ** 2 + (si / c) ** 2 <= 1] = l
    brain = lab >= 4
    for l, (c0, c1, c2), (a, b, c), rot in STRUCTS:
        t = np.deg2rad(rot)
        u = np.cos(t) * (lr - c0) + np.sin(t) * (ap - c1)
        v = -np.sin(t) * (lr - c0) + np.cos(t) * (ap - c1)
        lab[brain & ((u / a) ** 2 + (v / b) ** 2 + ((si - c2) / c) ** 2 <= 1)] = l
    les_id = np.zeros(X.shape, np.int8)
    for i, (l, (c0, c1, c2), d) in enumerate(LESIONS):
        m = (lr - c0) ** 2 + (ap - c1) ** 2 + (si - c2) ** 2 <= (d / 2) ** 2
        lab[m] = l; les_id[m] = i + 1
    T1 = np.zeros(X.shape, np.float32); M0 = np.zeros(X.shape, np.float32)
    field = _smooth_field(X, Y, Z, seed)
    for l, (_, (lo, hi), m0) in TISSUE.items():
        m = lab == l
        T1[m] = lo + (hi - lo) * field[m]
        M0[m] = m0
    return lab, T1, M0, les_id


def coil_maps(X, Y, Z, n_coils=8, seed=1):
    """Smooth synthetic receive sensitivities: loops on a helmet around the head."""
    rng = np.random.default_rng(seed)
    lr, ap, si = Y, Z, X
    pos = [(100 * np.cos(a), 120 * np.sin(a), 10) for a in np.linspace(0, 2 * np.pi, 5)[:-1]]
    pos += [(70 * np.cos(a), 85 * np.sin(a), 80) for a in np.linspace(0, 2 * np.pi, 4)[:-1] + 0.5]
    pos += [(0, 0, 115)]
    S = np.zeros(X.shape + (n_coils,), np.complex64)
    for c, (p0, p1, p2) in enumerate(pos[:n_coils]):
        d2 = (lr - p0) ** 2 + (ap - p1) ** 2 + (si - p2) ** 2
        mag = 1.0 / (1.0 + d2 / 80.0 ** 2) ** 1.5
        u = rng.normal(size=3); u /= np.linalg.norm(u)
        ph = 2 * np.pi * (u[0] * lr + u[1] * ap + u[2] * si) / 300.0 + rng.uniform(0, 2 * np.pi)
        S[..., c] = mag * np.exp(1j * ph)
    return S


def background_phase(X, Y, Z):
    return (0.6 * X / 100 + 0.4 * (Y / 100) ** 2 - 0.3 * Z / 100).astype(np.float32)


def ir_signal(M0, T1_ms, TI_s, TR_s):
    T1 = np.maximum(T1_ms, 1e-3) / 1000.0
    return M0 * (1 - 2 * np.exp(-TI_s / T1) + np.exp(-TR_s / T1))


def block_mean(a, factor):
    s = a.shape
    return a.reshape(s[0] // factor[0], factor[0], s[1] // factor[1], factor[1],
                     s[2] // factor[2], factor[2]).mean(axis=(1, 3, 5))


def simulate(out_dir, factor=FACTOR, seed=0, noise_ratio=NOISE_RATIO,
             ti_phase_sd_deg=TI_PHASE_SD_DEG, motion=None, verbose=True):
    """motion: None (static head) or {scan name: dict(rot_deg, trans_mm)}, e.g. MOTION_PRESET."""
    out = Path(out_dir); (out / 'CFL').mkdir(parents=True, exist_ok=True)
    spec = load_spec()
    rng = np.random.default_rng(seed + 100)
    by_plane = {}
    for s in spec['scans']:
        by_plane.setdefault(s['plane'], []).append(s)
    for plane, scans in by_plane.items():
        scans.sort(key=lambda s: s['TI_s'])
        s0 = scans[0]
        X, Y, Z = grid_coords(s0['matrix'], s0['fov_mm'], s0['geometry'], factor)
        pose = {s['name']: (rigid_matrix(**motion[s['name']]) if motion else None) for s in scans}
        ref_name = [s['name'] for s in scans if s['name'].endswith('TI800')][0]
        lab, T1, M0, les = tissue_maps(*to_object(X, Y, Z, pose[ref_name]), seed)
        S = coil_maps(X, Y, Z, spec['n_coils'])
        bg = background_phase(X, Y, Z)
        # odd-echo extra phase: linear along this plane's readout axis
        read_ax = [X, Y, Z][AXIS[s0['geometry'][0]]]
        echo_phase = [np.zeros_like(bg), (0.4 + 0.01 * read_ax).astype(np.float32)]
        ksp, ideal = {}, []
        for s in scans:
            dphi = np.deg2rad(rng.normal(0, ti_phase_sd_deg))
            if motion:                      # this scan's own head pose
                _, T1s, M0s, _ = tissue_maps(*to_object(X, Y, Z, pose[s['name']]), seed)
            else:
                T1s, M0s = T1, M0
            sig = ir_signal(M0s, T1s, s['TI_s'], s['TR_s']).astype(np.float32)
            coord = np.load(HERE / 'acq_spec' / s['coord_file'])
            D, Tr = [], []
            for e in range(s['n_echo_groups']):
                img = sig * np.exp(1j * (bg + echo_phase[e] + dphi))
                t = bart_traj(coord, s['matrix'], s['n_read'], e)
                k = np.stack([np.squeeze(bart(1, 'nufft', t, (img * S[..., c]).astype(np.complex64)))
                              for c in range(S.shape[-1])], axis=-1)     # (read, pe, coil)
                D.append(k); Tr.append(t)
            ksp[s['name']] = (np.stack(D, -1), np.stack(Tr, -1), dphi)    # (read, pe, coil, echo)
            ideal.append(kspace_truncate(sig, s['matrix']))
            if verbose:
                print(f'{plane} TI={1000 * s["TI_s"]:.0f} ms simulated', flush=True)
        ref = [n for n in ksp if n.endswith('TI800')][0]
        sigma = noise_ratio * np.sqrt(np.mean(np.abs(ksp[ref][0][..., 0]) ** 2))
        for s in scans:
            D, Tr, dphi = ksp[s['name']]
            D = D + sigma / np.sqrt(2) * (rng.normal(size=D.shape) + 1j * rng.normal(size=D.shape))
            stem = out / 'CFL' / f'{s["name"]}_synth'
            cfl.writecfl(f'{stem}_data', D[None, :, :, :, None, :].astype(np.complex64))
            cfl.writecfl(f'{stem}_traj', Tr[:, :, :, None, None, :].astype(np.complex64))
            info = {'fov_mm': s['fov_mm'], 'matrix': s['matrix'], 'geom': s['geometry'],
                    'TI_s': s['TI_s'], 'TR_s': s['TR_s'], 'etl_s': s['etl_duration_s'],
                    'R_header': s['undersampling_header'], 'n_pe': s['n_pe'],
                    'acq_time': 'synthetic', 'noise_sd': float(sigma), 'ti_phase_rad': float(dphi)}
            json.dump(info, open(f'{stem}_info.json', 'w'), indent=2)
        # ground truth on the reconstruction grid of this plane. Partial-volume quantities are
        # block means over a fine grid whose blocks are CENTRED on the reconstruction voxels
        # (the simulation grid's blocks are offset by (f-1)/2 fine samples).
        Xb, Yb, Zb = grid_coords(s0['matrix'], s0['fov_mm'], s0['geometry'], factor, centred=True)
        _, T1b, _, lesb = tissue_maps(*to_object(Xb, Yb, Zb, pose[ref_name]), seed)
        Sb = coil_maps(Xb, Yb, Zb, spec['n_coils'])
        Xc, Yc, Zc = grid_coords(s0['matrix'], s0['fov_mm'], s0['geometry'])
        labc, T1c, M0c, lesc = tissue_maps(*to_object(Xc, Yc, Zc, pose[ref_name]), seed)
        S_ideal = np.stack(ideal, -1)                                   # (X, Y, Z, TI)
        TIa = np.array([s['TI_s'] for s in scans]); TRa = np.array([s['TR_s'] for s in scans])
        T1_ideal, _, _ = fit_t1_grid(S_ideal, TIa, TRa, labc >= 3)
        np.savez_compressed(out / f'truth_{plane}.npz', label=labc, T1_ms=T1c, M0=M0c, lesion_id=lesc,
                            S_ideal=S_ideal.astype(np.float32), T1_ideal_ms=1000 * T1_ideal,
                            lesion_fraction=block_mean((lesb > 0).astype(np.float32), factor),
                            lesion_frac_by_id=np.stack([block_mean((lesb == i + 1).astype(np.float32),
                                                                   factor) for i in range(len(LESIONS))]),
                            T1_ms_pv=block_mean(T1b, factor), sens=block_mean_c(Sb, factor),
                            noise_sd=sigma,
                            pose_ref=pose[ref_name] if motion else np.eye(4),
                            pose_note='truth is at this plane\'s TI800 head pose (object-to-scanner 4x4)')
        if verbose:
            print(f'{plane}: noise SD {sigma:.3g}, truth saved', flush=True)
    # 1.8 mm isotropic truth (scanner frame, voxel n//2 at isocentre)
    n = 128
    ax = (np.arange(n) - n // 2) * 1.8
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing='ij')
    lab, T1, M0, les = tissue_maps(X, Y, Z, seed)
    np.savez_compressed(out / 'truth_iso.npz', label=lab, T1_ms=T1, M0=M0, lesion_id=les,
                        spacing_mm=1.8, note='axes X (S/I), Y (L/R), Z (A/P); voxel 64 at isocentre')
    json.dump({'tissues': {str(k): v for k, v in TISSUE.items()}, 'lesions': LESIONS,
               'factor': list(factor), 'seed': seed, 'noise_ratio': noise_ratio,
               'ti_phase_sd_deg': ti_phase_sd_deg,
               'motion': ({k: dict(v, matrix=rigid_matrix(**v).tolist()) for k, v in motion.items()}
                          if motion else None),
               'motion_note': 'object-to-scanner pose per scan: p_scanner = R p_object + t; '
                              'truth_iso.npz is the unmoved head (identity pose)'},
              open(out / 'phantom_config.json', 'w'), indent=1)


def kspace_truncate(img_fine, matrix):
    """Resolution-limited reference: keep only the k-space extent the protocol encodes
    (the reconstruction matrix), fully sampled and noise-free, and return the real image
    on the reconstruction grid. Voxel n//2 at isocentre on both grids (centred FFTs)."""
    K = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img_fine)))
    c = [nf // 2 - n // 2 for nf, n in zip(img_fine.shape, matrix)]
    K = K[c[0]:c[0] + matrix[0], c[1]:c[1] + matrix[1], c[2]:c[2] + matrix[2]]
    x = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(K)))
    return (np.real(x) * np.prod(matrix) / np.prod(img_fine.shape)).astype(np.float32)


def block_mean_c(S, factor):
    return np.stack([block_mean(S[..., c].real, factor) + 1j * block_mean(S[..., c].imag, factor)
                     for c in range(S.shape[-1])], -1).astype(np.complex64)


def load_scans(out_dir):
    """{plane: [ {'name', 'stem', 'info'}, ... sorted by TI ]}, like the notebook's `scans`."""
    out = Path(out_dir); scans = {}
    for f in sorted((out / 'CFL').glob('*_synth_info.json')):
        name = f.name.replace('_synth_info.json', '')
        info = json.load(open(f))
        scans.setdefault(name.split('_')[0], []).append(
            {'name': name, 'stem': out / 'CFL' / f'{name}_synth', 'info': info})
    for p in scans:
        scans[p].sort(key=lambda s: s['info']['TI_s'])
    return scans


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--factor', type=int, nargs=3, default=list(FACTOR))
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--noise', type=float, default=NOISE_RATIO)
    ap.add_argument('--ti-phase-sd', type=float, default=TI_PHASE_SD_DEG)
    ap.add_argument('--motion', action='store_true',
                    help='apply the known rigid head motion MOTION_PRESET (per scan)')
    a = ap.parse_args()
    simulate(a.out_dir, tuple(a.factor), a.seed, a.noise, a.ti_phase_sd,
             MOTION_PRESET if a.motion else None)
