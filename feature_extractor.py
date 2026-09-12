"""
Python port of export_features.m
Extracts 30 morphological / texture features from a binary vessel mask.
"""
import numpy as np
import cv2
from skimage.morphology import skeletonize, binary_dilation, disk, square
from skimage.measure import label, regionprops
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans


def _bwskel(bw):
    return skeletonize(bw.astype(bool))


def _bwmorph_branchpoints(skel):
    skel_u8 = skel.astype(np.uint8)
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nbr = cv2.filter2D(skel_u8, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return (skel_u8 == 1) & (nbr >= 3)


def _bwmorph_endpoints(skel):
    skel_u8 = skel.astype(np.uint8)
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
    nbr = cv2.filter2D(skel_u8, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return (skel_u8 == 1) & (nbr == 1)


def _order_path_pixels(pts, img_shape):
    if len(pts) < 2:
        return pts
    n = len(pts)
    pts_arr = np.array(pts)
    idx_map = {}
    for i, (r, c) in enumerate(pts_arr):
        idx_map[(r, c)] = i
    neighbors = [[] for _ in range(n)]
    for i, (r, c) in enumerate(pts_arr):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                key = (r + dr, c + dc)
                if key in idx_map:
                    neighbors[i].append(idx_map[key])
    deg = np.array([len(nb) for nb in neighbors])
    start = int(np.argmax(deg == 1)) if np.any(deg == 1) else 0
    visited = np.zeros(n, dtype=bool)
    order = []
    cur = start
    while cur is not None and not visited[cur]:
        visited[cur] = True
        order.append(cur)
        nxt = [nb for nb in neighbors[cur] if not visited[nb]]
        cur = nxt[0] if nxt else None
    return pts_arr[order]


def _segment_geometry(ordered):
    if ordered.shape[0] < 2:
        return 0.0, 0.0, np.array([]), np.array([])
    diffs = np.diff(ordered, axis=0)
    step_lens = np.sqrt((diffs ** 2).sum(axis=1))
    arc_len = float(step_lens.sum())
    chord_len = float(np.linalg.norm(ordered[-1] - ordered[0]))
    if ordered.shape[0] < 3:
        return arc_len, chord_len, np.array([]), np.array([])
    v1 = diffs[:-1]; v2 = diffs[1:]
    dotp = (v1 * v2).sum(axis=1)
    crossz = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    signed_angles = np.arctan2(crossz, dotp)
    n1 = np.sqrt((v1 ** 2).sum(axis=1))
    n2 = np.sqrt((v2 ** 2).sum(axis=1))
    curv = signed_angles / ((n1 + n2) / 2 + 1e-12)
    return arc_len, chord_len, curv, signed_angles


def _trace_skeleton_segments(skel_no_branch):
    lbl = label(skel_no_branch, connectivity=2)
    segs = []
    for region in regionprops(lbl):
        coords = region.coords
        if len(coords) < 2:
            continue
        ordered = _order_path_pixels(coords, skel_no_branch.shape)
        if len(ordered) >= 2:
            segs.append(ordered)
    return segs


def _lacunarity_index(bw):
    sizes = [10, 20, 40]
    lacs = []
    for r in sizes:
        if r >= min(bw.shape):
            continue
        kernel = np.ones((r, r), dtype=np.float32)
        mass = cv2.filter2D(bw.astype(np.float32), -1, kernel,
                            borderType=cv2.BORDER_CONSTANT)
        m = mass.mean(); v = mass.var()
        lacs.append((v / (m ** 2 + 1e-12)) + 1.0)
    return float(np.mean(lacs)) if lacs else 0.0


def _density_features(bw):
    rows, cols = bw.shape
    F01 = bw.sum() / bw.size
    r_mid, c_mid = rows // 2, cols // 2
    q1, q2 = bw[:r_mid, :c_mid], bw[:r_mid, c_mid:]
    q3, q4 = bw[r_mid:, :c_mid], bw[r_mid:, c_mid:]
    qd = [q.sum() / max(q.size, 1) for q in (q1, q2, q3, q4)]
    F02 = max(qd) / (min(qd) + 1e-12)
    Y, X = np.mgrid[0:rows, 0:cols]
    cx, cy = cols / 2, rows / 2
    R = 0.4 * min(rows, cols) / 2
    central = ((X - cx) ** 2 + (Y - cy) ** 2) <= R ** 2
    central_d = (bw & central).sum() / max(central.sum(), 1)
    periph_d = (bw & ~central).sum() / max((~central).sum(), 1)
    F03 = central_d / (periph_d + 1e-12)
    skel = _bwskel(bw)
    F04 = skel.sum() / bw.size
    F05 = _lacunarity_index(bw)
    return np.array([F01, F02, F03, F04, F05], dtype=np.float32)


def _caliber_features(bw, skel):
    dist = cv2.distanceTransform(bw.astype(np.uint8), cv2.DIST_L2, 5)
    widths = 2 * dist[skel]
    widths = widths[widths > 0]
    if widths.size == 0:
        return np.array([0, 0, 0, 0, 0], dtype=np.float32)
    F06 = float(widths.mean()); F07 = float(widths.std()); F08 = float(widths.max())
    if widths.size >= 10:
        try:
            km = KMeans(n_clusters=2, n_init=3, random_state=0).fit(widths.reshape(-1, 1))
            m1 = widths[km.labels_ == 0].mean(); m2 = widths[km.labels_ == 1].mean()
            F09 = max(m1, m2) / (min(m1, m2) + 1e-12)
        except Exception:
            F09 = 0.0
    else:
        F09 = 0.0
    F10 = F07 / (F06 + 1e-12)
    return np.array([F06, F07, F08, F09, F10], dtype=np.float32)


def _tortuosity_features(arc_lens, chord_lens, seg_stats, img_area):
    n = len(arc_lens)
    if n == 0:
        return np.zeros(5, dtype=np.float32)
    arc_lens = np.asarray(arc_lens, dtype=np.float64)
    chord_lens = np.asarray(chord_lens, dtype=np.float64)
    valid = chord_lens > 1e-6
    ac = np.ones(n); ac[valid] = arc_lens[valid] / chord_lens[valid]
    curv_idx = np.zeros(n); inflect = np.zeros(n); bend = np.zeros(n)
    for k in range(n):
        c = seg_stats[k]["curvatures"]; sa = seg_stats[k]["signed_angles"]
        if c.size > 0:
            curv_idx[k] = np.mean(np.abs(c))
            bend[k] = np.sum(c ** 2) / max(arc_lens[k], 1e-12)
        if sa.size >= 2:
            s = np.sign(sa); s[np.abs(sa) < np.deg2rad(2)] = 0; s = s[s != 0]
            if s.size >= 2:
                inflect[k] = np.sum(np.diff(s) != 0)
    return np.array([float(ac.mean()), float(curv_idx.mean()), float(inflect.mean()),
                     float(np.sum(np.maximum(ac - 1, 0)) / max(img_area, 1) * 1e5),
                     float(bend.mean())], dtype=np.float32)


def _box_count_fractal_dim(bw):
    bw = bw.astype(bool)
    max_pow = min(int(np.floor(np.log2(min(bw.shape)))), 8)
    if max_pow < 2:
        return 0.0
    sizes = 2 ** np.arange(1, max_pow + 1)
    counts = np.zeros(len(sizes), dtype=np.float64)
    for si, s in enumerate(sizes):
        c = 0
        for r in range(0, bw.shape[0], s):
            for cc in range(0, bw.shape[1], s):
                if bw[r:r + s, cc:cc + s].any():
                    c += 1
        counts[si] = max(c, 1)
    p = np.polyfit(np.log(1.0 / sizes), np.log(counts), 1)
    return float(p[0])


def _mean_branching_angle(skel, branch_pts):
    R = 15
    branch_dil = binary_dilation(branch_pts, square(3))
    skel_stub = skel & ~branch_dil
    angles = []
    lbl_b = label(branch_dil, connectivity=2)
    rows, cols = skel.shape
    for region in regionprops(lbl_b):
        cy, cx = region.centroid
        rmin = max(0, int(round(cy - R))); rmax = min(rows, int(round(cy + R)))
        cmin = max(0, int(round(cx - R))); cmax = min(cols, int(round(cx + R)))
        local_mask = np.zeros_like(skel, dtype=bool)
        local_mask[rmin:rmax, cmin:cmax] = True
        local_skel = skel_stub & local_mask
        lbl_s = label(local_skel, connectivity=2)
        dirs = []
        for reg_s in regionprops(lbl_s):
            coords = reg_s.coords
            d2 = (coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2
            m = coords[np.argmax(d2)]
            v = np.array([m[1] - cx, m[0] - cy], dtype=np.float64)
            nv = np.linalg.norm(v)
            if nv < 1e-6:
                continue
            dirs.append(v / nv)
        dirs = np.array(dirs)
        if len(dirs) < 2:
            continue
        for a in range(len(dirs) - 1):
            for b in range(a + 1, len(dirs)):
                cosA = np.clip(np.dot(dirs[a], dirs[b]), -1, 1)
                angles.append(np.degrees(np.arccos(cosA)))
    return float(np.mean(angles)) if angles else 0.0


def _branching_features(bw, skel, branch_pts, end_pts):
    branch_dil = binary_dilation(branch_pts, disk(1))
    lbl_b = label(branch_dil, connectivity=2)
    F16 = float(lbl_b.max())
    F17 = F16 / max(bw.sum(), 1) * 1e4
    F18 = _mean_branching_angle(skel, branch_pts)
    F19 = _box_count_fractal_dim(bw)
    lbl_e = label(end_pts, connectivity=2)
    F20 = float(lbl_e.max())
    return np.array([F16, F17, F18, F19, F20], dtype=np.float32)


def _network_features(arc_lens, img_area):
    n = len(arc_lens)
    if n == 0:
        return np.zeros(5, dtype=np.float32)
    arc = np.asarray(arc_lens, dtype=np.float64)
    return np.array([float(n), float(arc.mean()),
                     float(arc.sum() / max(img_area, 1) * 1e3),
                     float(arc.max()), float(arc.std())], dtype=np.float32)


def _hist_entropy(img):
    hist, _ = np.histogram(img, bins=256, range=(0, 255))
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _texture_features(bw, orig_gray=None):
    if orig_gray is not None:
        img_masked = orig_gray.copy(); img_masked[~bw] = 0
        vessel_pixels = orig_gray[bw].astype(np.float64)
        img8 = (img_masked.astype(np.float32) / 255.0 * 7).astype(np.uint8)
        glcm = graycomatrix(img8, distances=[1], angles=[0],
                            levels=8, symmetric=True, normed=True)
        F26 = float(graycoprops(glcm, "contrast")[0, 0])
        F27 = float(graycoprops(glcm, "correlation")[0, 0])
        F28 = float(skew(vessel_pixels)) if vessel_pixels.size > 1 else 0.0
        F29 = float(kurtosis(vessel_pixels)) if vessel_pixels.size > 1 else 0.0
        F30 = _hist_entropy(img_masked)
    else:
        dist = cv2.distanceTransform(bw.astype(np.uint8), cv2.DIST_L2, 5)
        dm8 = (dist / (dist.max() + 1e-12) * 7).astype(np.uint8)
        glcm = graycomatrix(dm8, distances=[1], angles=[0],
                            levels=8, symmetric=True, normed=True)
        F26 = float(graycoprops(glcm, "contrast")[0, 0])
        F27 = float(graycoprops(glcm, "correlation")[0, 0])
        vv = dm8[bw].astype(np.float64)
        F28 = float(skew(vv)) if vv.size > 1 else 0.0
        F29 = float(kurtosis(vv)) if vv.size > 1 else 0.0
        F30 = _hist_entropy(dm8)
    return np.array([F26, F27, F28, F29, F30], dtype=np.float32)


def extract_all_features(bw, orig_gray=None, min_segment_len=8):
    bw = bw.astype(bool)
    img_area = bw.size
    skel = _bwskel(bw)
    branch_pts = _bwmorph_branchpoints(skel)
    end_pts = _bwmorph_endpoints(skel)
    branch_dil = binary_dilation(branch_pts, square(3))
    skel_no_branch = skel & ~branch_dil
    segments = _trace_skeleton_segments(skel_no_branch)
    arc_lens, chord_lens, seg_stats = [], [], []
    for seg in segments:
        a, c, curv, sang = _segment_geometry(seg)
        if a >= min_segment_len:
            arc_lens.append(a); chord_lens.append(c)
            seg_stats.append({"curvatures": curv, "signed_angles": sang})
    og = None
    if orig_gray is not None:
        og = orig_gray.astype(np.uint8) if orig_gray.dtype != np.uint8 else orig_gray
    A = _density_features(bw)
    B = _caliber_features(bw, skel)
    C = _tortuosity_features(arc_lens, chord_lens, seg_stats, img_area)
    D = _branching_features(bw, skel, branch_pts, end_pts)
    E = _network_features(arc_lens, img_area)
    F = _texture_features(bw, og)
    feats = np.concatenate([A, B, C, D, E, F]).astype(np.float32)
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


FEATURE_NAMES = [
    "F01_TotalVesselDensity", "F02_QuadrantDensityRatio", "F03_CentralPeripheralDensity",
    "F04_SkeletonPixelDensity", "F05_LacunarityIndex",
    "F06_MeanVesselWidth", "F07_StdVesselWidth", "F08_MaxVesselWidth",
    "F09_AVCaliberRatioEst", "F10_CaliberVariabilityIndex",
    "F11_ArcChordRatioMean", "F12_CurvatureIndex", "F13_InflectionCountPerVessel",
    "F14_TortuosityDensity", "F15_BendingEnergy",
    "F16_JunctionCount", "F17_BranchingDensity", "F18_MeanBranchingAngleDeg",
    "F19_FractalDimension", "F20_EndPointCount",
    "F21_SegmentCount", "F22_MeanSegmentLength", "F23_TotalVesselLengthDensity",
    "F24_LongestSegment", "F25_SegmentLengthStd",
    "F26_GLCMContrast", "F27_GLCMCorrelation", "F28_IntensitySkewness",
    "F29_IntensityKurtosis", "F30_MaskEntropy",
]