#!/usr/bin/env python3
"""
reconstruct_optimal.py

Robust jumbled-frame single-shot video reconstruction.

Key ideas:
 - ResNet-18 pretrained embeddings (auto-fallback to HSV hist if torch missing)
 - Keep top-k nearest neighbors per frame (k-candidates)
 - Compute directed ORB displacement + optical-flow magnitude only on candidate edges
 - Build directed cost for edges: cost = w_embed * (1 - cosine_sim) + w_flow * flow_mag + w_dir * dir_penalty
 - Order using: spectral init -> beam search over candidate graph -> 2-opt -> simulated annealing -> flow smoothing
 - Auto-detect and fix reversed ordering; optional rotate-last-seconds-to-front.

Usage:
  python reconstruct_optimal.py --video jumbled_video.mp4 --out reconstructed_optimal.mp4

Dependencies:
  pip install opencv-python numpy tqdm scikit-learn scipy
  Optional (recommended): pip install torch torchvision  # for ResNet embeddings

Notes:
  - This script is tuned to handle a wide variety of single-shot videos.
  - For very long videos (N >> 500) reduce k or downsample frames for performance.
"""
import argparse, os, sys, math, time, shutil, random
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm

# -------------------------
# Helpers
# -------------------------
def extract_frames(video_path, frames_dir="frames_opt"):
    frames_dir = Path(frames_dir)
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    i = 0
    pbar = tqdm(total=count, desc="Extract frames")
    while True:
        ret, f = cap.read()
        if not ret:
            break
        outp = frames_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(outp), f)
        i += 1
        pbar.update(1)
    pbar.close()
    cap.release()
    frame_paths = sorted([str(p) for p in frames_dir.iterdir() if p.suffix.lower() in ('.jpg','.png')])
    return frame_paths, fps, (w,h)

# -------------------------
# Embeddings: try ResNet; fallback to HSV hist
# -------------------------
def load_resnet(device):
    import torch
    from torchvision import models, transforms
    model = models.resnet18(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    preprocess = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return model, torch, preprocess

def get_resnet_embeddings(paths, model, torch_mod, preprocess, device, batch=16):
    N = len(paths)
    embs = np.zeros((N,512), dtype=np.float32)
    with torch_mod.no_grad():
        for b in tqdm(range(0, N, batch), desc="ResNet batches"):
            batch_paths = paths[b:b+batch]
            tensors = []
            for p in batch_paths:
                im = cv2.imread(p)[:,:,::-1]
                tensors.append(preprocess(im))
            tensor = torch_mod.stack(tensors).to(device)
            out = model(tensor).squeeze(-1).squeeze(-1).cpu().numpy()
            norms = np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
            out = out / norms
            embs[b:b+len(batch_paths)] = out
    return embs

def hsv_embeddings(paths):
    embs = []
    for p in tqdm(paths, desc="HSV embeddings"):
        im = cv2.imread(p)
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv],[0,1,2],None,[16,8,2],[0,180,0,256,0,256])
        hist = cv2.normalize(hist, hist).flatten()
        embs.append(hist)
    embs = np.vstack(embs)
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True)+1e-9)
    return embs

# -------------------------
# k-NN candidate graph (cosine similarity)
# -------------------------
def build_k_neighbors(embs, k=12):
    from sklearn.metrics.pairwise import cosine_similarity
    sim = (cosine_similarity(embs) + 1.0) / 2.0
    N = sim.shape[0]
    neighbors = [[] for _ in range(N)]
    for i in range(N):
        idxs = np.argsort(-sim[i])
        picked = []
        for j in idxs:
            if j==i: continue
            picked.append(j)
            if len(picked) >= k:
                break
        neighbors[i] = picked
    return neighbors, sim

# -------------------------
# Compute directed ORB displacement on candidate edges
# -------------------------
def compute_orb_on_candidates(paths, neighbors, max_match=200):
    N = len(paths)
    orb = cv2.ORB_create(nfeatures=1500)
    kps = [None]*N; des = [None]*N
    for i,p in enumerate(tqdm(paths, desc="ORB detect")):
        g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        kpi, desi = orb.detectAndCompute(g, None)
        kps[i] = kpi; des[i] = desi
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    dir_score = {}
    for i in tqdm(range(N), desc="ORB candidate edges"):
        des_i = des[i]; kpi = kps[i]
        if des_i is None:
            for j in neighbors[i]:
                dir_score[(i,j)] = 0.0
            continue
        for j in neighbors[i]:
            des_j = des[j]; kpj = kps[j]
            if des_j is None:
                dir_score[(i,j)] = 0.0
                continue
            matches = bf.match(des_i, des_j)
            matches = sorted(matches, key=lambda m: m.distance)[:max_match]
            if len(matches)==0:
                dir_score[(i,j)] = 0.0
                continue
            dxs = []
            for m in matches:
                pi = kpi[m.queryIdx].pt; pj = kpj[m.trainIdx].pt
                dxs.append(pj[0] - pi[0])
            dir_score[(i,j)] = float(np.mean(dxs))
    # normalize to [-1,1]
    vals = np.array(list(dir_score.values()))
    if vals.size == 0:
        return dir_score
    maxabs = np.max(np.abs(vals)) + 1e-9
    for k in list(dir_score.keys()):
        dir_score[k] = dir_score[k] / maxabs
    return dir_score

# -------------------------
# Compute flow magnitude only for candidate edges
# -------------------------
def compute_flow_on_candidates(paths, neighbors, down=6):
    N = len(paths)
    gray = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY) for p in paths]
    flow_mag = {}
    for i in tqdm(range(N), desc="Flow candidate edges"):
        gi = gray[i]
        for j in neighbors[i]:
            gj = gray[j]
            small_i = cv2.resize(gi, (gi.shape[1]//down, gi.shape[0]//down))
            small_j = cv2.resize(gj, (gj.shape[1]//down, gj.shape[0]//down))
            flow = cv2.calcOpticalFlowFarneback(small_i, small_j, None, 0.5, 2, 12, 2, 5, 1.1, 0)
            mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
            flow_mag[(i,j)] = float(np.mean(mag))
    # normalize [0,1]
    vals = np.array(list(flow_mag.values()))
    if vals.size>0:
        mn, mx = vals.min(), vals.max()
        for k in list(flow_mag.keys()):
            flow_mag[k] = (flow_mag[k] - mn) / (mx - mn + 1e-9)
    return flow_mag

# -------------------------
# Edge costs combine cues
# -------------------------
def build_edge_costs(sim, neighbors, dir_score, flow_mag, w_embed=0.6, w_flow=0.25, w_dir=0.15):
    edges = {}
    for i in range(sim.shape[0]):
        for j in neighbors[i]:
            d_embed = 1.0 - sim[i,j]
            f = flow_mag.get((i,j), 1.0)
            d = dir_score.get((i,j), 0.0)  # in [-1,1]
            dir_pen = 1.0 - (d + 1.0)/2.0
            edges[(i,j)] = w_embed*d_embed + w_flow*f + w_dir*dir_pen
    return edges

# -------------------------
# Ordering: spectral init, beam search on candidate graph, then refinements
# -------------------------
def spectral_init(cost_sparse, N):
    # cost_sparse not used directly; we'll create similarity for spectral using neighbors as affinity
    W = np.zeros((N,N), dtype=float)
    for (i,j),c in cost_sparse.items():
        # convert cost to affinity (higher when cost small)
        W[i,j] = math.exp(-c)
        W[j,i] = math.exp(-c)
    Wsym = (W + W.T)/2.0
    D = np.diag(Wsym.sum(axis=1))
    L = D - Wsym
    try:
        from scipy.sparse.linalg import eigsh
        vals, vecs = eigsh(L, k=2, which='SM')
        fiedler = vecs[:,1]
        order0 = list(np.argsort(fiedler))
    except Exception:
        order0 = list(range(N))
        random.shuffle(order0)
    return order0

def beam_search(N, neighbors, edges, beam_width=30):
    # Beam search over candidate graph. Start from best seeds (nodes with high centrality).
    central = {}
    for i in range(N):
        central[i] = sum(math.exp(-edges.get((i,j),1.0)) for j in neighbors[i])
    seeds = sorted(central.keys(), key=lambda x: -central[x])[:min(10, N)]
    beams = []
    for s in seeds:
        beams.append((0.0, [s], {s}))
    beams = sorted(beams, key=lambda x:x[0])[:beam_width]
    step = 0
    while True:
        new_beams = {}
        expanded = 0
        for cost_so_far, seq, visited in beams:
            last = seq[-1]
            for nxt in neighbors[last]:
                if nxt in visited: continue
                edge_cost = edges.get((last,nxt), None)
                if edge_cost is None: continue
                nc = cost_so_far + edge_cost
                nseq = seq + [nxt]
                nvisited = visited | {nxt}
                # keep best for same prefix length and last node
                key = (nseq[-1], len(nseq))
                if key not in new_beams or nc < new_beams[key][0]:
                    new_beams[key] = (nc, nseq, nvisited)
                expanded += 1
        if expanded == 0: break
        # take top beam_width best partial sequences
        cand_list = sorted(new_beams.values(), key=lambda x:x[0])[:beam_width]
        beams = [(c,s,v) for c,s,v in cand_list]
        step += 1
        if any(len(b[1]) == N for b in beams): break
        if step > N: break
    # pick best full if exists
    full = [b for b in beams if len(b[1])==N]
    if full:
        return min(full, key=lambda x:x[0])[1]
    # otherwise take best partial and greedily finish
    best = min(beams, key=lambda x:x[0])
    seq = best[1]; visited = set(seq)
    while len(seq) < N:
        last = seq[-1]
        cand = None; bestc = 1e9
        for j in neighbors[last]:
            if j in visited: continue
            c = edges.get((last,j), 1e9)
            if c < bestc:
                bestc = c; cand = j
        if cand is None:
            for u in range(N):
                if u not in visited:
                    cand = u; break
        seq.append(cand); visited.add(cand)
    return seq

# -------------------------
# Refinements
# -------------------------
def path_cost_with_edges(order, edges):
    s = 0.0
    for i in range(len(order)-1):
        s += edges.get((order[i], order[i+1]), 1.0)
    return s

def two_opt(order, edges, max_iter=200):
    N = len(order); best = order.copy(); bestc = path_cost_with_edges(best, edges)
    improved = True; it = 0
    while improved and it < max_iter:
        improved = False; it += 1
        for i in range(1, N-2):
            for j in range(i+1, N-1):
                cand = best[:i] + best[i:j+1][::-1] + best[j+1:]
                c = path_cost_with_edges(cand, edges)
                if c + 1e-9 < bestc:
                    best, bestc = cand, c; improved = True
    return best

def simulated_annealing(order, edges, iters=600):
    cur = order.copy(); curc = path_cost_with_edges(cur, edges)
    best = cur.copy(); bestc = curc
    T = 1.0
    for k in range(iters):
        i = random.randint(0, len(cur)-2); j = random.randint(i+1, len(cur)-1)
        cand = cur[:i] + cur[i:j+1][::-1] + cur[j+1:]
        cc = path_cost_with_edges(cand, edges)
        if cc < curc or math.exp(-(cc-curc)/max(T,1e-9)) > random.random():
            cur, curc = cand, cc
            if curc < bestc: best, bestc = cur.copy(), curc
        T *= 0.995
    return best

def flow_neighbor_smoothing(order, frame_imgs, max_passes=3):
    def flow_score(a,b):
        g1 = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY); g2 = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
        small = (max(64, g1.shape[1]//6), max(64, g1.shape[0]//6))
        g1s = cv2.resize(g1, small); g2s = cv2.resize(g2, small)
        flow = cv2.calcOpticalFlowFarneback(g1s, g2s, None, 0.5, 2, 12, 2, 5, 1.1, 0)
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        return float(np.mean(mag))
    N = len(order); it = 0
    while it < max_passes:
        it += 1; improved = False
        for k in range(N-1):
            a, b = order[k], order[k+1]
            before = 0.0
            if k-1 >= 0: before += flow_score(frame_imgs[order[k-1]], frame_imgs[a])
            before += flow_score(frame_imgs[a], frame_imgs[b])
            if k+2 < N: before += flow_score(frame_imgs[b], frame_imgs[order[k+2]])
            # after swap
            after = 0.0
            if k-1 >= 0: after += flow_score(frame_imgs[order[k-1]], frame_imgs[b])
            after += flow_score(frame_imgs[b], frame_imgs[a])
            if k+2 < N: after += flow_score(frame_imgs[a], frame_imgs[order[k+2]])
            if after + 1e-9 < before:
                order[k], order[k+1] = order[k+1], order[k]
                improved = True
        if not improved: break
    return order

# -------------------------
# Orientation detection & rotate tail-to-head
# -------------------------
def choose_orientation_by_flow(frames):
    # compare sum of flow magnitudes for forward vs reversed; pick lower (smoother)
    N = len(frames)
    def total_mag(seq):
        s = 0.0
        for i in range(len(seq)-1):
            g1 = cv2.cvtColor(seq[i], cv2.COLOR_BGR2GRAY); g2 = cv2.cvtColor(seq[i+1], cv2.COLOR_BGR2GRAY)
            small = (max(64, g1.shape[1]//6), max(64, g1.shape[0]//6))
            g1s = cv2.resize(g1, small); g2s = cv2.resize(g2, small)
            flow = cv2.calcOpticalFlowFarneback(g1s, g2s, None, 0.5, 2, 12, 2, 5, 1.1, 0)
            mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
            s += float(np.mean(mag))
        return s
    fwd = total_mag(frames)
    rev = total_mag(frames[::-1])
    return (rev < fwd), fwd, rev

# -------------------------
# Main
# -------------------------
def main(args):
    random.seed(123)
    t0 = time.time()
    paths, fps, size = extract_frames(args.video, frames_dir=args.frames_dir)
    N = len(paths)
    print(f"Frames: {N}, fps: {fps}, size: {size}")

    # embeddings
    use_resnet = False
    embeddings = None
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, torch_mod, preprocess = load_resnet(device)
        embeddings = get_resnet_embeddings(paths, model, torch_mod, preprocess, device, batch=args.batch)
        use_resnet = True
        print("Using ResNet embeddings on device:", device)
    except Exception as e:
        print("ResNet not available or failed:", e)
        print("Falling back to HSV embeddings (less accurate but works).")
        embeddings = hsv_embeddings(paths)

    # build k-NN candidate graph
    neighbors, sim = build_k_neighbors(embeddings, k=args.k)
    print("Built k-NN candidate graph with k=", args.k)

    # compute candidate edge cues
    dir_score = compute_orb_on_candidates(paths, neighbors, max_match=args.orb_max_match)
    flow_mag = compute_flow_on_candidates(paths, neighbors, down=args.flow_down)
    edges = build_edge_costs(sim, neighbors, dir_score, flow_mag, w_embed=args.w_embed, w_flow=args.w_flow, w_dir=args.w_dir)
    print("Built directed edge costs (sparse).")

    # spectral init (optional)
    order0 = spectral_init(edges, N)

    # beam search ordering
    order = beam_search(N, neighbors, edges, beam_width=args.beam)
    print("Beam search order obtained. cost:", path_cost_with_edges(order, edges))

    # quick greedy fallback if beam failed
    if order is None or len(order) != N:
        print("Beam search didn't complete; falling back to greedy sequential.")
        order = []
        visited = set()
        cur = order0[0]
        order.append(cur); visited.add(cur)
        for _ in range(N-1):
            cand = min([(edges.get((cur,j),1e9), j) for j in neighbors[cur] if j not in visited] + [(1e9, None)], key=lambda x:x[0])[1]
            if cand is None:
                # pick any unvisited
                for u in range(N):
                    if u not in visited:
                        cand = u; break
            order.append(cand); visited.add(cand); cur = cand

    # 2-opt and SA refinements
    order = two_opt(order, edges, max_iter=args.twoopt_iters)
    order = simulated_annealing(order, edges, iters=args.sa_iters)

    # direction detection using flow on frames
    frames_cache = [cv2.imread(p) for p in paths]
    is_rev, mag_fwd, mag_rev = choose_orientation_by_flow([frames_cache[i] for i in order])
    print(f"Flow mag forward:{mag_fwd:.3f} reversed:{mag_rev:.3f} => reversed smoother? {is_rev}")
    if is_rev ^ args.force_no_reverse:
        print("Reversing final order (flow indicates reversed).")
        order = order[::-1]

    # rotate last K seconds to front if requested
    if args.rotate_seconds > 0:
        k = int(round(args.rotate_seconds * fps))
        if k > 0 and k < N:
            print(f"Rotating last {args.rotate_seconds}s ({k} frames) to front.")
            order = order[-k:] + order[:-k]

    # flow-based neighbor smoothing
    order = flow_neighbor_smoothing(order, frames_cache, max_passes=args.smooth_passes)

    # write output
    out_path = args.out
    print("Writing reconstructed video to:", out_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (size[0], size[1]))
    for idx in order:
        out.write(frames_cache[idx])
    out.release()

    print("Done. Time elapsed:", int(time.time()-t0), "s")
    print("First 40 frames in order:", order[:40])

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--out', default='reconstructed_optimal.mp4')
    p.add_argument('--frames_dir', default='frames_opt')
    p.add_argument('--k', type=int, default=12, help='k-nearest candidates per frame')
    p.add_argument('--beam', type=int, default=30, help='beam width for beam search')
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--w_embed', type=float, default=0.6)
    p.add_argument('--w_flow', type=float, default=0.25)
    p.add_argument('--w_dir', type=float, default=0.15)
    p.add_argument('--orb_max_match', type=int, default=200)
    p.add_argument('--flow_down', type=int, default=6)
    p.add_argument('--twoopt_iters', type=int, default=200)
    p.add_argument('--sa_iters', type=int, default=600)
    p.add_argument('--smooth_passes', type=int, default=3)
    p.add_argument('--rotate_seconds', type=float, default=0.0, help='move last X seconds to front after ordering')
    p.add_argument('--force_no_reverse', action='store_true', help='do not auto-reverse even if flow suggests reversed')
    args = p.parse_args()
    main(args)
