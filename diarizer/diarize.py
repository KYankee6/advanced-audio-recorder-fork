#!/usr/bin/env python3
"""
diarize.py — 로컬 화자분리 (100% 오프라인, 게이트 없는 모델만 사용)
- 아키텍처: pyannote segmentation-3.0 (회의 학습, 원거리 강건) 로
  10초 윈도우별 화자활성 임베딩(프레임x7)을 만들고 병합 클러스터링으로
  화자를 나눈 뒤, whisper 세그먼트에 프레임 다수결으로 화자를 붙인다.
- 입력: 16kHz mono WAV + whisper 세그먼트 JSON ([{start, end, text}, ...])
- 출력(stdout): {"speakers": N, "assignments": [{"index": i, "speaker": "발화자1"}]}

사용법:
  python diarize.py --wav rec.wav --segments segs.json --max-speakers 3
  --max-speakers 0 이면 자동 추정 (고유값 갭). 회의 인원을 알면 고정 권장.
"""
import argparse
import json
import sys

import numpy as np
import torch

MIRROR = "ubitec/pyannote-segmentation-3.0"  # 비게이트 pyannote 세그멘테이션 미러
WIN_SEC = 10.0
HOP_SEC = 5.0  # 중첩 홉: 경계 정밀도


def load_wav(path: str, target_sr: int = 16000):
    import wave

    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != target_sr:
        from scipy.signal import resample_poly
        x = resample_poly(x, target_sr, sr)
    return x


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if x.size else 0.0


def segmentation_activations(model, wav: np.ndarray, sr: int):
    """10초 윈도우를 HOP_SEC 간격으로 이동하며 (윈도우수, 프레임수, 7) 활성행렬."""
    win = int(WIN_SEC * sr)
    hop = int(HOP_SEC * sr)
    chunks = []
    n = len(wav)
    for s in range(0, max(n - hop, 1), hop):
        clip = wav[s:s + win]
        if len(clip) < win:
            clip = np.pad(clip, (0, win - len(clip)))
        with torch.no_grad():
            out = model(torch.tensor(clip, dtype=torch.float32).unsqueeze(0))
        logits = out["logits"] if isinstance(out, dict) else out
        if isinstance(logits, torch.Tensor):
            logits = logits.squeeze(0)
        else:
            logits = logits[0] if isinstance(logits, (list, tuple)) else logits
        chunks.append(torch.sigmoid(logits if torch.is_tensor(logits)
                                    else torch.tensor(logits)).cpu().numpy())
    return np.stack(chunks)  # (W, F, 7)


def cluster_windows(emb: np.ndarray, max_speakers: int, threshold: float):
    """윈도우 임베딩(F*7) 클러스터링. 발화 있는 윈도우만 대상."""
    from sklearn.cluster import AgglomerativeClustering

    activeness = (emb * emb).sum(axis=1)
    cut = np.percentile(activeness, 10)
    idx = np.where(activeness >= cut)[0]
    sub = emb[idx]
    sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-9)

    labels = np.full(len(emb), -1, dtype=int)  # -1 = 무음/제외
    if max_speakers > 1:
        labels_sub = fixed_k_cluster(sub, min(max_speakers, len(idx)))
    else:
        labels_sub = auto_cluster(sub)
    labels[idx] = labels_sub
    return labels


def fixed_k_cluster(sub: np.ndarray, k: int, thr: float = 0.52):
    """임계값 클러스터링 후 발화량 상위 k개만 남기고, 나머지 창은
    살아남은 클러스터 중심과의 코사인으로 재배정한다.
    (n_clusters=k 직접 컷은 중첩 윈도우 체이닝 때문에 한 덩어리 붕괴 발생)"""
    from sklearn.cluster import AgglomerativeClustering

    lab = AgglomerativeClustering(
        n_clusters=None, distance_threshold=thr,
        metric="cosine", linkage="average").fit_predict(sub)
    sizes = np.bincount(lab)
    keep = sorted(range(len(sizes)), key=lambda i: -sizes[i])[:k]
    if len(keep) < len(sizes):
        cent = np.stack([sub[lab == c].mean(axis=0) for c in keep])
        cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
        remap = {c: j for j, c in enumerate(keep)}
        out = []
        for row, c in zip(sub, lab):
            if c in remap:
                out.append(remap[c])
            else:
                out.append(int(np.argmax(cent @ row)))
        return np.array(out, dtype=int)
    remap = {c: j for j, c in enumerate(keep)}
    return np.array([remap[c] for c in lab], dtype=int)


def auto_cluster(sub: np.ndarray, min_frac: float = 0.05):
    """임계값 안정구간 + 최소크기 필터로 자동 클러스터링.
    잡음성 미니 클러스터(전체의 min_frac 미만)는 -1(제외) 처리한다."""
    from sklearn.cluster import AgglomerativeClustering

    best = None
    for d in np.arange(0.50, 0.95, 0.05):
        try:
            lab = AgglomerativeClustering(
                n_clusters=None, distance_threshold=float(d),
                metric="cosine", linkage="average").fit_predict(sub)
        except Exception:
            continue
        sizes = np.bincount(lab)
        big = int((sizes >= max(2, min_frac * len(sub))).sum())
        print(f"  거리 {d:.2f} → 클러스터 {len(set(lab))}개 중 유효 {big}개", file=sys.stderr)
        if best is None or big > best[0]:
            best = (big, lab.copy())
    if best is None:
        return np.zeros(len(sub), dtype=int)
    lab = best[1]
    sizes = np.bincount(lab)
    keep = sizes >= max(2, min_frac * len(sub))
    # 유효 클러스터만 발화량 순으로 재라벨, 나머지는 -1
    order_ids = [i for i in range(len(sizes)) if keep[i]]
    order_ids.sort(key=lambda i: -sizes[i])
    remap = {old: new for new, old in enumerate(order_ids)}
    return np.array([remap.get(l, -1) for l in lab], dtype=int)


def estimate_speaker_count(emb: np.ndarray, max_k: int = 8) -> int:
    aff = (emb @ emb.T + 1.0) / 2.0
    np.fill_diagonal(aff, 1.0)
    deg = aff.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-9))
    lap = np.diag(np.maximum(deg, 1e-9)) - aff
    eigvals = np.clip(
        np.linalg.eigvalsh(lap * np.outer(d_inv_sqrt, d_inv_sqrt)), 0, None)
    gaps = np.diff(eigvals[: max_k + 1])
    if len(gaps) == 0 or gaps.max() < 0.05:
        return 1
    return int(np.argmax(gaps[:max_k]) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--threshold", type=float, default=0.0, help="호환용(미사용)")
    ap.add_argument("--max-speakers", type=int, default=0)
    ap.add_argument("--min-duration", type=float, default=1.0, help="호환용(미사용)")
    ap.add_argument("--energy-floor", type=float, default=0.004, help="호환용(미사용)")
    ap.add_argument("--cache", default="")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    with open(args.segments) as f:
        segments = json.load(f)
    if isinstance(segments, dict):
        segments = segments.get("segments", [])
    if not segments:
        json.dump({"speakers": 0, "assignments": []}, sys.stdout)
        return

    import os
    acts = None
    if args.cache and os.path.exists(args.cache):
        acts = np.load(args.cache)
        print(f"캐시 로드: {args.cache}", file=sys.stderr)

    wav = load_wav(args.wav)
    sr = 16000
    if acts is None:
        from pyannote.audio import Model
        model = Model.from_pretrained(MIRROR)
        # PyanNet의 instance_norm이 MPS에서 미구현 버그 → CPU 고정
        device = "cpu"
        model = model.to(device).eval()
        print(f"세그멘테이션 추론: {len(wav)/sr:.0f}초, device={device}", file=sys.stderr)
        acts = segmentation_activations(model, wav, sr)
        if args.cache:
            np.save(args.cache, acts)

    n_win, n_frame, _ = acts.shape
    dur = len(wav) / sr
    print(f"윈도우 {n_win}개 (10초창, {HOP_SEC}초 홉) × 프레임 {n_frame} × 7화자슬롯",
          file=sys.stderr)

    # 1) 윈도우 임베딩 클러스터링
    emb = acts.reshape(n_win, -1)  # (W, F*7)
    labels = cluster_windows(emb, args.max_speakers, args.threshold)

    # 발화량 순 라벨링
    order = {}
    for lab in sorted(set(labels.tolist()) - {-1},
                      key=lambda l: -(labels == l).sum()):
        order[lab] = f"발화자{len(order) + 1}"

    if args.debug:
        from collections import Counter
        print(f"[debug] 윈도우 클러스터: {Counter(labels.tolist())}", file=sys.stderr)

    # 2) 세그먼트 배정: 커버 윈도우들과의 겹침 길이 가중 다수결
    win_starts = np.arange(n_win) * HOP_SEC
    assign = []
    for i, seg in enumerate(segments):
        s, e = float(seg["start"]), min(float(seg["end"]), dur)
        overlap = np.minimum(win_starts + WIN_SEC, e) - np.maximum(win_starts, s)
        votes = {}
        for w, ov in enumerate(overlap):
            if ov > 0 and labels[w] in order:
                votes[labels[w]] = votes.get(labels[w], 0.0) + float(ov)
        if not votes:
            # 커버 윈도우 없음 → 시간상 가장 가까운 윈도우
            w = int(np.clip(np.argmin(np.abs(win_starts - s)), 0, n_win - 1))
            assign.append({"index": i, "speaker": order.get(labels[w], "-")})
            continue
        lab = max(votes, key=votes.get)
        assign.append({"index": i, "speaker": order.get(lab, "-")})

    json.dump({"speakers": len(order), "assignments": assign},
              sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
