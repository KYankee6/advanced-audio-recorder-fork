#!/usr/bin/env python3
"""
diarize.py v2 - pyannote community-1 파이프라인 화자분리 (100% 로컬 추론)

v1(segments 활성 패턴 클러스터링)을 pyannote 공식 community-1으로 교체:
  segmentation + wespeaker 임베딩 + VBx(베이지안 HMM) 클러스터링.
최초 1회 HF 다운로드(게이트 토큰) 후 로컬 캐시로 오프라인 동작한다.

계약(v1과 동일 - 플러그인 수정 불필요):
  입력: 16kHz mono WAV + whisper 세그먼트 JSON ([{start, end, text}, ...])
  사용법: python diarize.py --wav rec.wav --segments segs.json --max-speakers 4
  출력(stdout): {"speakers": N, "assignments": [{"index": i, "speaker": "발화자N"}]}

--max-speakers 0 이면 파이프라인 자동 추정. 세그먼트 배정은 exclusive 출력의
  턴 중 최대 겹침(없으면 2초 이내 최근접)으로 한다.
"""
import argparse
import json
import os
import sys


def _reexec_with_dyld_path():
    """macOS: torchcodec이 홈브루 FFmpeg 공유라이브러리(libavutil 등)를
    찾으려면 DYLD_FALLBACK_LIBRARY_PATH가 필요하다. 플러그인은 환경변수 없이
    이 스크립트를 실행하므로, 본체 임포트 전에 스스로 재실행한다."""
    if (
        sys.platform == "darwin"
        and not os.environ.get("DYLD_FALLBACK_LIBRARY_PATH")
        and os.path.isdir("/opt/homebrew/lib")
    ):
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = "/opt/homebrew/lib"
        os.execv(sys.executable, [sys.executable, os.path.abspath(__file__)] + sys.argv[1:])


_reexec_with_dyld_path()

REPO_ID = "pyannote/speaker-diarization-community-1"
NEAREST_GAP_SEC = 2.0  # 겹침 없는 세그먼트를 최근접 턴으로 배정하는 상한


def load_pipeline():
    from pyannote.audio import Pipeline
    try:
        return Pipeline.from_pretrained(REPO_ID)
    except Exception:
        # 네트워크 불가 환경: HF 캐시로만 로드
        os.environ["HF_HUB_OFFLINE"] = "1"
        return Pipeline.from_pretrained(REPO_ID)


def run_pipeline(wav_path: str, max_speakers: int, cache_path: str):
    """파이프라인 exclusive 다이어라이제이션. [(start, end, label), ...] 반환."""
    import time

    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            print(f"캐시 로드: {cache_path}", file=sys.stderr)
            return [tuple(t) for t in json.load(f)]

    pipeline = load_pipeline()
    kwargs = {"min_speakers": 1}
    if max_speakers > 0:
        kwargs["max_speakers"] = max_speakers
    t0 = time.time()
    out = pipeline(wav_path, **kwargs)
    # 겹침 보존 출력 사용: exclusive 모드는 짧은 끼어들기를 우세 화자에 흡수한다.
    diarization = getattr(out, "speaker_diarization", out)
    turns = [
        (float(turn.start), float(turn.end), label)
        for turn, _, label in diarization.itertracks(yield_label=True)
    ]
    turns.sort()
    elapsed = time.time() - t0
    dur_cache = f"{len(turns)}턴"
    print(f"community-1 추론 완료: {elapsed:.0f}초 ({dur_cache})", file=sys.stderr)
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(turns, f, ensure_ascii=False)
    return turns


def assign_segments(segments, turns):
    """세그먼트별 최대 겹침 턴 배정 (겹침 없으면 2초 이내 최근접, 아니면 '-')."""
    assign = []
    for i, seg in enumerate(segments):
        s, e = float(seg["start"]), float(seg["end"])
        best, best_ov = None, 0.0
        for ts, te, label in turns:
            ov = min(te, e) - max(ts, s)
            if ov > best_ov:
                best, best_ov = label, ov
        if best is None:
            gap, best = min((min(abs(ts - e), abs(te - s)), label) for ts, te, label in turns)
            if gap > NEAREST_GAP_SEC:
                best = None
        assign.append({"index": i, "speaker": best if best is not None else "-"})
    return assign


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--segments", required=True)
    ap.add_argument("--threshold", type=float, default=0.0, help="호환용(미사용)")
    ap.add_argument("--max-speakers", type=int, default=0)
    ap.add_argument("--min-duration", type=float, default=1.0, help="호환용(미사용)")
    ap.add_argument("--energy-floor", type=float, default=0.004, help="호환용(미사용)")
    ap.add_argument("--cache", default="", help="파이프라인 결과 캐시(디버그용)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    with open(args.segments) as f:
        segments = json.load(f)
    if isinstance(segments, dict):
        segments = segments.get("segments", [])
    if not segments:
        json.dump({"speakers": 0, "assignments": []}, sys.stdout)
        return

    turns = run_pipeline(args.wav, args.max_speakers, args.cache)

    # 발화량 순 라벨링 (v1과 동일: 발화 시간이 긴 사람이 발화자1)
    talk = {}
    for ts, te, label in turns:
        talk[label] = talk.get(label, 0.0) + (te - ts)
    order = {
        label: f"발화자{i + 1}"
        for i, label in enumerate(sorted(talk, key=lambda l: -talk[l]))
    }

    assignments = assign_segments(segments, turns)
    for a in assignments:
        a["speaker"] = order.get(a["speaker"], a["speaker"])

    if args.debug:
        import collections
        counts = collections.Counter(a["speaker"] for a in assignments)
        print(f"[debug] 세그먼트 배정: {dict(counts)}", file=sys.stderr)
        for label, name in sorted(order.items(), key=lambda kv: kv[1]):
            print(f"[debug] {name} = {label} (발화 {talk[label]:.0f}초)", file=sys.stderr)

    json.dump({"speakers": len(order), "assignments": assignments}, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
