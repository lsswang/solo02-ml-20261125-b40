"""Required negative/positive calibration teaching cases."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from .calibrate import calibrate_file
from .ir import convolve_room, h0_matrix, room_from_dict
from .linalg import max_abs, rms
from .render import inverse_render
from .verify import verify_track
from .wavio import read_wav, write_wav


def _two_point_result(
    room,
    target,
    enabled=(True, True),
    out_wav: str | None = None,
    room_json: str | None = None,
    speaker_override: tuple[list[float], list[float]] | None = None,
    sample_rate: int = 4000,
) -> Dict[str, Any]:
    zero = [0.0] * len(target)
    if speaker_override is None:
        x1, x2, info = inverse_render([target, zero], room, enabled)
    else:
        x1, x2 = speaker_override
        h0 = h0_matrix(room)
        det = h0[0][0] * h0[1][1] - h0[0][1] * h0[1][0]
        info = {"rank_deficient_h0": abs(det) < 1e-10}
    if out_wav:
        write_wav(out_wav, sample_rate, [x1, x2])
    mics = convolve_room([x1, x2], room, len(target))
    near_err = [mics[0][i] - target[i] for i in range(len(target))]
    far = mics[1]
    result = {
        "rank_deficient_h0": info["rank_deficient_h0"],
        "enabled_speakers": list(enabled),
        "speaker_1_peak": max_abs(x1),
        "speaker_2_peak": max_abs(x2),
        "speakers_within_plus_minus_1": max(max_abs(x1), max_abs(x2)) <= 1.0 + 1e-9,
        "near_rmse": rms(near_err),
        "near_peak_error": max_abs(near_err),
        "far_leakage_rms": rms(far),
        "far_leakage_peak": max_abs(far),
        "meets_near_one_far_zero": (
            rms(near_err) < 1e-6
            and max_abs(far) < 1e-6
            and max(max_abs(x1), max_abs(x2)) <= 1.0 + 1e-9
        ),
    }
    if out_wav and room_json:
        verify_json = os.path.splitext(out_wav)[0] + "_verify.json"
        mics_wav = os.path.splitext(out_wav)[0] + "_mics.wav"
        result["pcm_verification"] = verify_track(out_wav, room_json, verify_json, mics_wav)
    return result


def run_diagnostics(samples_dir: str, outdir: str) -> Dict[str, Any]:
    os.makedirs(outdir, exist_ok=True)
    rec = os.path.join(samples_dir, "recordings")
    clips = os.path.join(samples_dir, "clips")
    sentence_path = os.path.join(clips, "calibration_sentence.wav")
    _sr, channels = read_wav(sentence_path)
    target = channels[0]

    sd_learned_path = os.path.join(outdir, "learned_sum_diff.json")
    fs_learned_path = os.path.join(outdir, "learned_far_sum.json")
    calibrate_file(os.path.join(rec, "cal_sum_diff.json"), sd_learned_path)
    calibrate_file(os.path.join(rec, "cal_far_sum.json"), fs_learned_path)

    with open(sd_learned_path, encoding="utf-8") as f:
        sd_learned = json.load(f)
    with open(fs_learned_path, encoding="utf-8") as f:
        fs_learned = json.load(f)
    sd_room = room_from_dict(sd_learned)
    fs_room = room_from_dict(fs_learned)

    sd_wav = os.path.join(outdir, "sum_diff_calibration_sentence.wav")
    sum_diff = _two_point_result(sd_room, target, (True, True), sd_wav, sd_learned_path, sample_rate=_sr)
    # Explicitly demonstrate x1=x2=target/2 sample-for-sample.
    sr2, played = read_wav(sd_wav)
    n = len(target)
    half_dev = max(max(abs(played[0][i] - target[i] / 2), abs(played[1][i] - target[i] / 2)) for i in range(n))
    sum_diff["max_deviation_from_each_channel_half"] = half_dev
    sum_diff["emits_two_half_level_channels"] = half_dev < 4.0e-5  # about one PCM-16 LSB

    fs_wav = os.path.join(outdir, "far_sum_attempt.wav")
    half_sentence = ([v / 2.0 for v in target], [v / 2.0 for v in target])
    far_sum = _two_point_result(
        fs_room, target, (True, True), fs_wav, fs_learned_path,
        speaker_override=half_sentence, sample_rate=_sr,
    )
    far_sum["isolable"] = False
    far_sum["same_half_sentence_used"] = True
    far_sum["reason"] = "H rows are identical: near=x1+x2 and far=x1+x2; the half/half sentence gives target at both."

    unplug_wav = os.path.join(outdir, "unplugged_speaker2_attempt.wav")
    unplugged = _two_point_result(sd_room, target, (True, False), unplug_wav, sd_learned_path, sample_rate=_sr)
    unplugged["isolable"] = False
    unplugged["reason"] = "With speaker 2 unplugged, near=x1 and far=x1 in the tailless room."

    report = {
        "format": "soundbench-diagnostics-v1",
        "sample_rate": _sr,
        "requirement": "near target = calibration sentence, far target = 0, each speaker in [-1, +1]",
        "sum_diff_original": sum_diff,
        "far_also_sum_recalibration": far_sum,
        "restored_sum_diff_then_speaker_2_unplugged": unplugged,
    }
    out_json = os.path.join(outdir, "diagnostics.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md = os.path.join(outdir, "diagnostics_summary.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Required calibration diagnostics\n\n")
        f.write("| Case | Speaker peaks | Near RMSE | Far leakage RMS | Verdict |\n")
        f.write("|---|---:|---:|---:|---|\n")
        f.write(
            f"| Sum/diff original | {sum_diff['speaker_1_peak']:.3f}, {sum_diff['speaker_2_peak']:.3f} "
            f"| {sum_diff['near_rmse']:.2e} | {sum_diff['far_leakage_rms']:.2e} | "
            f"{'PASS / x1=x2=target/2' if sum_diff['meets_near_one_far_zero'] else 'FAIL'} |\n"
        )
        f.write(
            f"| Far also receives sum | {far_sum['speaker_1_peak']:.3f}, {far_sum['speaker_2_peak']:.3f} "
            f"| {far_sum['near_rmse']:.2e} | {far_sum['far_leakage_rms']:.3f} | impossible to isolate |\n"
        )
        f.write(
            f"| Speaker 2 unplugged | {unplugged['speaker_1_peak']:.3f}, 0 "
            f"| {unplugged['near_rmse']:.3f} | {unplugged['far_leakage_rms']:.3f} | impossible to satisfy both |\n"
        )
    return report
