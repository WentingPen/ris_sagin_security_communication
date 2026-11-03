"""Offline checks of analytic bounds; imports environment only, no training."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from system_model.params import get_system_params
from system_model.system_env import SAGIN_ENV


ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
DELTA = np.pi / 6.0


def source_hashes():
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in ("system_model", "algorithm/ppo", "algorithm/IPPO")
        for p in sorted((ROOT / directory).glob("*.py"))
    }


def main():
    before = source_hashes()
    rng = np.random.default_rng(20261101)
    rows = []
    for n in (64, 96, 112, 128, 256):
        params = get_system_params([
            "--num_ris_elements", str(n), "--num_ris_groups", "16",
            "--ris_group_rows", "1", "--ris_group_cols", str(n // 16),
            "--risk_grid_size", "4",
        ])
        x0, x1, y0, y1 = params.area_bounds
        env = SAGIN_ENV(
            params,
            [[0.5 * (x0 + x1), 0.5 * (y0 + y1), params.satellite_height]],
            [[params.ris_fixed_x, params.ris_fixed_y, params.uav_height]],
            [[100.0, 0.0, params.ground_height]],
        )
        for layout in range(10):
            env.reset(seed=31001 + layout)
            for slot in range(10):
                for strategy in ("MRT_active_coherent", "random_feasible_group_action"):
                    continuous, _ = env.get_phase_only_reference()
                    mode = np.ones(16)
                    residual = np.zeros(16)
                    if strategy.startswith("random"):
                        beam = rng.normal(size=16) + 1j * rng.normal(size=16)
                        beam *= rng.uniform(0.1, 1.0) / np.linalg.norm(beam)
                        continuous = np.r_[beam.real, beam.imag]
                        mode = rng.integers(0, 2, 16)
                        residual = rng.uniform(-DELTA, DELTA, 16)
                    base = env.get_ris_phase_reference(continuous)
                    groups = env.uav_ris.element_to_group
                    phase = base + residual[groups]
                    env._apply_action({
                        "sat_beam_real": continuous[:16],
                        "sat_beam_imag": continuous[16:],
                        "ris_element_mode": mode[groups],
                        "ris_phase_real": np.cos(phase),
                        "ris_phase_imag": np.sin(phase),
                    })
                    metrics = env._evaluate()
                    w = env.satellite.W[:, 0]
                    z = env.h_sat_ris @ w
                    rho = env.uav_ris.amplitude
                    active = env.uav_ris.element_mode
                    g_b = env.bobs[0].ris_channel
                    g_e = np.vstack([env.active_eve.ris_channel,
                                     *[e.ris_channel for e in env.passive_eves]])
                    reflected_e = (g_e * env.uav_ris.ris_coeff) @ env.h_sat_ris
                    direct_e = (metrics["eve_effective_channel"] - reflected_e) @ w
                    d = abs(env.bobs[0].direct_channel @ w)
                    s_b = np.sum(rho * abs(g_b * z))
                    q_b = params.bob_noise_power + metrics["bob_ris_noise_power"][0]
                    q_e = params.eve_noise_power + params.ris_noise_power * np.sum(
                        abs(g_e) ** 2 * active * rho ** 2, axis=1)
                    terms = g_e * (rho * z * np.exp(1j * base))
                    grouped = np.column_stack([
                        terms[:, groups == group].sum(axis=1) for group in range(16)
                    ])
                    group_abs_sum = np.sum(abs(grouped), axis=1)
                    envelope = np.minimum(
                        abs(direct_e) + group_abs_sum,
                        abs(direct_e + grouped.sum(axis=1))
                        + 2.0 * np.sin(DELTA / 2.0) * group_abs_sum,
                    )
                    lower = np.log2(1.0 + (d + np.cos(DELTA) * s_b) ** 2 / q_b)
                    lower -= np.log2(1.0 + np.max(envelope ** 2 / q_e))
                    element_sum = np.sum(abs(g_e * (rho * z)), axis=1)
                    eve_lower = np.maximum(abs(direct_e) - element_sum, 0.0) ** 2 / q_e
                    upper = np.log2(1.0 + (d + s_b) ** 2 / q_b)
                    upper -= np.log2(1.0 + np.max(eve_lower))
                    actual = float(metrics["raw_security_gap"][0])
                    assert lower <= actual + 1e-10, (n, strategy, lower, actual)
                    assert actual <= upper + 1e-10, (n, strategy, actual, upper)
                    assert env.ris_power <= params.ris_max_power + 1e-10
                    rows.append(dict(N=n, layout=layout, slot=slot, strategy=strategy,
                                     lower=float(lower), actual=actual, upper=float(upper)))
                    env.metrics = metrics
                env._refresh_common_channels()
                env._refresh_legitimate_channels()
        env.close()
    after = source_hashes()
    assert before == after, "application source files changed during verification"
    with (OUT / "bound_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for n in (64, 96, 112, 128, 256):
        sample = [r for r in rows if r["N"] == n and r["strategy"] == "MRT_active_coherent"]
        summary[n] = {key: float(np.mean([r[key] for r in sample]))
                      for key in ("lower", "actual", "upper")}
        summary[n].update(
            samples=len(sample),
            empirical_SOP=float(np.mean([r["actual"] < 0.1 for r in sample])),
            empirical_SOP_upper=float(np.mean([r["lower"] < 0.1 for r in sample])),
            empirical_SOP_lower=float(np.mean([r["upper"] < 0.1 for r in sample])),
        )
    report = dict(
        assumptions="single Bob, zero jammer; executed gains; group residuals bounded by 30 degrees",
        sample_design="10 geometries x 10 fading slots; reference and random actions share CSI; not independent slots across one geometry",
        no_training=True, risk_grid="4x4, unused for analytic actions", total_checks=len(rows),
        largest_lower_violation=max(r["lower"] - r["actual"] for r in rows),
        largest_upper_violation=max(r["actual"] - r["upper"] for r in rows),
        source_files_unchanged=before == after, source_sha256=after, reference_summary=summary,
    )
    (OUT / "verification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"total_checks": len(rows), "reference_summary": summary,
                      "source_files_unchanged": before == after}, indent=2))


if __name__ == "__main__":
    main()
