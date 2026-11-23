"""Projected finite-difference AO with greedy binary group-mode updates.

This is an online, full-CSI reference, not an RL policy or a global bound.
All candidate evaluations use a private environment copy and common fading.
"""

import copy
from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class AOConfig:
    max_iterations: int = 3
    gradient_step: float = 0.25
    finite_difference: float = 1e-3
    backtracking_steps: int = 6
    tolerance: float = 1e-7

    def __post_init__(self):
        if self.max_iterations < 1 or self.backtracking_steps < 1:
            raise ValueError("AO iteration counts must be positive")
        for name in ("gradient_step", "finite_difference", "tolerance"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass
class GroupAction:
    beam: np.ndarray
    phase: np.ndarray
    mode: np.ndarray

    def copy(self):
        return GroupAction(self.beam.copy(), self.phase.copy(), self.mode.copy())


@dataclass
class AOResult:
    action: dict
    groups: GroupAction
    initial_objective: float
    objective: float
    predicted_reward: float
    evaluations: int
    solve_seconds: float
    trace: list


def project_beam(beam):
    beam = np.asarray(beam, dtype=float)
    return beam / max(float(np.linalg.norm(beam)), 1.0)


def to_env_action(env, groups):
    """Same beam ball, element base phase, group residuals and modes as MAPPO."""
    m = int(env.params.num_sat_antennas)
    g = int(env.params.num_ris_groups)
    if groups.beam.shape != (2 * m,) or groups.phase.shape != (g,) or groups.mode.shape != (g,):
        raise ValueError("group action dimensions do not match system parameters")
    if not all(np.all(np.isfinite(x)) for x in (groups.beam, groups.phase, groups.mode)):
        raise ValueError("group actions must be finite")
    if not np.all(np.isin(groups.mode, (0, 1))):
        raise ValueError("group modes must be binary")
    beam = project_beam(groups.beam).astype(np.float32)
    mapping = env.uav_ris.element_to_group
    base = env.get_ris_phase_reference(beam)
    phase = base + np.radians(env.params.ris_phase_residual_max_deg) * np.clip(groups.phase, -1, 1)[mapping]
    return {
        "sat_beam_real": beam[:m],
        "sat_beam_imag": beam[m:],
        "ris_element_mode": groups.mode[mapping].astype(np.int8),
        "ris_phase_real": np.cos(phase).astype(np.float32),
        "ris_phase_imag": np.sin(phase).astype(np.float32),
    }


class FrozenSlotEvaluator:
    """Genie access to the full CSI realization of the next executed slot.

    Eve/direct-jammer fading is drawn inside _evaluate in this project. Rewinding
    the private RNG fixes those links without consuming the real environment RNG.
    The existing jammer response and RIS power projection are evaluated afresh.
    """

    def __init__(self, env):
        if not env._ready or env.step_count >= env.horizon:
            raise RuntimeError("AO requires a reset, unfinished environment")
        self.env = copy.deepcopy(env)
        self.rng_state = copy.deepcopy(self.env.channel_rng.bit_generator.state)
        self.previous_metrics = copy.deepcopy(self.env.metrics)
        self.evaluations = 0

    def evaluate(self, groups):
        action = to_env_action(self.env, groups)
        self.env.channel_rng.bit_generator.state = copy.deepcopy(self.rng_state)
        self.env.metrics = copy.deepcopy(self.previous_metrics)
        self.env._apply_action(self.env._validate_action(action))
        self.env.metrics = self.env._evaluate()
        self.env._add_costs()
        self.evaluations += 1
        objective = float(self.env.metrics["system_margin"])
        reward = float(self.env._get_reward())
        if not np.isfinite(objective) or not np.isfinite(reward):
            raise FloatingPointError("AO candidate produced non-finite metrics")
        return objective, reward


class AOSolver:
    def __init__(self, config=None):
        self.config = AOConfig() if config is None else config
        self.warm_start = None

    def reset(self):
        self.warm_start = None

    def _continuous_block(self, evaluator, current, score, block):
        values = getattr(current, block)
        gradient = np.zeros_like(values)
        h = self.config.finite_difference
        for index in range(values.size):
            plus, minus = current.copy(), current.copy()
            getattr(plus, block)[index] += h
            getattr(minus, block)[index] -= h
            gradient[index] = (evaluator.evaluate(plus)[0] - evaluator.evaluate(minus)[0]) / (2 * h)
        norm = float(np.linalg.norm(gradient))
        if norm <= 1e-12:
            return current, score
        direction = gradient / norm
        step = self.config.gradient_step
        for _ in range(self.config.backtracking_steps):
            candidate = current.copy()
            proposed = values + step * direction
            setattr(candidate, block, project_beam(proposed) if block == "beam" else np.clip(proposed, -1, 1))
            candidate_score = evaluator.evaluate(candidate)[0]
            if candidate_score > score + self.config.tolerance:
                return candidate, candidate_score
            step *= 0.5
        return current, score

    def solve(self, env):
        started = time.perf_counter()
        evaluator = FrozenSlotEvaluator(env)
        beam, _ = evaluator.env.get_phase_only_reference()
        g = int(env.params.num_ris_groups)
        initial = [GroupAction(beam.copy(), np.zeros(g), np.full(g, mode, dtype=np.int8)) for mode in (0, 1)]
        if self.warm_start is not None:
            initial.append(self.warm_start.copy())
        scored = [(evaluator.evaluate(candidate)[0], candidate) for candidate in initial]
        score, current = max(scored, key=lambda item: item[0])
        initial_score = score
        trace = []
        for iteration in range(1, self.config.max_iterations + 1):
            before = score
            current, score = self._continuous_block(evaluator, current, score, "beam")
            after_beam = score
            current, score = self._continuous_block(evaluator, current, score, "phase")
            after_phase = score
            for index in range(g):
                candidate = current.copy()
                candidate.mode[index] = 1 - candidate.mode[index]
                candidate_score = evaluator.evaluate(candidate)[0]
                if candidate_score > score + self.config.tolerance:
                    current, score = candidate, candidate_score
            trace.append({
                "iteration": iteration, "objective_before": before,
                "objective_after": score, "improvement": score - before,
                "beam_gain": after_beam - before,
                "phase_gain": after_phase - after_beam,
                "mode_gain": score - after_phase,
                "evaluations": evaluator.evaluations,
            })
            if score - before <= self.config.tolerance:
                break
        score, reward = evaluator.evaluate(current)
        self.warm_start = current.copy()
        return AOResult(
            action=to_env_action(evaluator.env, current), groups=current.copy(),
            initial_objective=initial_score, objective=score, predicted_reward=reward,
            evaluations=evaluator.evaluations,
            solve_seconds=time.perf_counter() - started, trace=trace,
        )
