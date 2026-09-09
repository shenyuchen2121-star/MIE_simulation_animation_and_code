"""MIE study: network training, formal interaction, analysis and eight outputs.

File structure:
1. Dependencies, experiment configuration and output utilities.
2. Explicit GRU cell, actor, critic, prediction heads and reward functions.
3. Investor pretraining and simultaneous investor/trustee interaction.
4. Conditional fixed points, A1-reference ridge CCA and equilibrium detection.
5. Stress-level regression using eligible phase observations.
6. Joint loss/reward, equilibrium trajectories and four correlation figures.
7. Two time-varying neural vector-field animations.
8. Data loading, complete experiment execution and command-line entry point.

Requirements: Python 3.10+, numpy, scipy, torch, matplotlib, Pillow.
"""

FILE_STRUCTURE = [
    "1. Dependencies, configuration and output utilities",
    "2. GRU, actor, critic, prediction heads and rewards",
    "3. Pretraining and formal interaction",
    "4. Fixed points, ridge CCA and equilibrium detection",
    "5. Stress-level regression",
    "6. Six static publication figures",
    "7. Investor and trustee vector-field animations",
    "8. Experiment pipeline and command-line interface",
]

import argparse
import copy
import csv
import json
import time
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.linalg import eigh, svd
from scipy.optimize import root
from scipy.stats import linregress
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from PIL import Image, GifImagePlugin


# 1. Experiment configuration and output utilities.


@dataclass(frozen=True)
class Config:
    seed: int = 20260903
    hidden_size: int = 26
    pretrain_games: int = 300
    rounds: int = 150
    multiplier: float = 1.5
    consistency_weight: float = 0.5
    value_weight: float = 0.2
    pretrain_lr: float = 0.013333333333333334
    pretrain_opponent_centers: tuple = (0.25, 0.75)
    pretrain_opponent_halfwidth: float = 0.125
    trustee_actor_orthogonal_init: bool = True
    investor_shared_readout: bool = False
    online_optimizer: str = 'sgd'
    attractor_mixing: float = 0.125
    initial_temperature: float = 0.35
    temperature_floor: float = 0.05
    investor_actor_lr: float = 0.001
    trustee_actor_lr: float = 16.0
    recurrent_lr: float = 0.001
    predictor_lr: float = 1.0
    critic_lr: float = 0.05
    temperature_lr: float = 32.0
    context_lr: float = 0.001
    stress_slope: float = 0.6
    levels: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    replicates: int = 10
    neural_threshold: float = 0.05
    dwell_radius: float = 0.05
    dwell_drift: float = 0.02
    fixed_point_tolerance: float = 1e-07
    cca_ridge: float = 0.1
    primary_replicate: int = 0
    cca_cutoff_a: float = 0.999
    cca_cutoff_b: float = 0.99995
    cognitive_cutoff: float = 0.03
    behavioral_epsilon: float = 0.01

    def as_dict(self):
        return asdict(self)

    def validate(self):
        if self.hidden_size != 26 or self.rounds != 150:
            raise ValueError("The study requires 26 neurons and 150 interaction rounds")
        if self.replicates < 1 or self.pretrain_games < 1:
            raise ValueError("Positive replicate and pretraining counts are required")
        if not 0 <= self.primary_replicate < self.replicates:
            raise ValueError("The primary replicate must be included")
        if tuple(self.levels) != (0., .25, .5, .75, 1.):
            raise ValueError("The study uses five fixed stress levels")
        if not 0 <= self.stress_slope < 1:
            raise ValueError("Stress must retain positive learning rates")


PHASES = ((0, 50, "A1"), (50, 100, "B"), (100, 150, "A2"))
METRICS = ("neural_distance", "neural", "cognitive", "behavioral", "MIE")
PLOT_METRICS = ("behavioral", "cognitive", "MIE", "neural_distance")
LEVELS = torch.linspace(0., 1., 101)
COLORS = ("#00897b", "#c62828")
LIGHT_GREEN = "#94CF9A"
DARK_GREEN = "#17633C"
METRIC_COLORS = {"behavioral": "#17633C", "cognitive": "#EF6C00",
                 "MIE": "#7B1FA2", "neural_distance": "#1565C0"}


def configure_runtime():
    torch.set_num_threads(1)
    plt.rcParams.update({"font.family": "Arial", "font.size": 25,
        "axes.labelsize": 25, "axes.titlesize": 30, "xtick.labelsize": 25,
        "ytick.labelsize": 25, "axes.linewidth": 1.1, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white"})


def log(message):
    print(time.strftime('%H:%M:%S') + ' ' + message, flush=True)


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    return value


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


# 2. Neural architecture and learning objectives.


class AppendixGRUCell(nn.Module):

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias = nn.Parameter(torch.empty(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = self.hidden_size ** (-0.5)
        nn.init.uniform_(self.weight_ih, -bound, bound)
        nn.init.uniform_(self.weight_hh, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        input_terms = nn.functional.linear(features, self.weight_ih, self.bias)
        recurrent_terms = nn.functional.linear(hidden, self.weight_hh)
        input_z, input_r, input_h = input_terms.chunk(3, dim=-1)
        recurrent_z, recurrent_r, _ = recurrent_terms.chunk(3, dim=-1)
        update = torch.sigmoid(input_z + recurrent_z)
        reset = torch.sigmoid(input_r + recurrent_r)
        candidate = torch.tanh(input_h + nn.functional.linear(reset * hidden, self.weight_hh[2 * self.hidden_size:]))
        return (1.0 - update) * hidden + update * candidate

def grid(a):
    return float(np.floor(100 * np.clip(a, 0.0, 1.0) + 0.5) / 100)


def rewards(a, b, cfg):
    """Return both payoffs, including the shared action-consistency reward."""
    c = 1 - abs(a - b)
    return (cfg.multiplier * b - a + cfg.consistency_weight * c, cfg.multiplier * a - b + cfg.consistency_weight * c)


def features(own, other, reward, history, scale):
    """Encode actions, reward, five-round opponent mean and latest change."""
    delta = history[-1] - history[-2] if len(history) > 1 else 0.0
    return torch.tensor([2 * own - 1, 2 * other - 1, np.tanh(reward), 2 * np.mean(history[-5:]) - 1, np.clip(delta / scale, -1, 1)], dtype=torch.float32).unsqueeze(0)


class Agent(nn.Module):
    """A 26-unit GRU with two 101-action heads, two predictors and a critic.

    The investor mixes the recurrent state with a trainable context state.
    Both roles learn their action temperature and all active network modules.
    """

    def __init__(self, cfg, investor=False, contexts=True):
        super().__init__()
        self.recurrent = AppendixGRUCell(5, cfg.hidden_size)
        self.shared_readout = investor and cfg.investor_shared_readout
        n = 2 if contexts and (not self.shared_readout) else 1
        self.actor = nn.ModuleList([nn.Linear(cfg.hidden_size, 101) for _ in range(n)])
        self.predictor = nn.ModuleList([nn.Linear(cfg.hidden_size, 1) for _ in range(n)])
        self.value = nn.Linear(cfg.hidden_size, 1)
        self.raw_temperature = nn.Parameter(torch.tensor(float(np.log(np.expm1(cfg.initial_temperature - cfg.temperature_floor)))))
        self.floor = cfg.temperature_floor
        self.alpha = cfg.attractor_mixing if investor else 0.0
        self.context_states = nn.Parameter(torch.randn(2, cfg.hidden_size) / np.sqrt(cfg.hidden_size)) if investor and self.alpha else None
        if not investor and cfg.trustee_actor_orthogonal_init:
            for head in self.actor:
                nn.init.orthogonal_(head.weight, gain=1 / 100)
                nn.init.zeros_(head.bias)

    def temperature(self):
        return self.floor + nn.functional.softplus(self.raw_temperature)

    def transition(self, x, h, k):
        new = self.recurrent(x, h)
        if self.context_states is not None:
            new = (1 - self.alpha) * new + self.alpha * torch.tanh(self.context_states[k])
        return new

    def forward(self, x, h=None, k=0):
        if h is None:
            h = torch.zeros(x.shape[0], self.recurrent.hidden_size, dtype=x.dtype)
        h = self.transition(x, h, k)
        output_k = 0 if self.shared_readout else k
        logits = self.actor[output_k](h)
        value = self.value(h).squeeze(-1)
        pred = torch.sigmoid(self.predictor[output_k](h)).squeeze(-1)
        return (logits, value, pred, h)

    def probabilities(self, logits):
        return torch.softmax(logits / self.temperature(), -1)


def optimizer_for(agent, cfg, investor, stress):
    g = 1.0 if investor else 1 - cfg.stress_slope * stress
    groups = [{'params': agent.actor.parameters(), 'lr': g * (cfg.investor_actor_lr if investor else cfg.trustee_actor_lr)}, {'params': agent.recurrent.parameters(), 'lr': g * cfg.recurrent_lr}, {'params': agent.predictor.parameters(), 'lr': g * cfg.predictor_lr}, {'params': agent.value.parameters(), 'lr': g * cfg.critic_lr}, {'params': [agent.raw_temperature], 'lr': g * cfg.temperature_lr}]
    if agent.context_states is not None:
        groups.append({'params': [agent.context_states], 'lr': cfg.context_lr})
    return torch.optim.Adam(groups) if cfg.online_optimizer == 'adam' else torch.optim.SGD(groups)


def online_loss(agent, logits, value, pred, opponent, team_reward, cfg):
    """Use exact expected team reward, prediction MSE and critic MSE.

    """
    q = 0.5 * (cfg.multiplier - 1) * (LEVELS + opponent) + cfg.consistency_weight * (1 - torch.abs(LEVELS - opponent))
    probs = agent.probabilities(logits).squeeze(0)
    advantage = (q - value.detach().squeeze()).detach()
    actor = -(probs * advantage).sum() - value.detach().squeeze()
    prediction = (pred - opponent).square().mean()
    critic = (value - team_reward).square().mean()
    return (actor + prediction + cfg.value_weight * critic, (actor, prediction, critic))


# 3. Investor pretraining and formal interaction.


def pretrain(cfg, folder, progress):
    """Train the investor for 300 games against context-conditioned opponents.

    Each game starts with a rounded Normal action. Subsequent investor actions
    are categorical samples; the opponent follows its lagged response rule.
    Backpropagation spans a complete game, followed by one Adam update.
    The action-change feature uses its physical range, so its scale is one.
    """
    torch.manual_seed(cfg.seed)
    agent = Agent(cfg, investor=True)
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.pretrain_lr)
    rng = np.random.default_rng(np.random.SeedSequence((cfg.seed, 33, 1)))
    gen = torch.Generator().manual_seed(cfg.seed + 3301)
    environment_rng = np.random.default_rng(np.random.SeedSequence((cfg.seed, 33, 2)))
    actions = np.empty((cfg.pretrain_games, cfg.rounds, 2))
    reward_data = np.empty_like(actions)
    loss_data = np.empty((cfg.pretrain_games, cfg.rounds, 3))
    contexts = np.empty((cfg.pretrain_games, cfg.rounds), dtype=np.int8)
    opponent_baselines = np.empty((cfg.pretrain_games, 2))
    initial_actions = np.empty((cfg.pretrain_games, 2))
    scales = []
    for e in range(cfg.pretrain_games):
        scale = 1.0
        scales.append(scale)
        baselines = np.asarray(cfg.pretrain_opponent_centers, dtype=float).copy()
        if cfg.pretrain_opponent_halfwidth:
            baselines += environment_rng.uniform(-cfg.pretrain_opponent_halfwidth, cfg.pretrain_opponent_halfwidth, size=2)
        opponent_baselines[e] = baselines
        a, b = (0.5, grid(baselines[0]))
        initial_actions[e] = (a, b)
        ra, rb = rewards(a, b, cfg)
        hist = [b]
        hidden = None
        losses = []
        for t in range(cfg.rounds):
            context = 0 if t < 50 or t >= 100 else 1
            x = features(a, b, ra, hist, scale)
            logits, v, pred, hidden = agent(x, hidden, context)
            probs = agent.probabilities(logits)
            old_a, old_b = (a, b)
            if t == 0:
                a = grid(rng.normal(0.5, 0.25))
            else:
                a = int(torch.multinomial(probs.detach().squeeze(0), 1, generator=gen)) / 100
            b = grid(0.5 * old_b + 0.25 * baselines[context] + 0.25 * old_a)
            ra, rb = rewards(a, b, cfg)
            logp = torch.log_softmax(logits / agent.temperature(), -1)[0, round(100 * a)]
            _, parts = online_loss(agent, logits, v, pred, b, (ra + rb) / 2, cfg)
            actor = -logp if t == 0 else parts[0]
            prediction = parts[1]
            critic = parts[2]
            loss = actor + prediction + cfg.value_weight * critic
            losses.append(loss)
            actions[e, t] = (a, b)
            reward_data[e, t] = (ra, rb)
            loss_data[e, t] = [float(actor.detach()), float(prediction.detach()), float(critic.detach())]
            contexts[e, t] = context
            hist.append(b)
        opt.zero_grad()
        torch.stack(losses).mean().backward()
        nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
        opt.step()
        if (e + 1) % 10 == 0 or e == 0:
            progress('pretrain', game=e + 1, total=cfg.pretrain_games, loss=float(torch.stack(losses).mean().detach()))
    np.savez_compressed(folder / 'pretraining_all_episodes.npz', actions=actions, rewards=reward_data, loss_parts=loss_data, normalization_scales=scales, contexts=contexts, critic_targets=reward_data.mean(axis=-1), opponent_baselines=opponent_baselines, initial_actions=initial_actions)
    scale = 1.0
    torch.save({'investor': agent.state_dict(), 'scale': scale, 'config': cfg.as_dict()}, folder / 'initial_investor.pt')
    progress('pretrain_complete', scale=scale, temperature=float(agent.temperature().detach()))
    return (agent, scale)


def simulate(initial_investor, scale, cfg, replicate, stress, folder, capture_frames=False):
    """Run simultaneous actions and online learning over A1, B and A2.

    Each phase lasts 50 rounds. The investor selects context 0, 1, then 0;
    the trustee selects its head from the preceding investor action. Stress
    scales trustee learning rates by 1 - stress_slope * stress. Snapshots
    contain the parameters used for each recorded forward pass.
    """
    investor = copy.deepcopy(initial_investor)
    torch.manual_seed(cfg.seed + 1 + replicate)
    trustee = Agent(cfg)
    agents = (investor, trustee)
    opts = [optimizer_for(a, cfg, i == 0, stress) for i, a in enumerate(agents)]
    rng = np.random.default_rng(np.random.SeedSequence((cfg.seed, replicate, 51)))
    record = {k: np.empty((150, 2)) for k in ('actions', 'categorical_actions', 'rewards', 'predictions', 'values', 'loss', 'actor_loss', 'prediction_loss', 'value_loss', 'temperature', 'heads')}
    record['hidden'] = np.empty((150, 2, 26))
    record['features'] = np.empty((150, 2, 5))
    record['policy'] = np.empty((150, 2, 101))
    groups = ('recurrent', 'actor', 'predictor', 'value', 'raw_temperature', 'context_states')
    acts = [0.5, 0.5]
    rews = rewards(*acts, cfg)
    hist = [[0.5], [0.5]]
    hidden = [None, None]
    snapshots = []
    frames = []
    for t in range(150):
        heads = [0 if t < 50 or t >= 100 else 1, 0 if acts[0] <= 0.5 else 1]
        xs = [features(acts[i], acts[1 - i], rews[i], hist[1 - i], scale) for i in range(2)]
        outputs = [a(xs[i], hidden[i], heads[i]) for i, a in enumerate(agents)]
        probs = [a.probabilities(outputs[i][0]).detach().numpy()[0] for i, a in enumerate(agents)]
        acts = [int(rng.choice(101, p=p)) / 100 for p in probs]
        rews = rewards(*acts, cfg)
        team = float(np.mean(rews))
        totals = []
        parts = []
        for i, a in enumerate(agents):
            logits, value, pred, h = outputs[i]
            total, terms = online_loss(a, logits, value, pred, acts[1 - i], team, cfg)
            totals.append(total)
            parts.append(terms)
            record['hidden'][t, i] = h.detach().numpy()[0]
            record['features'][t, i] = xs[i].numpy()[0]
            record['predictions'][t, i] = float(pred.detach())
            record['values'][t, i] = float(value.detach())
            record['temperature'][t, i] = float(a.temperature().detach())
            record['policy'][t, i] = probs[i]
            for name, term in zip(('actor_loss', 'prediction_loss', 'value_loss'), terms):
                record[name][t, i] = float(term.detach())
            record['loss'][t, i] = float(total.detach())
        if t in (49, 99, 149) or capture_frames:
            frame = {'round': t + 1, 'agents': [copy.deepcopy(a.state_dict()) for a in agents], 'heads': heads.copy(), 'input': np.asarray([x.numpy()[0] for x in xs]), 'hidden': record['hidden'][t].copy()}
            if t in (49, 99, 149):
                snapshots.append(frame)
            if capture_frames:
                frames.append(frame)
        for o in opts:
            o.zero_grad()
        sum(totals).backward()
        for a in agents:
            for group in groups:
                params = [p for n, p in a.named_parameters() if n.split('.')[0] == group]
                if params:
                    nn.utils.clip_grad_norm_(params, 1.)
        for o in opts:
            o.step()
        record['actions'][t] = record['categorical_actions'][t] = acts
        record['rewards'][t] = rews
        record['heads'][t] = heads
        hidden = [out[3].detach() for out in outputs]
        for i in range(2):
            hist[i].append(acts[i])
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / 'raw_interaction.npz', **record)
    torch.save(snapshots, folder / 'phase_snapshots.pt')
    if capture_frames:
        torch.save(frames, folder / 'all_round_snapshots.pt')
    return record, snapshots, frames


# 4. Conditional fixed points, neural relation and equilibrium rules.


def stable_suffix(data, radius, drift, minimum=10):
    y = np.asarray(data).reshape(len(data), -1)
    for start in range(len(y) - minimum + 1):
        suffix = y[start:]
        center = np.mean(suffix, axis=0)
        distances = np.linalg.norm(suffix - center, axis=1)
        mid = len(suffix) // 2
        change = np.linalg.norm(suffix[:mid].mean(0) - suffix[mid:].mean(0))
        steps = np.linalg.norm(np.diff(suffix, axis=0), axis=1)
        if np.max(distances) <= radius and change <= drift and (np.max(steps) <= 2 * radius):
            return {'valid': True, 'start': start + 1, 'point': center, 'radius': float(np.max(distances)), 'drift': float(change), 'points': len(suffix)}
    return {'valid': False, 'start': None, 'point': None, 'reason': 'no suffix passes radius, drift and step tests'}


def equilibrium(curve, threshold, above=False, peak_multiplier=None):
    """Find an onset with at least 8/10 passing rounds and no four-round relapse."""
    y = np.asarray(curve)
    if not np.all(np.isfinite(y)):
        return None
    good = y > threshold if above else y < threshold
    for t in range(len(y) - 9):
        if not good[t] or (t > 0 and good[t - 1]) or good[t:t + 10].sum() < 8:
            continue
        bad = ~good[t + 1:]
        run = 0
        relapse = False
        for b in bad:
            run = run + 1 if b else 0
            if run >= 4:
                relapse = True
        if peak_multiplier is not None and np.any(y[t + 1:] >= peak_multiplier * threshold):
            relapse = True
        if not relapse:
            return t + 1
    return None


def fixed_points(snapshot, cfg, role):
    agent = Agent(cfg, investor=role == 0).double()
    agent.load_state_dict(snapshot['agents'][role])
    x = torch.tensor(snapshot['input'][role], dtype=torch.float64).unsqueeze(0)
    k = int(snapshot['heads'][role])

    def transition(z):
        return agent.transition(x, z.reshape(1, 26), k).reshape(26)

    def fun(z):
        with torch.no_grad():
            return transition(torch.tensor(z, dtype=torch.float64)).numpy() - z

    def jac(z):
        return torch.autograd.functional.jacobian(transition, torch.tensor(z, dtype=torch.float64)).detach().numpy()
    candidates = []
    starts = [np.zeros(26), snapshot['hidden'][role], np.full(26, 0.5), np.full(26, -0.5)]
    for initial in starts:
        sol = root(fun, initial, jac=lambda z: jac(z) - np.eye(26), method='hybr')
        residual = float(np.linalg.norm(fun(sol.x)))
        if residual > cfg.fixed_point_tolerance or not np.all(np.isfinite(sol.x)):
            continue
        if any((np.linalg.norm(sol.x[c] - p['point'][c]) < 1e-06 for p in candidates for c in [slice(None)])):
            continue
        matrix = jac(sol.x)
        spectral = float(np.max(np.abs(np.linalg.eigvals(matrix))))
        z = sol.x + 0.001 * np.eye(26)[0]
        for _ in range(50):
            z = z + fun(z)
        candidates.append({'point': sol.x, 'residual': residual, 'spectral_radius': spectral, 'stable': spectral < 1, 'perturbation_after_50': float(np.linalg.norm(z - sol.x))})
    stable = [p for p in candidates if p['stable']]
    selected = min(stable, key=lambda p: np.linalg.norm(p['point'] - snapshot['hidden'][role])) if stable else None
    return {'roots': candidates, 'selected': selected, 'round': snapshot['round'], 'head': k, 'input': snapshot['input'][role], 'definition': 'conditional fixed point: endpoint parameters, endpoint observation and active head held fixed for analysis only', 'not_proof_of_same_map_bistability': True}

def standardize_neurons(hidden):
    a = np.asarray(hidden, dtype=np.float64)
    std = a.std(axis=-1, keepdims=True, ddof=0)
    return np.divide(a - a.mean(axis=-1, keepdims=True), std, out=np.full_like(a, np.nan), where=std > 1e-12)


def fit_ridge(x, y, ridge=0.1):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or len(x) < 3 or ridge <= 0:
        raise ValueError('Paired training rows and positive ridge required')
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Undefined standardized activity; do not impute it')
    mx, my = (x.mean(0), y.mean(0))
    xc, yc = (x - mx, y - my)
    if np.linalg.norm(xc) < 1e-06 or np.linalg.norm(yc) < 1e-06:
        raise ValueError('No measurable variation in CCA training rows')
    sxx = xc.T @ xc / (len(x) - 1)
    syy = yc.T @ yc / (len(x) - 1)
    sxy = xc.T @ yc / (len(x) - 1)
    lx = ridge * np.trace(sxx) / x.shape[1]
    ly = ridge * np.trace(syy) / y.shape[1]

    def whitening(cov, penalty):
        values, vectors = eigh(cov + penalty * np.eye(len(cov)))
        return vectors / np.sqrt(values) @ vectors.T
    wx, wy = (whitening(sxx, lx), whitening(syy, ly))
    left, strengths, right = svd(wx @ sxy @ wy, full_matrices=False)
    ax, ay = (wx @ left, wy @ right.T)
    for k in range(ax.shape[1]):
        if np.dot(xc @ ax[:, k], yc @ ay[:, k]) < 0:
            ay[:, k] *= -1
    return {'mean_x': mx, 'mean_y': my, 'weights_x': ax, 'weights_y': ay, 'regularized_strengths': strengths, 'lambda_x': lx, 'lambda_y': ly}

def a1_cca(hidden, ridge):
    """Fit CC1 to A1 and score centered five-round windows by cosine similarity.

    A1 fitting excludes the scoring window; B and A2 use the full A1 fit.
    Each round is standardized across neurons. Undefined scores remain NaN.
    """
    z = standardize_neurons(hidden)
    scores = np.full(150, np.nan)
    weights = np.full((150, 2, 26), np.nan)
    centers = np.full_like(weights, np.nan)
    fit_masks = np.zeros((150, 150), dtype=bool)
    score_masks = np.zeros_like(fit_masks)
    norms = np.full((150, 2), np.nan)
    invalid = []
    a1 = np.arange(50)
    try:
        shared = fit_ridge(z[a1, 0], z[a1, 1], ridge)
        shared_error = None
    except ValueError as exc:
        shared, shared_error = (None, str(exc))
    for t in range(150):
        test = np.arange(max(0, t - 2), min(150, t + 3))
        train = np.setdiff1d(a1, test) if t < 50 else a1
        fit_masks[t, train] = True
        score_masks[t, test] = True
        try:
            fit = fit_ridge(z[train, 0], z[train, 1], ridge) if t < 50 else shared
            if fit is None:
                raise ValueError(shared_error)
        except ValueError as exc:
            invalid.append({'round': t + 1, 'reason': str(exc)})
            continue
        weights[t] = [fit['weights_x'][:, 0], fit['weights_y'][:, 0]]
        centers[t] = [fit['mean_x'], fit['mean_y']]
        u = (z[test, 0] - centers[t, 0]) @ weights[t, 0]
        v = (z[test, 1] - centers[t, 1]) @ weights[t, 1]
        norms[t] = [np.linalg.norm(u), np.linalg.norm(v)]
        if min(norms[t]) <= 1e-06:
            invalid.append({'round': t + 1, 'reason': 'canonical_score_norm_at_most_1e-6'})
            continue
        scores[t] = np.dot(u, v) / np.prod(norms[t])
    return {'scores': scores, 'weights': weights, 'centers': centers, 'fit_mask': fit_masks, 'score_mask': score_masks, 'score_norms': norms, 'invalid_windows': invalid}

def behavior_time(actions, epsilon):
    """Detect stability of the signed investor-minus-trustee action gap.

    """
    units = np.rint(actions * 100).astype(np.int64)
    gap = units[:, 0] - units[:, 1]
    delta_units = np.abs(np.diff(gap))
    good = delta_units <= epsilon * 100
    t = None
    for start in range(len(good) - 9):
        if not good[start] or (start > 0 and good[start - 1]) or good[start:start + 10].sum() < 8:
            continue
        run = 0
        relapse = False
        for bad in ~good[start + 1:]:
            run = run + 1 if bad else 0
            if run >= 4:
                relapse = True
        if np.any(delta_units[start + 1:] >= 1.5 * epsilon * 100):
            relapse = True
        if not relapse:
            t = start + 1
            break
    return (None if t is None else t + 1, gap / 100.0, delta_units / 100.0)


def analyze_case(record, snapshots, cfg, saved_roots=None):
    """Compute the four component onsets and their maximum in each phase."""
    roots = saved_roots if saved_roots is not None else [
        [fixed_points(snapshot, cfg, role) for role in (0, 1)]
        for snapshot in snapshots]
    gaps = []
    for role in (0, 1):
        a, b = roots[0][role]['selected'], roots[1][role]['selected']
        gaps.append(float(np.linalg.norm(np.asarray(a['point']) - np.asarray(b['point'])))
                    if a and b else np.nan)
    mean_gap = float(np.mean(gaps))
    cutoff = min(cfg.neural_threshold, .09 * mean_gap) if np.isfinite(mean_gap) and mean_gap > 0 else cfg.neural_threshold
    relation = a1_cca(record['hidden'], cfg.cca_ridge)
    curves = {'neural_distance': np.full((150, 2), np.nan),
              'neural': relation['scores'],
              'cognitive': np.abs(record['predictions'] - record['actions'][:, ::-1]).mean(1),
              'behavioral': record['actions'].copy()}
    phases = []
    for p, (start, end, label) in enumerate(PHASES):
        role_times = []
        for role in (0, 1):
            selected = roots[p][role]['selected']
            if selected is not None:
                curves['neural_distance'][start:end, role] = np.linalg.norm(
                    record['hidden'][start:end, role] - np.asarray(selected['point']), axis=1)
            t = equilibrium(curves['neural_distance'][start:end, role], cutoff)
            dwell = stable_suffix(record['hidden'][start:end, role], cfg.dwell_radius, cfg.dwell_drift)
            role_times.append(t if dwell['valid'] else None)
        behavior, _, _ = behavior_time(record['actions'][start:end], cfg.behavioral_epsilon)
        times = {
            'neural_distance': max(role_times) if all(t is not None for t in role_times) else None,
            'neural': equilibrium(curves['neural'][start:end],
                                  cfg.cca_cutoff_b if label == 'B' else cfg.cca_cutoff_a, above=True),
            'cognitive': equilibrium(curves['cognitive'][start:end], cfg.cognitive_cutoff),
            'behavioral': behavior,
        }
        times['MIE'] = max(times.values()) if all(t is not None for t in times.values()) else None
        phases.append({'label': label, 'equilibria': times, 'eligible': times['MIE'] is not None})
    return {'new_curves': curves, 'new_phases': phases, 'fixed_points': roots,
            'neural_threshold': cutoff, 'cca': relation}


# 5. Regression across the five stress-level means.
def regression_summary(cases, levels, metric):
    """Use phases with all four components present; display sample SD, ddof=1."""
    means, sd, counts = [], [], []
    for stress in levels:
        values = [phase['equilibria'][metric]
                  for case in cases if case['stress'] == stress
                  for phase in case['analysis']['new_phases'] if phase['eligible']]
        counts.append(len(values))
        means.append(float(np.mean(values)) if values else np.nan)
        sd.append(float(np.std(values, ddof=1)) if len(values) > 1 else np.nan)
    ready = all(n >= 5 for n in counts) and np.isfinite(means).all() and np.ptp(means) > 0
    stats = linregress(levels, means) if ready else None
    return {'means': means, 'sd': sd, 'counts': counts, 'regression_ready': bool(ready),
            'slope': float(stats.slope) if stats else None,
            'intercept': float(stats.intercept) if stats else None,
            'r': float(stats.rvalue) if stats else None,
            'p': float(stats.pvalue) if stats else None,
            'errorbar_definition': 'Sample SD (ddof=1)',
            'unit': 'Five stress-level means; phase SD is descriptive, not cluster inference'}


def save_analysis(cases, cfg, output_dir):
    summaries = {m: regression_summary(cases, cfg.levels, m) for m in PLOT_METRICS}
    write_json(output_dir / 'regression_summary.json', summaries)
    rows = []
    for case in cases:
        for phase in case['analysis']['new_phases']:
            rows.append({'replicate': case['replicate'], 'stress': case['stress'],
                         'phase': phase['label'], **phase['equilibria'], 'eligible': phase['eligible']})
    with (output_dir / 'equilibrium_rounds.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return summaries


# 6. Final publication figures.
def save_figure(fig, name, output_dir):
    for extension in ('png', 'svg', 'pdf', 'eps'):
        fig.savefig(output_dir / (name + '.' + extension), dpi=300,
                    bbox_inches='tight', pad_inches=.16)
    plt.close(fig)
    log('Saved ' + name)


def boundaries(ax, top=False):
    ax.set_xlim(.5, 150.5)
    for start, end, label in PHASES:
        if end < 150:
            ax.axvline(end + .5, color='.72', lw=.8, zorder=0)
        if top:
            ax.text((start + 1 + end) / 2, 1.025, label,
                    transform=ax.get_xaxis_transform(), ha='center', va='bottom', fontsize=25)
    ax.spines['right'].set_visible(False)
    if top:
        ax.spines['top'].set_visible(False)
    ax.tick_params(direction='out', length=5, pad=7)


def joint_plot(case, output_dir):
    """Sum both agents' recorded losses and rewards on separate colored axes."""
    record = case['record']
    fig, ax = plt.subplots(figsize=(28 / 3, 9.36))
    right = ax.twinx()
    rounds = np.arange(1, 151)
    ax.plot(rounds, record['loss'].sum(1), color='#6a1b9a', lw=1.5)
    right.plot(rounds, record['rewards'].sum(1), color='#2e7d32', lw=1.5)
    ax.set_ylabel('Joint loss', labelpad=12)
    right.set_ylabel('Joint reward', labelpad=12)
    ax.set_xlabel('Interaction round', labelpad=12)
    boundaries(ax)
    ax.spines['top'].set_visible(False)
    right.spines['top'].set_visible(False)
    right.tick_params(direction='out', length=5, pad=7)
    ax.set_xticks([1, 25, 50, 75, 100, 125, 150])
    for axis in (ax, right):
        axis.yaxis.set_major_locator(MaxNLocator(5))
        color = axis.lines[0].get_color()
        axis.tick_params(axis='both', which='both', labelsize=31.25)
        axis.tick_params(axis='y', which='both', colors=color)
        axis.yaxis.label.set_color(color)
        for coordinate in (axis.xaxis, axis.yaxis):
            coordinate.label.set_fontsize(37.5)
            coordinate.get_offset_text().set_fontsize(31.25)
        axis.yaxis.get_offset_text().set_color(color)
    fig.subplots_adjust(left=.12, right=.88, top=.88, bottom=.24)
    save_figure(fig, 'joint_loss_and_reward_new_version', output_dir)


def trajectory_plot(case, output_dir):
    """Show neural relation, neural distance, prediction error, betting and MIE."""
    curves = case['analysis']['new_curves']
    phases = case['analysis']['new_phases']
    order = ('neural', 'neural_distance', 'cognitive', 'behavioral')
    labels = ('Neural\nrelation', 'Neural\ndistance', 'Prediction\nerror', 'Betting\nproportion')
    ticks = ([.5, .75, 1.], [0., 1.2, 2.4], [0., .1, .2], [0., .4, .8])
    fig = plt.figure(figsize=(15.75, 32 / 3))
    grid = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, .18], hspace=0)
    distance_axis = fig.add_subplot(grid[1])
    relation_axis = fig.add_subplot(grid[0], sharex=distance_axis)
    axes = [relation_axis, distance_axis] + [
        fig.add_subplot(grid[index], sharex=distance_axis) for index in (2, 3, 4)]
    fig.subplots_adjust(left=.20, right=.98, bottom=.08, top=.96)
    for j, (ax, metric, label) in enumerate(zip(axes[:4], order, labels)):
        arr = np.asarray(curves[metric])
        if arr.ndim == 1:
            arr = arr[:, None]
        for role in range(arr.shape[1]):
            ax.plot(np.arange(1, 151), arr[:, role],
                    color=COLORS[role] if arr.shape[1] == 2 else '#1565c0',
                    lw=2.4 if metric == 'neural_distance' else 2.6, zorder=2)
        for p, (start, end, _) in enumerate(PHASES):
            eq = phases[p]['equilibria'][metric]
            if eq is not None:
                index = start + eq - 1
                marker = ax.scatter(index + 1, np.max(arr[index]), s=128, c='black', zorder=100)
                transform = marker.get_offset_transform()
                marker.remove()
                marker.set_offset_transform(transform)
                marker.set_clip_on(False)
                fig.add_artist(marker)
        if metric == 'neural':
            finite = arr[np.isfinite(arr)]
            low = max(-1.01, min(.45, float(finite.min()) - .03)) if finite.size else -1.01
            ax.set_ylim(low, 1.01)
        elif metric == 'behavioral':
            ax.set_ylim(0, 1)
        else:
            ax.set_ylim(bottom=0)
        ax.set_ylabel(label, labelpad=15)
        ax.yaxis.set_label_coords(-.125, .5)
        ax.yaxis.set_major_locator(FixedLocator(ticks[j]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, pos: f'{value:g}'))
        ax.minorticks_off()
        boundaries(ax, top=j == 0)
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
        ax.tick_params(axis='both', which='major', labelsize=37.5)
    axes[3].get_yticklabels()[-1].set_verticalalignment('top')
    ax = axes[-1]
    boundaries(ax)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel('MIE', rotation=0, va='center')
    ax.yaxis.set_label_coords(-.10, .5)
    for phase, (start, end, _) in zip(phases, PHASES):
        eq = phase['equilibria']['MIE']
        if eq is not None:
            ax.axvspan(start + eq - .5, end + .5, color='#F8D19F', lw=0)
    ax.set_xlabel('Interaction round', labelpad=12)
    ax.set_xticks([1, 25, 50, 75, 100, 125, 150])
    ax.tick_params(axis='both', which='major', labelsize=37.5)
    handles = [Line2D([], [], color=COLORS[0], lw=2.6, label='Investor'),
               Line2D([], [], color=COLORS[1], lw=2.6, label='Trustee'),
               Line2D([], [], color='#1565c0', lw=2.6, label='Neural relation / prediction error'),
               Line2D([], [], color='black', marker='o', linestyle='none',
                      markersize=np.sqrt(128), label='Equilibrium')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, 1.02), ncol=4,
               frameon=False, fontsize=20, handlelength=1.4, handletextpad=.5, columnspacing=1.3)
    save_figure(fig, 'Three level equilibrium trajectories_new_version_swapped_neural_panels', output_dir)


def scatter_plot(metric, summary, cfg, output_dir):
    """Plot all five means and SD; compute the filename from the actual regression."""
    fig = plt.figure(figsize=(6.375101453993056, 7.2))
    ax = fig.add_axes([.201133866759167, .17057065610532407,
                      .7607403783317377, .7188181936005014])
    levels = np.asarray(cfg.levels)
    means, errors = np.asarray(summary['means']), np.asarray(summary['sd'])
    good = np.isfinite(means) & np.isfinite(errors)
    color = METRIC_COLORS[metric]
    ax.errorbar(levels[good], means[good],
        yerr=np.vstack((np.minimum(errors[good], means[good]), errors[good])),
        fmt='o', markersize=7, color=color, ecolor=color, capsize=5, elinewidth=1.4, zorder=3)
    if summary['regression_ready']:
        xx = np.array([min(cfg.levels), max(cfg.levels)])
        ax.plot(xx, summary['intercept'] + summary['slope'] * xx, color=color, lw=1.3)
        suffix = f"_r_{summary['r']:.3f}_p_{summary['p']:.4f}"
    else:
        suffix = '_r_undefined_p_undefined'
    ax.set_xlabel('stress', labelpad=12)
    ax.set_ylabel('Equilibrium rounds', labelpad=12)
    ax.set_title(metric.title().replace('_', ' ') + '\nequilibrium vs stress', pad=18)
    ax.set_xticks(cfg.levels)
    ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1'])
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_ylim(bottom=0)
    ax.set_xlim(-.06, 1.06)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out', length=5, pad=7)
    save_figure(fig, f'stress_level_vs_{metric}_equilibrium_rounds_new_version' + suffix, output_dir)


def field_geometry(case, role):
    """Use one PCA basis fitted to all 150 recorded states for each role."""
    hidden = case['record']['hidden'][:, role]
    origin = hidden.mean(0)
    _, _, vt = np.linalg.svd(hidden - origin, full_matrices=False)
    basis = vt[:2]
    trajectory = (hidden - origin) @ basis.T
    roots = []
    for phase in case['analysis']['fixed_points']:
        selected = phase[role]['selected']
        if selected is None:
            raise ValueError('A stable conditional endpoint fixed point is required for the vector movie')
        roots.append(np.asarray(selected['point']))
    projected = [(point - origin) @ basis.T for point in roots]
    return None, origin, basis, trajectory, roots, projected


# 7. Time-varying vector fields and GIF encoding.


class ExactFrameWriter(PillowWriter):

    def finish(self):
        first = self._frames[0].convert('RGB').convert('P', palette=Image.Palette.ADAPTIVE, colors=256)
        header, _ = GifImagePlugin.getheader(first, info={'loop': 0})
        with open(self.outfile, 'wb') as stream:
            for block in header:
                stream.write(block)
            for j, frame in enumerate(self._frames):
                indexed = first if j == 0 else frame.convert('RGB').convert('P', palette=Image.Palette.ADAPTIVE, colors=256)
                for block in GifImagePlugin.getdata(indexed, duration=round(1000 / self.fps), disposal=2, include_color_table=True):
                    stream.write(block)
            stream.write(b';')


def field_axes(title, grid):
    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    ax.set_title(title, pad=20)
    ax.set_xlabel('PC1', labelpad=10)
    ax.set_ylabel('PC2', labelpad=10)
    ax.set_xlim(grid[:, 0].min(), grid[:, 0].max())
    ax.set_ylim(grid[:, 1].min(), grid[:, 1].max())
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(direction='out', length=5, pad=7)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_aspect('equal', adjustable='box')
    fig.tight_layout(pad=0.8)
    return (fig, ax)


def arrows(ax, grid, vectors):
    return ax.quiver(grid[:, 0], grid[:, 1], vectors[:, 0], vectors[:, 1], angles='xy', scale_units='xy', scale=5.0, width=0.003, color=DARK_GREEN, zorder=1, minlength=0)


def circles(ax, roots):
    arr = np.asarray(roots)
    return ax.scatter(arr[:, 0], arr[:, 1], s=100, marker='o', facecolors='none', edgecolors='#D73027', linewidths=1.5, zorder=5)


def grid_common(z, projected):
    pts = np.vstack([z, projected])
    lo, hi = (pts.min(0), pts.max(0))
    span = np.maximum(hi - lo, 0.1)
    xx, yy = np.meshgrid(np.linspace(lo[0] - 0.2 * span[0], hi[0] + 0.2 * span[0], 8), np.linspace(lo[1] - 0.2 * span[1], hi[1] + 0.2 * span[1], 9))
    return np.column_stack([xx.ravel(), yy.ravel()])


def computed_field(snapshot, cfg, role, grid, origin, basis, anchor):
    model = Agent(cfg, investor=role == 0).double()
    model.load_state_dict(snapshot['agents'][role])
    lifted = anchor + (grid - (anchor - origin) @ basis.T) @ basis
    x = torch.as_tensor(snapshot['input'][role], dtype=torch.float64).unsqueeze(0).expand(len(grid), -1)
    with torch.no_grad():
        after = model.transition(x, torch.as_tensor(lifted), int(snapshot['heads'][role])).numpy()
    return ((after - lifted) @ basis.T, lifted)


def vector_movie(case, cfg, frames, role, output_dir):
    _, origin, basis, z, roots, projected = field_geometry(case, role)
    grid = grid_common(z, projected[:2])
    anchor = (roots[0] + roots[1]) / 2
    vectors = []
    for t, snapshot in enumerate(frames):
        v, lifted = computed_field(snapshot, cfg, role, grid, origin, basis, anchor)
        vectors.append(v)
        if (t + 1) % 50 == 0:
            log(f"{('Investor', 'Trustee')[role]} genuine conditional fields {t + 1}/150")
    vectors = np.asarray(vectors)
    role_name = ('investor', 'trustee')[role]
    np.savez_compressed(output_dir / (role_name + '_time_varying_vector_field_data.npz'), origin=origin, basis=basis, anchor=anchor, grid=grid, lifted_hidden=lifted, vectors_unscaled=vectors, display_vectors=vectors / 5, trajectory=z, roots_26d=roots[:2], roots_pc=projected[:2])
    fig, ax = field_axes(f'{role_name.title()}: A1, round 150', grid)
    quiver = arrows(ax, grid, vectors[0])
    line, = ax.plot([], [], color=LIGHT_GREEN, lw=1.4, zorder=3)
    circles(ax, projected[:2])

    def update(t):
        quiver.set_UVC(vectors[t, :, 0], vectors[t, :, 1])
        line.set_data(z[:t + 1, 0], z[:t + 1, 1])
        label = 'A1' if t < 50 else 'B' if t < 100 else 'A2'
        ax.set_title(f'{role_name.title()}: {label}, round {t + 1}', pad=20)
        return (quiver, line)
    movie = FuncAnimation(fig, update, frames=150, interval=1000 / 12, blit=False)
    movie.save(output_dir / (role_name + '_time_varying_vector_field.gif'), writer=ExactFrameWriter(fps=12), dpi=100, progress_callback=lambda current, total: log(f'{role_name} GIF {current + 1}/{total}') if (current + 1) % 25 == 0 else None)
    plt.close(fig)


# 8. Data loading, experiment execution and command-line interface.
def load_config(data_dir):
    values = json.loads((data_dir / 'config.json').read_text(encoding='utf-8'))
    allowed = {f.name for f in fields(Config)}
    cfg = Config(**{key: value for key, value in values.items() if key in allowed})
    cfg.validate()
    return cfg


def load_record(folder):
    with np.load(folder / 'raw_interaction.npz', allow_pickle=False) as source:
        return {key: source[key].copy() for key in source.files}


def recorded_cases(data_dir, output_dir):
    cfg = load_config(data_dir)
    cases = []
    for replicate in range(cfg.replicates):
        for stress in cfg.levels:
            folder = data_dir / f'replicate_{replicate:02d}' / f'stress_{stress:.2f}'
            record = load_record(folder)
            snapshots = torch.load(folder / 'phase_snapshots.pt', map_location='cpu', weights_only=False)
            analysis_path = folder / 'analysis.json'
            saved_roots = (json.loads(analysis_path.read_text(encoding='utf-8'))['fixed_points']
                           if analysis_path.exists() else None)
            analysis = analyze_case(record, snapshots, cfg, saved_roots=saved_roots)
            case = {'replicate': replicate, 'stress': stress, 'record': record,
                    'snapshots': snapshots, 'analysis': analysis}
            cases.append(case)
            log(f'Analyzed replicate {replicate + 1}/{cfg.replicates}, stress {stress:g}')
    return cfg, cases


def train_cases(output_dir):
    cfg = Config()
    cfg.validate()
    data_dir = output_dir / 'data'
    data_dir.mkdir()
    write_json(data_dir / 'config.json', cfg.as_dict())

    def progress(event, **values):
        log(event + ' ' + ', '.join(f'{key}={value}' for key, value in values.items()))

    investor, scale = pretrain(cfg, data_dir, progress)
    cases = []
    for replicate in range(cfg.replicates):
        for stress in cfg.levels:
            folder = data_dir / f'replicate_{replicate:02d}' / f'stress_{stress:.2f}'
            primary = replicate == cfg.primary_replicate and stress == 0.
            record, snapshots, frames = simulate(investor, scale, cfg, replicate, stress, folder,
                                                 capture_frames=primary)
            analysis = analyze_case(record, snapshots, cfg)
            write_json(folder / 'analysis.json', analysis)
            cases.append({'replicate': replicate, 'stress': stress, 'record': record,
                          'snapshots': snapshots, 'frames': frames, 'analysis': analysis})
            log(f'Completed replicate {replicate + 1}/{cfg.replicates}, stress {stress:g}')
    return cfg, cases, data_dir


def recorded_frames(primary, cfg, data_dir, output_dir):
    """Collect all forward-pass maps from the saved pretrained network."""
    folder = data_dir / f'replicate_{cfg.primary_replicate:02d}' / 'stress_0.00'
    candidates = [folder / 'all_round_snapshots.pt', data_dir.parent / 'all_round_snapshots.pt']
    cache = next((path for path in candidates if path.exists()), None)
    if cache is not None:
        frames = torch.load(cache, map_location='cpu', weights_only=False)
    else:
        checkpoint = torch.load(data_dir / 'initial_investor.pt', map_location='cpu', weights_only=False)
        investor = Agent(cfg, investor=True)
        investor.load_state_dict(checkpoint['investor'])
        record, _, frames = simulate(investor, checkpoint['scale'], cfg,
            primary['replicate'], primary['stress'], output_dir / 'primary_interaction', capture_frames=True)
        if not all(np.array_equal(value, primary['record'][key], equal_nan=True)
                   for key, value in record.items()):
            raise RuntimeError('Recorded interaction differs from the deterministic run in this environment')
    if len(frames) != 150:
        raise ValueError('Expected exactly 150 forward-pass snapshots')
    for t, frame in enumerate(frames):
        if frame['round'] != t + 1 or not all(np.array_equal(frame[key], primary['record'][other][t])
                for key, other in (('hidden', 'hidden'), ('input', 'features'), ('heads', 'heads'))):
            raise ValueError(f'Snapshot does not match recorded round {t + 1}')
    for endpoint in primary['snapshots']:
        frame = frames[endpoint['round'] - 1]
        if not all(torch.equal(value, frame['agents'][role][name])
                   for role in (0, 1) for name, value in endpoint['agents'][role].items()):
            raise ValueError('Endpoint parameters do not match the animation snapshots')
    return frames


def create_output(requested):
    if requested is not None:
        path = requested.resolve()
        path.mkdir(parents=True, exist_ok=False)
        return path
    base = Path(__file__).resolve().parent / 'MIE_results'
    for index in range(10000):
        path = base if index == 0 else base.with_name(f'MIE_results_{index:03d}')
        try:
            path.mkdir(exist_ok=False)
            return path
        except FileExistsError:
            continue
    raise RuntimeError('No unused output directory is available')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=('full', 'recorded'), default='full',
                        help='Run the complete experiment or analyze recorded interactions')
    parser.add_argument('--part', choices=('all', 'static', 'movies', 'analysis'), default='all',
                        help='Select outputs after analysis')
    parser.add_argument('--data-dir', type=Path,
                        help='Recorded run directory containing config.json and replicate folders')
    parser.add_argument('--output', type=Path, help='New output directory; existing directories are refused')
    args = parser.parse_args()
    if args.mode == 'full' and args.data_dir is not None:
        parser.error('--data-dir applies only to --mode recorded')
    configure_runtime()
    if args.mode == 'recorded':
        data_dir = args.data_dir or (Path(__file__).resolve().parent /
                    '\u6570\u636e\u4e0e\u5ba1\u8ba1' / '\u539f\u59cb\u8fd0\u884c')
        data_dir = data_dir.resolve()
        if not (data_dir / 'config.json').is_file():
            parser.error('Recorded data were not found; provide --data-dir or use --mode full')
    output_dir = create_output(args.output)
    log('Output directory: ' + str(output_dir))
    if args.mode == 'full':
        cfg, cases, data_dir = train_cases(output_dir)
    else:
        cfg, cases = recorded_cases(data_dir, output_dir)
    write_json(output_dir / 'config.json', cfg.as_dict())
    summaries = save_analysis(cases, cfg, output_dir)
    primary = next(case for case in cases if case['replicate'] == cfg.primary_replicate and case['stress'] == 0.)
    write_json(output_dir / 'primary_analysis.json', primary['analysis']['new_phases'])
    if args.part in ('all', 'static'):
        joint_plot(primary, output_dir)
        trajectory_plot(primary, output_dir)
        for metric in PLOT_METRICS:
            scatter_plot(metric, summaries[metric], cfg, output_dir)
    if args.part in ('all', 'movies'):
        frames = primary.get('frames') or recorded_frames(primary, cfg, data_dir, output_dir)
        for role in (0, 1):
            vector_movie(primary, cfg, frames, role, output_dir)
    write_json(output_dir / 'run_summary.json', {
        'mode': args.mode, 'part': args.part, 'dyads': len(cases),
        'primary_replicate': cfg.primary_replicate, 'primary_stress': 0.,
        'regressions': {metric: summaries[metric] for metric in PLOT_METRICS},
        'libraries': {'numpy': np.__version__, 'torch': torch.__version__, 'matplotlib': matplotlib.__version__}})
    log('Completed: ' + str(output_dir))


if __name__ == '__main__':
    main()
