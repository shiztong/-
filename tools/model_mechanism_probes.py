"""Analytic probes only: not trained models, throughput tests, or business lifts."""

import json
import math


def softmax(values):
    shifted = [math.exp(x - max(values)) for x in values]
    total = sum(shifted)
    return [x / total for x in shifted]


def attention_pool(events, positions=None):
    """A simplified member of the attention-pooling family, not ComiRec weights."""
    positions = [0.0] * len(events) if positions is None else positions
    scores = [math.tanh(e[0] + p) for e, p in zip(events, positions)]
    weights = softmax(scores)
    return [sum(w * e[d] for w, e in zip(weights, events)) for d in range(len(events[0]))]


def mmoe_scalar(logits, experts):
    return sum(w * e for w, e in zip(softmax(logits), experts))


def ple_indices(task_num, experts_per_task):
    expected = [[i * experts_per_task + j for j in range(experts_per_task)] for i in range(task_num)]
    inspected = [[i * task_num + j for j in range(experts_per_task)] for i in range(task_num)]
    return expected, inspected


def run_probes():
    length, interests, dim, hidden, batch = 256, 4, 64, 64, 128
    events = [[1.0, 0.0], [0.0, 1.0]]
    no_pos = attention_pool(events)
    reverse = attention_pool(list(reversed(events)))
    with_pos = attention_pool(events, [0.5, -0.5])
    reversed_pos = attention_pool(list(reversed(events)), [0.5, -0.5])
    logits, experts = [0.2, -0.3], [1.5, -0.5]
    value = mmoe_scalar(logits, experts)
    gate = softmax(logits)
    analytic = [w * (e - value) for w, e in zip(gate, experts)]
    step = 1e-6
    numeric = []
    for j in range(len(logits)):
        plus, minus = logits.copy(), logits.copy()
        plus[j] += step
        minus[j] -= step
        numeric.append((mmoe_scalar(plus, experts) - mmoe_scalar(minus, experts)) / (2 * step))
    expected, inspected = ple_indices(3, 2)
    return {
        'warning': __doc__,
        'fixed_source_parameter_arithmetic': {
            'L': length, 'K': interests, 'd': dim, 'B': batch,
            'mind_shared_projection_only': dim * dim,
            'comirec_dr_projection_only': length * interests * dim * dim,
            'comirec_sa_attention_and_positions_excluding_bias': dim * 4 * hidden + 4 * hidden * interests + length * dim,
            'literal_dr_fp32_broadcast_gib_if_materialized': batch * length * interests * dim * dim * 4 / (1024 ** 3),
        },
        'simplified_attention_order_probe': {
            'without_position': no_pos, 'permuted_without_position': reverse,
            'with_position': with_pos, 'permuted_with_fixed_positions': reversed_pos,
        },
        'mmoe_gate_derivative_of_scalar_output': {
            'analytic': analytic, 'finite_difference': numeric,
            'max_absolute_error': max(abs(a - b) for a, b in zip(analytic, numeric)),
        },
        'inspected_ple_indexing_static_probe': {
            'task_num': 3, 'experts_per_task': 2,
            'expected': expected, 'inspected_expression': inspected,
            'matches': expected == inspected,
        },
        'expectation_vs_transform': {
            'mean_duration_both_examples': 10,
            'sqrt_utility_0_or_20_equal_probability': math.sqrt(20) / 2,
            'sqrt_utility_constant_10': math.sqrt(10),
        },
    }


if __name__ == '__main__':
    print(json.dumps(run_probes(), ensure_ascii=False, indent=2, allow_nan=False))
