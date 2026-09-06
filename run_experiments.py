#!/usr/bin/env python3
"""Reproduce the examples and finite consistency checks from the paper."""

import argparse
import random
from pathlib import Path

from gc_semantics import (
    Affected, And, Apply, Atom, CLAIM, Contribute, EVIDENCE, Introduce,
    Justify, Not, Query, Record, Revise, State, Warrant, access_warrant,
    conjunction_warrant, grounded_continuation, TypingError,
)


def by_expr(expression):
    return lambda state: state.find(expression)


def running_examples():
    f_p, h_p, d_p = Atom("f_P"), Atom("h_P"), And(Atom("f_P"), Atom("h_P"))
    state = State({"operator", "assistant"}, {"f_P", "h_P"})
    Apply(Contribute("operator", f_p, EVIDENCE), state)
    Apply(Query("assistant", h_p), state)
    Apply(Contribute("operator", h_p, EVIDENCE), state)
    Apply(Justify("assistant", d_p, (by_expr(f_p), by_expr(h_p)),
                  conjunction_warrant()), state)
    assert set(state.records) == state.grounded_records()

    before = state.grounded_records()
    Apply([
        Introduce("f_T"),
        Revise("operator", by_expr(f_p)),
        Contribute("operator", Atom("f_T"), EVIDENCE),
        Contribute("operator", Not(f_p), EVIDENCE),
    ], state)
    expected = {state.find(h_p), state.find(Atom("f_T")), state.find(Not(f_p))}
    assert state.grounded_records() == expected
    assert state.invalid_fixed_point() == set(state.records) - expected
    assert Affected(before, state) == {state.find(d_p)}

    # Grounding is record-level: another grounded record with the same
    # expression does not rescue an unsupported claim.
    is_grounded, _ = grounded_continuation(
        state, "assistant", Contribute("assistant", h_p, CLAIM)
    )
    assert not is_grounded

    p, q, r, s, z = map(Atom, "pqrsz")
    state = State({"a", "b", "c", "v", "reg"}, set("pqrsz"))
    for actor, expression in (("a", p), ("b", q), ("c", r)):
        Apply(Contribute(actor, expression, EVIDENCE), state)
    bad = Warrant("w_three", 3, lambda premises, conclusion: False)
    Apply(Justify("v", z, (by_expr(p), by_expr(q), by_expr(r)), bad), state)
    assert state.find(z) not in state.grounded_records()
    Apply(Introduce("auditor", is_agent=True), state)
    Apply(Contribute("reg", s, EVIDENCE), state)
    good = access_warrant((p, q, r, s), "w_access")
    Apply(Justify("auditor", z, (by_expr(p), by_expr(q), by_expr(r), by_expr(s)),
                  good, target=by_expr(z)), state)
    assert state.find(z) in state.grounded_records()

    # Evidence is a grounding base even when an incoming justification is stored.
    t = Atom("t")
    Apply(Introduce("t"), state)
    Apply(Contribute("a", t, EVIDENCE), state)
    identity = Warrant("w_id", 1, lambda premises, conclusion: premises[0] == conclusion)
    snapshot = (list(state.records), set(state.edges), set(state.standing))
    try:
        Apply(Justify("a", t, (by_expr(t),), identity, target=by_expr(t)), state)
        raise AssertionError("a cyclic justification was accepted")
    except ValueError as error:
        assert "cycle" in str(error)
    assert snapshot == (state.records, state.edges, state.standing)


def edge_case_checks():
    # Repeated occurrences retain independent identities and standing.
    p, q, r = map(Atom, "pqr")
    state = State({"a", "b"}, set("pqr"))
    first = Apply(Contribute("a", p, EVIDENCE), state)[0]
    second = Apply(Contribute("a", p, EVIDENCE), state)[0]
    assert first != second and first.uid != second.uid
    Apply(Revise("a", first), state)
    assert first not in state.grounded_records() and second in state.grounded_records()

    # An actor remains accountable when attributing a claim to another source.
    grounded, _ = grounded_continuation(state, "a", Contribute("a", q, CLAIM, {"b"}))
    assert not grounded
    grounded, _ = grounded_continuation(state, "a", Contribute("b", q, CLAIM, {"a"}))
    assert grounded  # The event advances no record for actor a.

    # A failed component rolls back all prior changes, including the signature.
    snapshot = (set(state.agents), set(state.propositions), list(state.records),
                set(state.edges), set(state.standing), set(state.evidence))
    try:
        Apply([Introduce("fresh"), Contribute("a", Atom("fresh"), EVIDENCE),
               Revise("b", second)], state)
    except TypingError:
        pass
    else:
        raise AssertionError("ill-typed compound event was accepted")
    assert snapshot == (state.agents, state.propositions, state.records,
                        state.edges, state.standing, state.evidence)

    # Withdrawal removes optional support, never the independent evidence base.
    state = State({"a"}, set("pqr"))
    source = Apply(Contribute("a", p, EVIDENCE), state)[0]
    claim = Apply(Contribute("a", q, CLAIM), state)[0]
    evidence = Apply(Contribute("a", r, EVIDENCE), state)[0]
    rule = Warrant("test", 1, lambda _p, _c: True)
    Apply(Justify("a", q, (source,), rule, claim), state)
    Apply(Justify("a", r, (claim,), rule, evidence), state)
    before = state.grounded_records()
    Apply(Revise("a", source), state)
    assert state.grounded_records() == {evidence}
    assert state.invalid_fixed_point() == {source, claim}
    assert state.invalid_fixed_point_worklist() == {source, claim}
    assert Affected(before, state) == {claim}
    compare_procedures(state)

    # Independent support survives; losing the last alternative propagates.
    state = State({"a"}, set("pqr"))
    left = Apply(Contribute("a", p, EVIDENCE), state)[0]
    right = Apply(Contribute("a", q, EVIDENCE), state)[0]
    target = Apply(Justify("a", r, (left,), rule), state)[0]
    Apply(Justify("a", r, (right,), rule, target), state)
    Apply(Revise("a", left), state)
    assert target in state.grounded_records()
    compare_procedures(state)
    Apply(Revise("a", right), state)
    assert target not in state.grounded_records()
    compare_procedures(state)

    # Each initial evidence premise must be counted exactly once.
    state = State({"a"}, set("pqr"))
    left = Apply(Contribute("a", p, EVIDENCE), state)[0]
    right = Apply(Contribute("a", q, CLAIM), state)[0]
    binary = Warrant("binary", 2, lambda _p, _c: True)
    target = Apply(Justify("a", r, (left, right), binary), state)[0]
    assert target not in state.grounded_records_worklist()
    compare_procedures(state)


def random_state(rng, size):
    names = {f"p{i}" for i in range(size)}
    state = State({"a"}, names)
    records = [Record(i, Atom(f"p{i}"), frozenset({"a"})) for i in range(size)]
    state.records = records
    state.standing = set(records)
    # Evidence may appear anywhere, including at the head of stored edges.
    state.evidence = {x for x in records if rng.random() < 0.2}
    order = list(records)
    rng.shuffle(order)
    for position, conclusion in enumerate(order[1:], 1):
        for edge_no in range(rng.randint(0, 3)):
            width = rng.randint(1, min(4, position))
            premises = frozenset(rng.sample(order[:position], width))
            licensed = rng.random() >= 0.2
            warrant = Warrant(
                f"w{conclusion.uid}_{edge_no}", width,
                lambda _premises, _conclusion, allowed=licensed: allowed,
            )
            state.edges.add((premises, warrant, conclusion))
    assert state.well_formed()
    return state


def compare_procedures(state):
    from scaling_experiment import compile_state, grounding_counted, invalidation_counted
    grounded = state.grounded_records()
    assert grounded == state.grounded_records_worklist()
    invalid = state.invalid_fixed_point()
    assert invalid == state.invalid_fixed_point_worklist()
    assert invalid == set(state.records) - grounded
    compiled, _ = compile_state(state)
    numeric_grounded, _ = grounding_counted(compiled)
    numeric_invalid, _ = invalidation_counted(compiled)
    assert {x for x, flag in zip(state.records, numeric_grounded) if flag} == grounded
    assert {x for x, flag in zip(state.records, numeric_invalid) if flag} == invalid


def randomized_checks(trials, seed):
    rng = random.Random(seed)
    coverage = dict(evidence_with_edges=0, late_premises=0, invalid_warrants=0)
    for _ in range(trials):
        state = random_state(rng, rng.randint(8, 60))
        # Sampling from record order avoids dependence on Python's hash seed.
        for record in rng.sample(state.records, rng.randint(0, min(5, len(state.records)))):
            state.standing.remove(record)
        for premises, warrant, conclusion in state.edges:
            coverage['evidence_with_edges'] += conclusion in state.evidence
            coverage['late_premises'] += any(x.uid > conclusion.uid for x in premises)
            coverage['invalid_warrants'] += not state.valid(premises, warrant, conclusion)
        compare_procedures(state)
    if trials >= 100:
        assert all(coverage.values()), coverage
    print("Generated-state coverage:", coverage)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--plot", action="store_true", help="write Figure 3 (requires pdflatex and pgfplots)")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args(argv)
    running_examples()
    edge_case_checks()
    randomized_checks(args.trials, args.seed)
    if args.scale or args.plot:
        from scaling_experiment import scaling
        scaling(args.output_dir, plot=args.plot)
    print(f"All examples, regression cases and {args.trials} randomized checks passed.")


if __name__ == "__main__":
    main()
