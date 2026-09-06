# What Still Stands? Reproducibility artifact

This directory contains the single reference implementation for *What Still
Stands? A Compositional Semantics for Grounded Continuation in LLM Conversations*.
The state, typed events, grounding certificates and fixed-point invalidation
follow the accompanying paper. Record identifiers distinguish repeated
contributions, even with identical content and attributed sources.

## Reproduce the checks and Figure 3

From this directory, run:

```sh
python3 run_experiments.py --scale --plot
```

Python 3.9 or later is required. The checks and CSV generation use only the
standard library; plotting additionally requires `pdflatex` with the `standalone` and `pgfplots`
packages (included in standard TeX Live installations). To run without
plotting, omit `--plot`. To run correctness checks only, omit both flags.

The command runs the two scenarios, focused regression cases, and 800 generated
acyclic states with 8--60 records. It compares the declarative definitions,
object worklists and counted integer worklists. The regression cases cover:

- repeated records and independent withdrawal;
- contributing actors that differ from attributed sources;
- rollback of a compound event after earlier components changed the state;
- standing evidence with failed incoming support;
- survival of alternative support and loss of the last alternative;
- initial evidence counters in conjunctive support;
- rejection of a cyclic justification.

Generated states place evidence throughout a shuffled topological order, so
evidence can have incoming edges and premises can be recorded after their
conclusions. Some warrants are invalid; withdrawals are sampled from stable
record order. Coverage counts are printed. `--seed` (default 20260902) and
`--trials` change these correctness checks, not the fixed Figure 3 experiment.

Outputs are written to `results/` (or `--output-dir PATH`):

- `scaling.csv`: grounding counts at 2,000--64,000 records;
- `revision.csv`: separate setup, seeding and propagation counts for seven
  withdrawal batches on the 64,000-record graph;
- `scaling.pdf` and `scaling.tex`: the two-panel Figure 3 and its generated
  vector source when `--plot` is supplied.

For the paper workspace, `python3 gc_experiment.py --scale --plot` from the parent
folder runs the same command. The LaTeX source includes
`github_artifact/results/scaling.pdf` directly. The parent `gc_semantics.py` is an
import entry point for the same maintained implementation, not a second engine.

## Graphs and counting conventions

For each scaling size, the graph seed equals the record count. The first 15%
of records are evidence. Each later record receives one or two candidate
binary hyperedges, with probabilities 2/3 and 1/3; premises are sampled from
earlier records and duplicate edges are stored once. Every warrant licenses,
so all records are initially grounded. Revision uses `random.Random(1)` to
sample successive batches of 1, 2, 5, 20, 100, 1,000 and 10,000 withdrawals,
resetting standing records before each batch.

Counts name specific loop operations, not CPU instructions or elapsed time:

- Setup reports record scans, recorded-edge scans, recorded-premise inspections,
  live-premise index writes and the number of live edges.
- Grounding reports node initialisations and premise visits after setup.
- Revision reports incoming-count initialisations, the full-state seeding scan,
  seeded records, queue removals and propagation premise visits.
- The right-hand plot sums queue removals and propagation premise visits.
  Its `n+m` line is a state-size reference, not a measured runtime baseline.

The implementation rebuilds indices and scans the whole state on every call.
It is O(n+m) overall, under constant-time warrant checks. The experiment
establishes selective propagation on the generated graphs, not sublinear
end-to-end revision or production performance. The counted kernels are checked
against the declarative definitions on the generated correctness states and
against the object worklists on every scaling/revision graph.

## Scope

The external interpreter chooses evidence versus claim under an application
admission policy. The implementation does not establish evidence truth,
interpret natural language, or automatically defeat warrants when contrary
information arrives. Such changes require explicit withdrawal of supporting
records. Warrant predicates receive premise expressions and must not depend
on the enumeration order of a premise set.

Before publishing a new artifact release, add the open-source licence agreed
by the authors. This local revision does not publish or license the package.
