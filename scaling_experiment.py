"""Reproduce Figure 3 with separately reported setup and worklist counts."""

import csv
import random
from pathlib import Path

try:
    from .gc_semantics import Atom, Record, State, Warrant
except ImportError:
    from gc_semantics import Atom, Record, State, Warrant


def cost_state(size):
    """Seed = size; 15% evidence; one/two candidate binary justifications."""
    rng = random.Random(size)
    records = [Record(i, Atom(f"p{i}"), frozenset({"a"})) for i in range(size)]
    cut = max(3, int(size * 0.15))
    state = State({"a"}, {f"p{i}" for i in range(size)}, records,
                  standing=set(records), evidence=set(records[:cut]))
    warrant = Warrant("binary", 2, lambda _p, _c: True)
    for j in range(cut, size):
        for _ in range(rng.choice([1, 1, 2])):
            premises = frozenset(records[i] for i in rng.sample(range(j), 2))
            state.edges.add((premises, warrant, records[j]))
    return state


def compile_state(state):
    """Build live indices afresh; count structural scans, not elapsed time.

    The counters enumerate named loop operations, not all Python instructions.
    Every recorded premise is inspected, even if another premise is withdrawn.
    Warrant evaluation cost is outside this structural operation model.
    """
    records = state.records
    index = {x: i for i, x in enumerate(records)}
    standing = [x in state.standing for x in records]
    base = [x in state.evidence for x in records]
    heads, premises, used_by = [], [], [[] for _ in records]
    counts = dict(setup_record_scans=len(records), setup_edge_scans=0,
                  setup_premise_inspections=0, live_premise_index_writes=0)
    for prems, warrant, target in state.edges:
        counts['setup_edge_scans'] += 1
        premise_ids = []
        all_standing = target in state.standing
        for premise in prems:
            counts['setup_premise_inspections'] += 1
            premise_ids.append(index[premise])
            if premise not in state.standing:
                all_standing = False
        if not all_standing or not state.valid(prems, warrant, target):
            continue
        edge_id = len(heads)
        heads.append(index[target])
        premises.append(premise_ids)
        for premise_id in premise_ids:
            used_by[premise_id].append(edge_id)
            counts['live_premise_index_writes'] += 1
    counts['live_edges'] = len(heads)
    return (standing, base, heads, premises, used_by), counts


def grounding_counted(compiled):
    standing, base, heads, premises, used_by = compiled
    remaining = [len(p) for p in premises]
    grounded = [False] * len(standing)
    queue = []
    for i in range(len(standing)):
        if standing[i] and base[i]:
            grounded[i] = True
            queue.append(i)
    visits = 0
    while queue:
        node = queue.pop()
        for edge in used_by[node]:
            visits += 1
            remaining[edge] -= 1
            target = heads[edge]
            if remaining[edge] == 0 and not grounded[target]:
                grounded[target] = True
                queue.append(target)
    return grounded, dict(node_initialisations=len(standing), premise_visits=visits,
                          grounding_worklist_steps=len(standing) + visits)


def invalidation_counted(compiled):
    standing, base, heads, premises, used_by = compiled
    incoming = [0] * len(standing)
    for target in heads:
        incoming[target] += 1
    dead = [False] * len(heads)
    invalid = [False] * len(standing)
    queue = []
    for i in range(len(standing)):
        if not standing[i] or (not base[i] and incoming[i] == 0):
            invalid[i] = True
            queue.append(i)
    seeded = len(queue)
    pops = visits = 0
    while queue:
        node = queue.pop()
        pops += 1
        for edge in used_by[node]:
            visits += 1
            if dead[edge]:
                continue
            dead[edge] = True
            target = heads[edge]
            if invalid[target]:
                continue
            incoming[target] -= 1
            if incoming[target] == 0 and not base[target]:
                invalid[target] = True
                queue.append(target)
    return invalid, dict(incoming_count_initialisations=len(heads),
                         seed_record_scans=len(standing), seeded_records=seeded,
                         queue_removals=pops, propagation_premise_visits=visits,
                         propagation_steps=pops + visits)


def _write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scaling(output, plot=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    grounding_rows = []
    for size in (2000, 4000, 8000, 16000, 32000, 64000):
        state = cost_state(size)
        compiled, setup = compile_state(state)
        grounded, work = grounding_counted(compiled)
        assert all(grounded)
        assert {x for x, flag in zip(state.records, grounded) if flag} == state.grounded_records_worklist()
        grounding_rows.append(dict(records=size, seed=size,
                                   premise_occurrences=setup['setup_premise_inspections'],
                                   **setup, **work))
    revision_rows = []
    rng = random.Random(1)
    for count in (1, 2, 5, 20, 100, 1000, 10000):
        state.standing = set(state.records)
        state.standing.difference_update(rng.sample(state.records, count))
        compiled, setup = compile_state(state)
        invalid, work = invalidation_counted(compiled)
        invalid_records = {x for x, flag in zip(state.records, invalid) if flag}
        assert invalid_records == state.invalid_fixed_point_worklist()
        assert invalid_records == set(state.records) - state.grounded_records_worklist()
        revision_rows.append(dict(records=len(state.records), graph_seed=64000,
                                  withdrawal_seed=1, withdrawn=count,
                                  invalid_records=sum(invalid), **setup, **work))
    _write_csv(output / 'scaling.csv', grounding_rows)
    _write_csv(output / 'revision.csv', revision_rows)
    if plot:
        plot_scaling(grounding_rows, revision_rows, output / 'scaling.pdf')
    for row in revision_rows:
        print(f"Withdrawn {row['withdrawn']:5}: X*={row['invalid_records']:5}; "
              f"queue removals={row['queue_removals']:5}; "
              f"premise visits={row['propagation_premise_visits']:5}; "
              f"propagation total={row['propagation_steps']:5}")
    print(f"Wrote reproducibility data to {output.resolve()}")
    return grounding_rows, revision_rows


def plot_scaling(grounding, revision, output):
    """Write a vector figure with the same TeX/PGFPlots toolchain as the paper."""
    import shutil
    import subprocess
    engine = shutil.which('pdflatex')
    if engine is None:
        raise RuntimeError('--plot requires pdflatex with standalone and pgfplots')
    sizes = [r['records'] + r['premise_occurrences'] for r in grounding]
    def coords(xs, ys):
        return ' '.join(f'({x},{y})' for x, y in zip(xs, ys))
    template = r'''\documentclass[border=2pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\definecolor{groundblue}{HTML}{2647A3}
\definecolor{revisionorange}{HTML}{A74300}
\begin{document}
\begin{tikzpicture}
\begin{groupplot}[group style={group size=2 by 1,horizontal sep=1.35cm},
 width=7.0cm,height=5.5cm,xmode=log,ymode=log,
 tick label style={font=\small},label style={font=\small},
 title style={font=\small},grid=major,grid style={gray!25},
 legend style={font=\scriptsize,draw=none,fill=none,at={(0.5,-0.3)},anchor=north},
 legend cell align=left]
\nextgroupplot[title={Grounding after live-edge setup},
 xlabel={State size $n+m$},ylabel={Counted operations}]
\addplot[groundblue,thick,mark=*] coordinates {@GROUND@};
\addlegendentry{grounding worklist}
\addplot[black!65,densely dotted,thick] coordinates {@SIZE@};
\addlegendentry{$n+m$}
\nextgroupplot[title={Revision propagation only},
 xlabel={Ungrounded records $|X^*|$},ymax=400000]
\addplot[revisionorange,thick,mark=square*] coordinates {@REVISION@};
\addlegendentry{queue removals + visits}
\addplot[black!65,densely dotted,thick] coordinates {@INVALID@};
\addlegendentry{$|X^*|$}
\addplot[red!60!black,dashed] coordinates {@BASELINE@};
\addlegendentry{$n+m$ (size reference)}
\end{groupplot}
\end{tikzpicture}
\end{document}
'''
    xs = [r['invalid_records'] for r in revision]
    replacements = {
        '@GROUND@': coords(sizes, [r['grounding_worklist_steps'] for r in grounding]),
        '@SIZE@': coords(sizes, sizes),
        '@REVISION@': coords(xs, [r['propagation_steps'] for r in revision]),
        '@INVALID@': coords(xs, xs),
        '@BASELINE@': coords([min(xs), max(xs)], [sizes[-1], sizes[-1]]),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    output = Path(output).resolve()
    source = output.with_suffix('.tex')
    source.write_text(template, encoding='utf-8')
    result = subprocess.run([engine, '-interaction=nonstopmode', '-halt-on-error',
                             source.name], cwd=source.parent, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Figure compilation failed:\n' + result.stdout[-4000:])
