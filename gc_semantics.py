"""Reference semantics for *What Still Stands?*

The implementation follows the paper's state D = (Sigma, Arg, E, S, B).
It uses only the Python standard library.
"""

from dataclasses import dataclass, field
from typing import Callable, FrozenSet, Iterable, Optional


def Atom(name): return ("atom", name)
def Not(alpha): return ("not", alpha)
def And(left, right): return ("and", left, right)


def atoms(alpha):
    if alpha[0] == "atom":
        return {alpha[1]}
    if alpha[0] == "not":
        return atoms(alpha[1])
    return atoms(alpha[1]) | atoms(alpha[2])


def show(alpha):
    if alpha[0] == "atom":
        return alpha[1]
    if alpha[0] == "not":
        return "~" + show(alpha[1])
    return f"({show(alpha[1])}&{show(alpha[2])})"


EVIDENCE = "evidence"
CLAIM = "claim"
NEW = object()


@dataclass(frozen=True)
class Warrant:
    name: str
    arity: int
    licenses: Callable = field(compare=False, hash=False, repr=False)

    def __post_init__(self):
        if self.arity < 1:
            raise ValueError("a warrant must have positive arity")

    def __repr__(self):
        return self.name


@dataclass(frozen=True)
class Record:
    uid: int
    expression: object
    sources: FrozenSet[str]

    def __repr__(self):
        return f"x_{show(self.expression)}#{self.uid}"


Edge = tuple[FrozenSet[Record], Warrant, Record]


@dataclass
class State:
    agents: set[str] = field(default_factory=set)
    propositions: set[str] = field(default_factory=set)
    records: list[Record] = field(default_factory=list)
    edges: set[Edge] = field(default_factory=set)
    standing: set[Record] = field(default_factory=set)
    evidence: set[Record] = field(default_factory=set)

    def find(self, expression) -> Optional[Record]:
        return next((x for x in self.records if x.expression == expression), None)

    def valid(self, premises, warrant, conclusion):
        expressions = tuple(x.expression for x in premises)
        return len(premises) == warrant.arity and warrant.licenses(
            expressions, conclusion.expression
        )

    def live_edges(self):
        return {
            (premises, warrant, conclusion)
            for premises, warrant, conclusion in self.edges
            if premises | {conclusion} <= self.standing
            and self.valid(premises, warrant, conclusion)
        }

    def grounded_records(self):
        """Least set generated from standing evidence through live edges."""
        live = self.live_edges()
        grounded = self.evidence & self.standing
        changed = True
        while changed:
            changed = False
            for premises, _, conclusion in live:
                if conclusion not in grounded and premises <= grounded:
                    grounded.add(conclusion)
                    changed = True
        return grounded

    def grounded_records_worklist(self):
        """O(n+m) forward worklist computation of the grounded set."""
        live = list(self.live_edges())
        remaining = [len(premises) for premises, _, _ in live]
        used_by = {x: [] for x in self.records}
        for edge_id, (premises, _, _) in enumerate(live):
            for premise in premises:
                used_by[premise].append(edge_id)
        grounded = set(self.evidence & self.standing)
        queue = list(grounded)
        while queue:
            premise = queue.pop()
            for edge_id in used_by[premise]:
                remaining[edge_id] -= 1
                if remaining[edge_id] == 0:
                    conclusion = live[edge_id][2]
                    if conclusion not in grounded:
                        grounded.add(conclusion)
                        queue.append(conclusion)
        return grounded

    def certificate(self, record):
        """Return one grounding certificate as (nodes, edges), or None."""
        grounded = self.grounded_records()
        if record not in grounded:
            return None
        live = self.live_edges()
        nodes, edges, pending = set(), set(), [record]
        while pending:
            node = pending.pop()
            if node in nodes:
                continue
            nodes.add(node)
            if node in self.evidence:
                continue
            edge = next(
                (e for e in live if e[2] == node and e[0] <= grounded), None
            )
            if edge is None:
                return None
            edges.add(edge)
            pending.extend(edge[0])
        return nodes, edges

    def invalid_fixed_point(self):
        """Least fixed point X* of F_D from Section 5.2."""
        live = self.live_edges()
        invalid = set()
        changed = True
        while changed:
            changed = False
            for record in self.records:
                if record in invalid:
                    continue
                if record not in self.standing:
                    invalid.add(record)
                    changed = True
                    continue
                if record in self.evidence:
                    continue
                incoming = [p for p, _, x in live if x == record]
                if all(premises & invalid for premises in incoming):
                    invalid.add(record)
                    changed = True
        return invalid

    def invalid_fixed_point_worklist(self):
        """Selective O(n+m) worklist computation of X*."""
        live = list(self.live_edges())
        incoming_count = {x: 0 for x in self.records}
        used_by = {x: [] for x in self.records}
        dead = [False] * len(live)
        for edge_id, (premises, _, conclusion) in enumerate(live):
            incoming_count[conclusion] += 1
            for premise in premises:
                used_by[premise].append(edge_id)
        invalid = {
            x for x in self.records
            if x not in self.standing
            or (x not in self.evidence and incoming_count[x] == 0)
        }
        queue = list(invalid)
        while queue:
            premise = queue.pop()
            for edge_id in used_by[premise]:
                if dead[edge_id]:
                    continue
                dead[edge_id] = True
                conclusion = live[edge_id][2]
                if conclusion in invalid:
                    continue
                incoming_count[conclusion] -= 1
                if incoming_count[conclusion] == 0 and conclusion not in self.evidence:
                    invalid.add(conclusion)
                    queue.append(conclusion)
        return invalid

    def well_formed(self):
        record_set = set(self.records)
        ids = [x.uid for x in self.records]
        if len(set(ids)) != len(ids) or any(not isinstance(i, int) or i < 0 for i in ids):
            return False
        if not self.standing <= record_set or not self.evidence <= record_set:
            return False
        for record in self.records:
            if not atoms(record.expression) <= self.propositions:
                return False
            if not record.sources or not record.sources <= self.agents:
                return False
        for premises, warrant, conclusion in self.edges:
            if not premises <= record_set or conclusion not in record_set:
                return False
            if len(premises) != warrant.arity:
                return False
        return self.acyclic()

    def acyclic(self):
        successors = {x: set() for x in self.records}
        indegree = {x: 0 for x in self.records}
        for premises, _, conclusion in self.edges:
            for premise in premises:
                if conclusion not in successors[premise]:
                    successors[premise].add(conclusion)
                    indegree[conclusion] += 1
        queue = [x for x in self.records if indegree[x] == 0]
        seen = 0
        while queue:
            node = queue.pop()
            seen += 1
            for successor in successors[node]:
                indegree[successor] -= 1
                if indegree[successor] == 0:
                    queue.append(successor)
        return seen == len(self.records)


class TypingError(ValueError):
    pass


@dataclass(frozen=True)
class Event:
    family: str
    actor: Optional[str] = None
    expression: object = None
    kind: Optional[str] = None
    sources: Optional[FrozenSet[str]] = None
    premises: tuple = ()
    warrant: Optional[Warrant] = None
    target: object = None
    symbol: Optional[str] = None
    symbol_is_agent: bool = False


def Introduce(symbol, is_agent=False):
    return Event("introduce", symbol=symbol, symbol_is_agent=is_agent)


def Contribute(actor, expression, kind, sources=None):
    source_set = frozenset({actor} if sources is None else sources)
    return Event("contribute", actor, expression, kind, source_set)


def Justify(actor, expression, premises, warrant, target=NEW):
    return Event("justify", actor, expression, premises=tuple(premises),
                 warrant=warrant, target=target)


def Revise(actor, target):
    return Event("revise", actor=actor, target=target)


def Query(actor, expression):
    return Event("query", actor=actor, expression=expression)


def _resolve(value, state):
    return value(state) if callable(value) else value


def _descendants(state, record):
    successors = {x: set() for x in state.records}
    for premises, _, conclusion in state.edges:
        for premise in premises:
            successors[premise].add(conclusion)
    found, pending = set(), [record]
    while pending:
        node = pending.pop()
        for successor in successors[node]:
            if successor not in found:
                found.add(successor)
                pending.append(successor)
    return found


def _apply_atomic(event, state):
    if event.family == "introduce":
        symbols = state.agents if event.symbol_is_agent else state.propositions
        if event.symbol in symbols:
            raise TypingError("symbol already occurs in the signature")
        symbols.add(event.symbol)
        return []
    if event.actor not in state.agents:
        raise TypingError("actor is not in the signature")
    if event.family in {"contribute", "justify", "query"}:
        if not atoms(event.expression) <= state.propositions:
            raise TypingError("event expression is outside the signature")
    if event.family == "query":
        return []
    if event.family == "contribute":
        if event.kind not in {EVIDENCE, CLAIM}:
            raise TypingError("contribution kind must be evidence or claim")
        if not event.sources or not event.sources <= state.agents:
            raise TypingError("attributed sources must be nonempty agents")
        record = Record(_fresh_id(state), event.expression, event.sources)
        state.records.append(record)
        state.standing.add(record)
        if event.kind == EVIDENCE:
            state.evidence.add(record)
        return [record]
    if event.family == "revise":
        target = _resolve(event.target, state)
        if target not in state.standing:
            raise TypingError("revision target must be a standing record")
        if event.actor not in target.sources:
            raise TypingError("only an attributed source may revise a record")
        state.standing.remove(target)
        return []
    if event.family == "justify":
        premises = frozenset(_resolve(x, state) for x in event.premises)
        if not premises <= set(state.records):
            raise TypingError("justification premise is not a record")
        if len(premises) != event.warrant.arity:
            raise TypingError("warrant arity does not match the premise set")
        if event.target is NEW:
            target = Record(_fresh_id(state), event.expression,
                            frozenset({event.actor}))
            state.records.append(target)
            state.standing.add(target)
        else:
            target = _resolve(event.target, state)
            if target not in state.standing:
                raise TypingError("target must be a standing record")
            if target.expression != event.expression:
                raise TypingError("target carries a different expression")
        if target in premises or premises & _descendants(state, target):
            raise TypingError("justification would create a cycle")
        state.edges.add((premises, event.warrant, target))
        return [target]
    raise TypingError(f"unknown event family: {event.family}")


def _fresh_id(state):
    used = {record.uid for record in state.records}
    uid = 0
    while uid in used:
        uid += 1
    return uid


def Apply(event, state):
    """Apply one event or a compound list transactionally."""
    events = event if isinstance(event, list) else [event]
    snapshot = (
        set(state.agents), set(state.propositions), list(state.records),
        set(state.edges), set(state.standing), set(state.evidence),
    )
    advanced = []
    try:
        for component in events:
            advanced.extend(_apply_atomic(component, state))
        if not state.well_formed():
            raise TypingError("event produced a malformed state")
        return advanced
    except (TypingError, ValueError):
        (state.agents, state.propositions, state.records, state.edges,
         state.standing, state.evidence) = snapshot
        raise


def Affected(grounded_before, state_after):
    return {
        x for x in state_after.invalid_fixed_point()
        if x in grounded_before and x in state_after.standing
    }


def grounded_continuation(state, actor, event):
    """Check Definition 4 without mutating the supplied state."""
    clone, twins = clone_state(state)
    events = event if isinstance(event, list) else [event]
    remapped = []
    for component in events:
        premises = tuple(
            twins.get(premise, premise) if isinstance(premise, Record) else premise
            for premise in component.premises
        )
        target = component.target
        if isinstance(target, Record):
            target = twins[target]
        remapped.append(Event(
            component.family, component.actor, component.expression,
            component.kind, component.sources, premises, component.warrant,
            target, component.symbol, component.symbol_is_agent,
        ))
    advanced = []
    for component in remapped:
        produced = Apply(component, clone)
        if component.actor == actor:
            advanced.extend(produced)
    grounded = clone.grounded_records()
    return all(x in grounded for x in advanced), clone


def clone_state(state):
    clone = State(set(state.agents), set(state.propositions))
    twins = {
        x: Record(x.uid, x.expression, x.sources) for x in state.records
    }
    clone.records = [twins[x] for x in state.records]
    clone.edges = {
        (frozenset(twins[y] for y in premises), warrant, twins[conclusion])
        for premises, warrant, conclusion in state.edges
    }
    clone.standing = {twins[x] for x in state.standing}
    clone.evidence = {twins[x] for x in state.evidence}
    return clone, twins


def conjunction_warrant():
    def licenses(premises, conclusion):
        return (
            conclusion[0] == "and"
            and frozenset(premises)
            == frozenset({conclusion[1], conclusion[2]})
        )
    return Warrant("w_and", 2, licenses)


def access_warrant(required, name):
    required = frozenset(required)
    return Warrant(
        name, len(required),
        lambda premises, conclusion: conclusion == Atom("z")
        and frozenset(premises) == required,
    )
