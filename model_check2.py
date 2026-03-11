#!/usr/bin/env python3
"""model_check2.py — CTL model checker for finite-state systems.

Verifies Computation Tree Logic properties over Kripke structures.
Supports: EX, EF, EG, EU, AX, AF, AG, AU, and boolean combinators.

One file. Zero deps. Does one thing well.
"""

import sys
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Atom:
    name: str
    def __repr__(self): return self.name

@dataclass(frozen=True)
class Not:
    sub: object
    def __repr__(self): return f"¬{self.sub}"

@dataclass(frozen=True)
class And:
    left: object
    right: object
    def __repr__(self): return f"({self.left} ∧ {self.right})"

@dataclass(frozen=True)
class Or:
    left: object
    right: object
    def __repr__(self): return f"({self.left} ∨ {self.right})"

@dataclass(frozen=True)
class EX:
    sub: object
    def __repr__(self): return f"EX({self.sub})"

@dataclass(frozen=True)
class EF:
    sub: object
    def __repr__(self): return f"EF({self.sub})"

@dataclass(frozen=True)
class EG:
    sub: object
    def __repr__(self): return f"EG({self.sub})"

@dataclass(frozen=True)
class EU:
    left: object
    right: object
    def __repr__(self): return f"E[{self.left} U {self.right}]"

@dataclass(frozen=True)
class AX:
    sub: object
    def __repr__(self): return f"AX({self.sub})"

@dataclass(frozen=True)
class AF:
    sub: object
    def __repr__(self): return f"AF({self.sub})"

@dataclass(frozen=True)
class AG:
    sub: object
    def __repr__(self): return f"AG({self.sub})"

@dataclass(frozen=True)
class AU:
    left: object
    right: object
    def __repr__(self): return f"A[{self.left} U {self.right}]"


@dataclass
class Kripke:
    states: set[str]
    transitions: dict[str, set[str]]
    labels: dict[str, set[str]]  # state → set of atomic props

    def pre(self, states: set[str]) -> set[str]:
        """States that have a successor in `states`."""
        return {s for s in self.states if self.transitions.get(s, set()) & states}

    def pre_all(self, states: set[str]) -> set[str]:
        """States where ALL successors are in `states`."""
        return {s for s in self.states
                if self.transitions.get(s, set()) and self.transitions[s] <= states}


def check(model: Kripke, formula) -> set[str]:
    """Return the set of states satisfying the CTL formula."""
    if isinstance(formula, Atom):
        return {s for s in model.states if formula.name in model.labels.get(s, set())}

    if isinstance(formula, Not):
        return model.states - check(model, formula.sub)

    if isinstance(formula, And):
        return check(model, formula.left) & check(model, formula.right)

    if isinstance(formula, Or):
        return check(model, formula.left) | check(model, formula.right)

    if isinstance(formula, EX):
        return model.pre(check(model, formula.sub))

    if isinstance(formula, AX):
        # AX φ = ¬EX(¬φ)
        return model.states - model.pre(model.states - check(model, formula.sub))

    if isinstance(formula, EF):
        # Least fixpoint: EF φ = μZ. φ ∨ EX Z
        sat = check(model, formula.sub)
        while True:
            new = sat | model.pre(sat)
            if new == sat:
                return sat
            sat = new

    if isinstance(formula, AG):
        # Greatest fixpoint: AG φ = νZ. φ ∧ AX Z
        sat = check(model, formula.sub)
        while True:
            new = sat & model.pre_all(sat)
            # Include states with no successors that satisfy φ
            no_succ = {s for s in model.states if not model.transitions.get(s, set())}
            new |= sat & no_succ
            if new == sat:
                return sat
            sat = new

    if isinstance(formula, AF):
        # AF φ = ¬EG(¬φ)
        return model.states - check(model, EG(Not(formula.sub)))

    if isinstance(formula, EG):
        # Greatest fixpoint: EG φ = νZ. φ ∧ EX Z
        sat = check(model, formula.sub)
        while True:
            new = sat & model.pre(sat)
            if new == sat:
                return sat
            sat = new

    if isinstance(formula, EU):
        # Least fixpoint: E[φ U ψ] = μZ. ψ ∨ (φ ∧ EX Z)
        phi = check(model, formula.left)
        psi = check(model, formula.right)
        sat = set(psi)
        while True:
            new = sat | (phi & model.pre(sat))
            if new == sat:
                return sat
            sat = new

    if isinstance(formula, AU):
        # A[φ U ψ] = ¬(E[¬ψ U (¬φ ∧ ¬ψ)] ∨ EG(¬ψ))
        not_phi = Not(formula.left)
        not_psi = Not(formula.right)
        return model.states - (check(model, EU(not_psi, And(not_phi, not_psi))) | check(model, EG(not_psi)))

    raise ValueError(f"Unknown formula: {type(formula)}")


def verify(model: Kripke, initial: str, formula) -> bool:
    """Check if formula holds in initial state."""
    return initial in check(model, formula)


def demo():
    print("=== CTL Model Checker ===\n")

    # Mutual exclusion protocol
    # States: n1n2 (neither critical), c1n2 (p1 critical), n1c2 (p2 critical), t1n2 (p1 trying)...
    model = Kripke(
        states={'nn', 'tn', 'nt', 'cn', 'nc', 'tt'},
        transitions={
            'nn': {'tn', 'nt'},
            'tn': {'cn', 'tt'},
            'nt': {'nc', 'tt'},
            'cn': {'nn'},
            'nc': {'nn'},
            'tt': {'cn', 'nc'},  # nondeterministic: one enters
        },
        labels={
            'nn': set(),
            'tn': {'try1'},
            'nt': {'try2'},
            'cn': {'crit1'},
            'nc': {'crit2'},
            'tt': {'try1', 'try2'},
        }
    )

    props = [
        ("Safety: AG ¬(crit1 ∧ crit2)", AG(Not(And(Atom('crit1'), Atom('crit2'))))),
        ("Liveness: AG(try1 → AF crit1)", AG(Or(Not(Atom('try1')), AF(Atom('crit1'))))),
        ("EF crit1 (reachable)", EF(Atom('crit1'))),
        ("AG(crit1 → EX ¬crit1)", AG(Or(Not(Atom('crit1')), EX(Not(Atom('crit1')))))),
    ]

    for desc, formula in props:
        sat = check(model, formula)
        holds = 'nn' in sat
        print(f"  {desc}")
        print(f"    {'✓' if holds else '✗'} (holds in {len(sat)}/{len(model.states)} states: {sorted(sat)})\n")


if __name__ == '__main__':
    if '--test' in sys.argv:
        m = Kripke(
            states={'s0', 's1', 's2'},
            transitions={'s0': {'s1'}, 's1': {'s2'}, 's2': {'s2'}},
            labels={'s0': {'a'}, 's1': {'b'}, 's2': {'c'}},
        )
        assert check(m, Atom('a')) == {'s0'}
        assert check(m, EX(Atom('b'))) == {'s0'}
        assert check(m, EF(Atom('c'))) == {'s0', 's1', 's2'}
        assert check(m, AG(Atom('c'))) == {'s2'}
        assert verify(m, 's0', EF(Atom('c')))
        assert not verify(m, 's0', AG(Atom('a')))
        print("All tests passed ✓")
    else:
        demo()
