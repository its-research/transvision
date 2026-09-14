"""Diagnostic single-class completion bounds; NEVER partition/mass bounds.

The existing tracker is not modified by this module. A fixed-prefix identity
class sums all equivalent parent paths. Unassigned parents may join any one
known root: pooling their weights with the largest known-root group yields a
relaxation of the largest completion, not the sum of all completions.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import sys


def logsum(values):
    values=tuple(values)
    if not values: return -math.inf
    maximum=max(values)
    if maximum==-math.inf: return maximum
    return maximum+math.log(math.fsum(math.exp(v-maximum) for v in values))


@dataclass(frozen=True)
class RankingBound:
    kind: str
    prefix_depth: int
    remaining_rows: int
    log_prefix_class_weight: float | None
    log_single_class_upper: float | None
    log_suffix_single_class_factor_upper: float
    log_suffix_row_partition_relaxation: float
    log_row_partition_upper: float | None
    floating_margin: float
    inspected_prefix_factor_terms: int
    inspected_suffix_factor_terms: int
    validated_catalog_factor_terms: int
    partition_mass_bound: bool = False
    formal_numeric_certificate: bool = False


class SingleClassBoundModel:
    """A bounded immutable factor catalog for proof and closed-state diagnostics.

    Prefix reconstruction and validation work is reported, not hidden as O(1).
    This deliberately simple diagnostic API is not an optimized online kernel.
    """
    def __init__(self,rows,*,slots=None,max_nodes=10_000,max_factor_terms=200_000):
        if (type(max_nodes) is not int or max_nodes<1 or type(max_factor_terms) is not int or max_factor_terms<1
                or not isinstance(rows,(tuple,list)) or len(rows)>max_nodes):
            raise ValueError('bounded indexed factor rows required')
        normalized=[];terms=0
        for index,row in enumerate(rows):
            values={}
            for parent,weight in row:
                terms+=1
                if terms>max_factor_terms: raise ValueError('factor catalog work limit exceeded')
                if (type(parent) is not int or not -1<=parent<index or parent in values
                        or isinstance(weight,bool) or not isinstance(weight,Real) or not math.isfinite(weight)):
                    raise ValueError('unique causal parents and finite log potentials required')
                values[parent]=float(weight)
            if -1 not in values: raise ValueError('positive birth potential required in every row')
            normalized.append(tuple(sorted(values.items())))
        self.rows=tuple(normalized);self.catalog_terms=terms
        if slots is not None:
            if len(slots)!=len(rows): raise ValueError('one source/frame slot per node required')
            try: self.slots=tuple(tuple(slot) for slot in slots);set(self.slots)
            except (TypeError,ValueError) as error: raise ValueError('hashable source/frame slots required') from error
            if any(len(slot)!=2 for slot in self.slots): raise ValueError('source/frame pairs required')
        else: self.slots=None

    def _suffix(self,groups):
        depth=len(groups);maximums=[];totals=[];terms=0
        for row in self.rows[depth:]:
            known={};unknown=[];all_weights=[];birth=None
            for parent,weight in row:
                terms+=1
                all_weights.append(weight)
                if parent<0: birth=weight
                elif parent<depth: known.setdefault(groups[parent],[]).append(weight)
                else: unknown.append(weight)
            best_known=max((logsum(weights) for weights in known.values()),default=-math.inf)
            matched=logsum((best_known,logsum(unknown)))
            maximums.append(max(birth,matched))
            totals.append(logsum(all_weights))
        return maximums,totals,terms

    def _result(self,groups,prefix_weight,prefix_terms,kind):
        maximums,totals,suffix_terms=self._suffix(groups)
        # Conservative floating slack only; not directed interval arithmetic.
        scale=max(1.,abs(prefix_weight or 0.),math.fsum(abs(v) for v in totals),
                  math.fsum(abs(v) for v in maximums))
        margin=64*sys.float_info.epsilon*(len(self.rows)+1)*scale
        suffix=math.fsum([*maximums,margin]);row_relaxation=math.fsum([*totals,margin])
        return RankingBound(kind,len(groups),len(self.rows)-len(groups),prefix_weight,
            None if prefix_weight is None else math.fsum((prefix_weight,suffix)),suffix,row_relaxation,
            None if prefix_weight is None else math.fsum((prefix_weight,row_relaxation)),margin,
            prefix_terms,suffix_terms,self.catalog_terms)

    def for_prefix(self,roots):
        """Bound completions of one legal canonical root-assignment prefix."""
        roots=tuple(roots)
        if len(roots)>len(self.rows): raise ValueError('prefix exceeds the current factor catalog')
        prefix_terms=0;terms=[];occupied=set()
        for index,root in enumerate(roots):
            if type(root) is not int or not 0<=root<=index or roots[root]!=root:
                raise ValueError('canonical causal identity roots required')
            if self.slots is not None:
                key=root,self.slots[index]
                if key in occupied: raise ValueError('one identity cannot occupy the same source/frame twice')
                occupied.add(key)
            row=self.rows[index];prefix_terms+=len(row)
            if root==index: terms.append(dict(row)[-1])
            else:
                selected=[weight for parent,weight in row if parent>=0 and roots[parent]==root]
                if not selected: raise ValueError('prefix class has no supported parent path')
                terms.append(logsum(selected))
        return self._result(roots,math.fsum(terms),prefix_terms,'fixed_prefix_single_class_ranking_bound_v1')

    def for_predecessor_partition(self,groups):
        """Universal suffix bound before choosing a retained prior combination.

        Every old support edge must remain within a group. Then no old identity
        can span groups, even though each group may contain many possible roots.
        Multiply this suffix bound by an independently valid prior-class weight
        bound; this API deliberately does not invent that prior weight.
        """
        groups=tuple(groups)
        if len(groups)>len(self.rows) or any(type(g) is not int or g<0 for g in groups):
            raise ValueError('bounded predecessor group assignment required')
        terms=0
        for index,row in enumerate(self.rows[:len(groups)]):
            for parent,_ in row:
                terms+=1
                if parent>=0 and groups[parent]!=groups[index]:
                    raise ValueError('predecessor partition cuts an old support edge')
        return self._result(groups,None,terms,'predecessor_partition_suffix_ranking_bound_v1')
