from pqfl.aggregators.scheme_a import SchemeA
from pqfl.aggregators.scheme_b import SchemeB
from pqfl.aggregators.scheme_c import SchemeC

AGGREGATOR_REGISTRY = {'A': SchemeA, 'B': SchemeB, 'C': SchemeC}

__all__ = ['AGGREGATOR_REGISTRY', 'SchemeA', 'SchemeB', 'SchemeC']
