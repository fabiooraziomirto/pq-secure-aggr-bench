from pqfl.aggregators.scheme_a import SchemeA
from pqfl.aggregators.scheme_a_dropout import SchemeADropout
from pqfl.aggregators.scheme_b import SchemeB
from pqfl.aggregators.scheme_b_bgv import SchemeBGV
from pqfl.aggregators.scheme_b_threshold import SchemeBGVThreshold
from pqfl.aggregators.scheme_c import SchemeC

# A-dropout and B-BGV-threshold use the 'quorum' participation policy, so they are covered by
# tests/test_dropout*.py and not by the all_or_abort conformance suite.
AGGREGATOR_REGISTRY = {'A': SchemeA, 'B': SchemeB, 'B-BGV': SchemeBGV, 'C': SchemeC,
                       'A-dropout': SchemeADropout, 'B-BGV-threshold': SchemeBGVThreshold}

__all__ = ['AGGREGATOR_REGISTRY', 'SchemeA', 'SchemeADropout', 'SchemeB', 'SchemeBGV', 'SchemeBGVThreshold', 'SchemeC']
