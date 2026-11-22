import importlib.util
import unittest

from native.document import Document


class WireFeedbackTests(unittest.TestCase):
    def test_dimension_mismatch_on_incomplete_graph_is_located_at_port(self):
        self.assertIsNotNone(importlib.util.find_spec('native.wire_feedback'), 'port feedback is missing')
        from native.wire_feedback import assess_link, source_contract
        document = Document()
        document.set_input('fuse', 'b', None)
        key = document.semantic_key()
        good = assess_link(document, 'fuse', 'b', 'call:encode_b:out')
        self.assertEqual(good['status'], 'compatible')
        bad = assess_link(document, 'fuse', 'b', 'input:b')
        self.assertEqual(bad['status'], 'incompatible')
        self.assertEqual(bad['port'], 'b')
        self.assertEqual(bad['actual']['shape'], [None, 4])
        self.assertEqual(bad['expected']['shape'], [None, 8])
        self.assertEqual(document.semantic_key(), key)
        self.assertIsNone(source_contract(document, 'call:fuse:out'))
        self.assertEqual(assess_link(document, 'classify', 'x', 'call:fuse:out')['status'], 'pending')
