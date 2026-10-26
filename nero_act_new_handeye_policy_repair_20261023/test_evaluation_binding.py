import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent


class EvaluationBindingTests(unittest.TestCase):
    def binding(self):
        path=ROOT/'evaluation_binding.py'
        self.assertTrue(path.is_file(),'Missing actual-runtime source binding')
        spec=importlib.util.spec_from_file_location('eval_binding',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_actual_runtime_and_predictive_guard_are_in_the_source_set(self):
        binding=self.binding()
        sys.path.insert(0,str(ROOT))
        import evaluate_adaptation as evaluator
        sources=binding.actual_sources(evaluator)
        wrapper=evaluator.evaluate_dense
        expected=[Path(wrapper.__file__),Path(wrapper.runtime.__file__),
            Path(wrapper.runtime.__file__).parent/'visual_evaluation.py',
            ROOT.parent/'nero_act_sampling_retention_20261023/predictive_guard.py']
        self.assertTrue({p.resolve() for p in expected}<=set(sources))

    def test_a_guard_edit_after_snapshot_is_rejected(self):
        binding=self.binding()
        sys.path.insert(0,str(ROOT.parent/'nero_act_grouped_gripper_20261021'))
        from training.integrity import snapshot,require_unchanged
        with tempfile.TemporaryDirectory() as directory:
            entry=Path(directory)/'entry.py';entry.write_text('entry')
            guard=Path(directory)/'guard.py';guard.write_text('original protection')
            wrapped=binding.bind_snapshot(snapshot,[guard])
            protected=wrapped([entry])
            self.assertIn(str(guard.resolve()),protected)
            require_unchanged(protected)
            guard.write_text('changed protection')
            with self.assertRaises((ValueError,RuntimeError)):
                require_unchanged(protected)


if __name__=='__main__':unittest.main()
