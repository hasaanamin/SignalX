"""Architectural tests enforcing clean separation of concerns and no DSP in aggregation."""

import ast
from pathlib import Path
from types import FunctionType, ModuleType

from signal_processing.context import SignalContext
from signal_processing.core import FeatureSet, FeatureValue
from signal_processing.features.aggregation import (
    FeatureAggregator,
    FeatureRegistry,
)


class DummyContext:
    def __init__(self) -> None:
        self.event_id = "ev-arch-001"

    @property
    def signal_iq(self) -> object:
        raise AssertionError("FeatureAggregator must NOT access context.signal_iq!")


class DummyExtractor:
    name = "dummy_arch"
    version = "1.0"

    def extract(self, context: SignalContext) -> FeatureSet:
        return FeatureSet(
            extractor=self.name,
            extractor_version=self.version,
            features={
                "dummy.feat": FeatureValue(value=1.23, unit="Hz", valid=True, reason=None)
            },
        )


class TestAggregationArchitecture:
    """Validate architectural invariants: no DSP, no raw IQ access, no global registries, no reference code."""

    def test_aggregator_does_not_access_raw_iq(self) -> None:
        """FeatureAggregator must never touch context.signal_iq directly."""
        ctx = DummyContext()
        registry = FeatureRegistry([DummyExtractor()])
        aggregator = FeatureAggregator(registry)

        vector = aggregator.extract(ctx)
        assert len(vector.features) == 1
        assert vector.features["dummy.feat"].value == 1.23

    def test_aggregation_source_contains_no_dsp_or_dynamic_discovery(self) -> None:
        """Inspect AST of signal_processing.features.aggregation to ensure no DSP or dynamic discovery."""
        pkg_dir = Path(__file__).resolve().parents[3] / "signal_processing" / "features" / "aggregation"

        forbidden_names = {
            "fft",
            "ifft",
            "psd",
            "compute_fft",
            "compute_psd",
            "compute_envelope",
            "compute_phase",
            "compute_autocorrelation",
            "compute_instantaneous_frequency",
            "compute_power",
            "DataFrame",
            "read_csv",
            "to_csv",
            "importlib",
            "pkgutil",
            "walk_packages",
            "StatisticalMetricFactory",
            "CorrelationMetricFactory",
            "HigherOrderStatisticsFactory",
        }

        for py_file in pkg_dir.glob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))

            for node in ast.walk(tree):
                # Check import statements
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        for forbidden in forbidden_names:
                            assert forbidden not in alias.name, f"Forbidden import '{alias.name}' in {py_file.name}"
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    for forbidden in forbidden_names:
                        assert forbidden not in mod, f"Forbidden from-import module '{mod}' in {py_file.name}"
                    for alias in node.names:
                        for forbidden in forbidden_names:
                            assert forbidden != alias.name, f"Forbidden from-import symbol '{alias.name}' in {py_file.name}"
                # Check function call names
                elif isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name):
                        assert func.id not in forbidden_names, f"Forbidden call '{func.id}' in {py_file.name}"
                    elif isinstance(func, ast.Attribute):
                        assert func.attr not in forbidden_names, f"Forbidden call attribute '{func.attr}' in {py_file.name}"

    def test_no_global_mutable_registry_exists(self) -> None:
        """Ensure no global mutable registry object exists in the aggregation package."""
        import signal_processing.features.aggregation as agg_pkg
        import signal_processing.features.aggregation.defaults as def_pkg
        import signal_processing.features.aggregation.registry as reg_pkg

        for module in (agg_pkg, def_pkg, reg_pkg):
            for attr_name in dir(module):
                if "registry" in attr_name.lower():
                    obj = getattr(module, attr_name)
                    # Allowed: FeatureRegistry class, build_default_feature_registry function, FeatureRegistryError, submodule 'registry'
                    assert isinstance(obj, (type, FunctionType, ModuleType)), (
                        f"Found potential global registry object '{attr_name}' in {module.__name__}"
                    )
                    assert not isinstance(obj, (dict, list, set, FeatureRegistry)), (
                        f"Found global registry instance '{attr_name}' in {module.__name__}"
                    )
