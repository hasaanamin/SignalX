if __name__ == "__main__" and not __package__:
    import sys
    from pathlib import Path
    _pkg_dir = Path(__file__).resolve().parent
    _src_dir = _pkg_dir.parent.parent
    for _p in [str(_src_dir), str(_pkg_dir.parent)]:
        if _p not in sys.path:
            sys.path.insert(0, _p)
    import importlib
    importlib.import_module(f"packages.{_pkg_dir.name}")
    __package__ = f"packages.{_pkg_dir.name}"

# Re-export from Template Matching
import importlib
_mod = importlib.import_module(".Template Matching", package=__package__)
TemplateMatchingMetric = _mod.TemplateMatchingMetric
compute_template_matching = _mod.compute_template_matching
findsignal = _mod.findsignal
__all__ = ["TemplateMatchingMetric", "compute_template_matching", "findsignal"]


def main() -> None:
    print("=" * 60)
    print("  Template_Matching: Demonstration (Alias)")
    print("=" * 60)
    if hasattr(_mod, "main"):
        _mod.main()
    else:
        print("Module loaded successfully.")


if __name__ == "__main__":
    main()
