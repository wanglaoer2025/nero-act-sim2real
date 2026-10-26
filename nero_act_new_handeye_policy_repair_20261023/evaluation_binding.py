"""Bind actual imported runtime files, including modules loaded later by rollouts."""
from pathlib import Path
import sys

BASE=Path(__file__).resolve().parent.parent


def actual_sources(evaluator):
    wrapper=evaluator.evaluate_dense
    paths={Path(__file__).resolve(),Path(wrapper.__file__).resolve(),
           Path(evaluator.__file__).resolve()}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            path=Path(filename).resolve()
            if path.is_relative_to(BASE) and path.suffix=='.py' and path.is_file():
                paths.add(path)
    # Dynamic imports use these package roots; bind their real files, not historical copies.
    roots=[Path(wrapper.runtime.__file__).parent]
    for name in ['executor','training','src.act']:
        module=sys.modules.get(name)
        roots.extend(Path(p) for p in getattr(module,'__path__',()))
    for root in roots:
        for path in root.rglob('*'):
            if path.is_file() and path.suffix.lower() in {'.py','.xml','.stl','.obj','.msh','.urdf','.json','.png'}:
                paths.add(path.resolve())
    return sorted(paths)


def bind_snapshot(snapshot, dependencies):
    dependencies=tuple(dependencies)
    def bound(paths):
        return snapshot([*paths,*dependencies])
    return bound
