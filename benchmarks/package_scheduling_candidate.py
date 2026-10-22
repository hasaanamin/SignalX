"""Package current source and actual evidence, never stale build/installed trees.

This deliberately creates a BLOCKED candidate, not a timing sign-off. Run only
after qualification finishes and documentation reflects the observed failures.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tomllib
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'))
    parser.add_argument('--verification',type=Path,required=True)
    args=parser.parse_args()
    root,q=args.root.resolve(),args.verification.resolve()
    execution=json.loads((q/'execution.json').read_text(encoding='utf-8-sig'))
    # Verify source identity has not drifted since installed verification.
    manifest=list(csv.DictReader((q/'production_manifest.csv').open(encoding='utf-8-sig')))
    for row in manifest:
        assert digest(root/row['path'])==row['sha256'], row['path']
    version=tomllib.loads((root/'pyproject.toml').read_text())['project']['version']
    out=q/'candidate_artifacts'
    out.mkdir(exist_ok=False)
    wheels=list((q/'wheels').glob('*.whl'))
    assert len(wheels)==1
    wheel=out/wheels[0].name
    shutil.copy2(wheels[0],wheel)
    roots=('signal_processing','benchmarks','tests','checks','docs')
    allowed={'.py','.md','.txt','.toml','.json','.npz','.csv','.ps1','.c','.h'}
    excluded={'__pycache__','build','dist','.git','node_modules'}
    files={path for directory in roots if (root/directory).exists()
        for path in (root/directory).rglob('*') if path.is_file()
        and (path.suffix in allowed or path.name.startswith(('LICENSE','NOTICE','COPYING')))
        and not excluded.intersection(path.relative_to(root).parts)}
    files.update(path for path in root.iterdir() if path.is_file() and
        (path.suffix=='.md' or path.name=='pyproject.toml' or path.name.startswith(('LICENSE','requirements'))))
    source=out/f'signal_processing-{version}-scheduling-candidate.zip'
    rows=[]
    with zipfile.ZipFile(source,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            name=path.relative_to(root).as_posix()
            archive.write(path,name)
            rows.append((name,digest(path),path.stat().st_size))
        content=io.StringIO()
        writer=csv.writer(content);writer.writerow(('path','sha256','bytes'));writer.writerows(rows)
        archive.writestr('SOURCE_MANIFEST.csv',content.getvalue())
    with zipfile.ZipFile(source) as archive:
        assert archive.testzip() is None
    evidence=out/f'signal_processing-{version}-scheduling-evidence.zip'
    with zipfile.ZipFile(evidence,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for base,prefix in ((q.parent,'observations'),(q,'qualification')):
            for path in sorted(base.iterdir()):
                if path.is_file() and path.suffix in {'.json','.log','.xml','.csv','.prof'}:
                    archive.write(path,f'{prefix}/{path.name}')
        for path in sorted((q/'numerical').iterdir()):
            if path.is_file():archive.write(path,'qualification/numerical/'+path.name)
        for name in ('ADAPTIVE_LTE_SCHEDULING_REPAIRS_20261020.md','FINAL_RELEASE_LOCK.md',
                     'requirements-acquisition-observed.txt'):
            if (root/name).exists():archive.write(root/name,name)
    with zipfile.ZipFile(evidence) as archive:
        assert archive.testzip() is None
    report=dict(version=version,status='BLOCKED',
        reason='Scheduled acquisition/tail-latency gates have not passed; not a frozen release.',
        source_entries=len(rows),production_modules=len(manifest),functional_checks=execution,
        artifacts={path.name:dict(bytes=path.stat().st_size,sha256=digest(path))
                   for path in (source,wheel,evidence)})
    (out/'ARTIFACTS.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
