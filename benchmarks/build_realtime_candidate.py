"""Build isolated source/wheel/install trees and verify production byte parity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-report',type=Path,required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    report = ET.parse(args.source_report).getroot()
    suites = [report] if report.tag == 'testsuite' else list(report.iter('testsuite'))
    counts = {name:sum(int(suite.get(name,'0')) for suite in suites)
              for name in ('tests','failures','errors','skipped')}
    if not counts['tests'] or any(counts[name] for name in ('failures','errors','skipped')):
        raise ValueError('A complete passing source report is required')
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    stage,installed,wheels = (output/name for name in ('build_source','installed','wheels'))
    stage.mkdir()
    for filename in ('pyproject.toml','README.md','LICENSE','LICENSE.txt'):
        if (root/filename).is_file():
            shutil.copy2(root/filename,stage/filename)
    shutil.copytree(root/'signal_processing',stage/'signal_processing',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.nbi','*.nbc'))
    modules = sorted((root/'signal_processing').rglob('*.py'))
    rows = [dict(path=path.relative_to(root).as_posix(),sha256=digest(path),bytes=path.stat().st_size)
            for path in modules]
    with (output/'production_manifest.csv').open('w',newline='',encoding='utf8') as target:
        writer = csv.DictWriter(target,fieldnames=('path','sha256','bytes'))
        writer.writeheader()
        writer.writerows(rows)
    with (output/'build.log').open('x',encoding='utf8') as log:
        subprocess.run([sys.executable,'-m','pip','wheel','--no-deps','--no-build-isolation',
                        '--wheel-dir',str(wheels),str(stage)],stdout=log,stderr=subprocess.STDOUT,check=True)
        wheel, = wheels.glob('*.whl')
        subprocess.run([sys.executable,'-m','pip','install','--no-deps','--target',str(installed),str(wheel)],
                        stdout=log,stderr=subprocess.STDOUT,check=True)
    with zipfile.ZipFile(wheel) as archive:
        assert archive.testzip() is None
        packed = {name for name in archive.namelist() if name.startswith('signal_processing/') and name.endswith('.py')}
        assert packed == {row['path'] for row in rows}
        for row in rows:
            name = row['path']
            assert digest(root/name) == row['sha256'],name
            assert hashlib.sha256(archive.read(name)).hexdigest() == row['sha256'],name
            assert digest(installed/name) == row['sha256'],name
    result = dict(source_checks=counts,source_report=str(args.source_report),
        source_report_sha256=digest(args.source_report),production_modules=len(rows),
        wheel=str(wheel),wheel_sha256=digest(wheel),all_production_modules_identical=True,
        scope='Packaging and production byte parity; performance qualification is separate')
    (output/'wheel_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
