"""Build a frozen, allowlisted release without copying results, secrets or environments."""

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def release_files(root=ROOT):
    root = Path(root)
    paths = []
    for pattern in ['*.py', '*.md', 'upstream_commit.txt', 'config/*.yaml', 'scripts/*.sh', 'scripts/*.sbatch']:
        paths.extend((root / 'assignment2').glob(pattern))
    for pattern in ['*.py', '*.yaml', '*.json', '*.md', '*.png', '*.pkl', '*.ttf']:
        paths.extend((root / 'balrog').rglob(pattern))
    paths.extend((root / 'tests').glob('test_assignment2*.py'))
    paths.extend((root / 'tests/fixtures').rglob('*'))
    return sorted({path for path in paths if path.is_file() and '__pycache__' not in path.parts})


def build_release(output=None):
    files = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in release_files()}
    content_id = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=normal'], cwd=ROOT,
                                capture_output=True, text=True, check=True).stdout.strip())
    name = f'ass2-v2-{content_id[:12]}'
    output = Path(output or ROOT / 'assignment2/deployment')
    output.mkdir(parents=True, exist_ok=True)
    archive_path = output / f'{name}.tar.gz'
    receipt = {'source_commit': commit, 'source_tree_dirty': dirty, 'source_content_id': content_id,
               'note': 'source_commit is the base Git commit; files/content hash identify the executed working tree.',
               'files': files}
    with tarfile.open(archive_path, 'w:gz') as archive:
        for relative in files:
            archive.add(ROOT / relative, arcname=f'{name}/{relative}', recursive=False)
        payload = json.dumps(receipt, indent=2).encode()
        info = tarfile.TarInfo(f'{name}/assignment2/deployed_version.json')
        info.size = len(payload)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(payload))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    archive_path.with_suffix(archive_path.suffix + '.sha256').write_text(f'{digest}  {archive_path.name}\n')
    print(f'Release: {name}\nBundle: {archive_path}\nSHA256: {digest}\nFiles: {len(files)}')
    return archive_path


if __name__ == '__main__':
    build_release()
