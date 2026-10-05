"""Unpack the byte-bound producer source; no repository checkout is used."""
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile
from clearml import Task

SOURCE_BASE64 = '__SOURCE_BASE64__'


def main():
    task = Task.current_task() or Task.init(project_name='Thesis/Recover-Before-Fuse/Training',
        task_name='RBF nested detector seen-val raw export',reuse_last_task_id=False,
        auto_connect_frameworks=False,auto_connect_arg_parser=False)
    parameters = task.get_parameters()
    recipe = json.loads(parameters['General/recipe'])
    if hashlib.sha256(json.dumps(recipe,sort_keys=True,separators=(',',':')).encode()).hexdigest() != parameters['General/recipe_sha256']:
        raise ValueError('recipe differs')
    raw = base64.b64decode(SOURCE_BASE64)
    if hashlib.sha256(raw).hexdigest() != recipe['execution_source_zip_sha256']:
        raise ValueError('source archive differs')
    root = Path('rbf-seen-val-export-source')
    root.mkdir()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if set(archive.namelist()) != set(recipe['source_inventory']):
            raise ValueError('source inventory differs')
        for name in archive.namelist():
            if Path(name).name != name:
                raise ValueError('unsafe source name')
            data = archive.read(name)
            item = recipe['source_inventory'][name]
            if len(data) != item['bytes'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('source member differs')
            (root/name).write_bytes(data)
    subprocess.run([sys.executable,str(root/'run_rbf_nested_seen_val_export.py')],check=True)


if __name__ == '__main__':
    main()
