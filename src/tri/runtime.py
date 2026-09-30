"""Small durable I/O helpers for long-running experiments."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile


def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def atomic_jsonl(path, rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def exclusive_run(path):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as stream:
        try:
            fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'another process already holds {path}') from error
        stream.seek(0)
        stream.truncate()
        stream.write(str(os.getpid()))
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(),fcntl.LOCK_UN)


def read_jsonl(path, *, recover_tail=False):
    path=Path(path)
    if not path.exists():
        return []
    data=path.read_bytes()
    lines=data.splitlines(keepends=True)
    rows=[]
    valid_bytes=0
    for i,line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (json.JSONDecodeError,UnicodeDecodeError):
            if not recover_tail or i!=len(lines)-1 or line.endswith(b'\n'):
                raise ValueError(f'corrupt result record {i+1} in {path}')
            with path.open('r+b') as stream:
                stream.truncate(valid_bytes)
            break
        valid_bytes+=len(line)
    # A complete final JSON record without newline is valid; separate the next
    # append explicitly so recovery cannot concatenate two objects.
    if recover_tail and rows and path.stat().st_size and not path.read_bytes().endswith(b'\n'):
        with path.open('ab') as stream:
            stream.write(b'\n')
    return rows
