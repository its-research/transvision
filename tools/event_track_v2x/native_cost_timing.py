"""Explicit instrumentation of the qualified DMSTrack native inference loop.

This module does not launch an experiment or qualify its results. The caller
must bind the exact author source and independently check full output parity.
"""
import ast
import contextlib
import hashlib
import json
import math
import pathlib
import sys
import time
import types


class NativeFrameTiming:
    def __init__(self, output, synchronize, clock=time.perf_counter):
        self.output = output
        self.synchronize = synchronize
        self.clock = clock
        self.current = None
        self.completed = 0
        self.failed = 0

    @contextlib.contextmanager
    def frame(self, sequence, frame):
        if self.current is not None:
            raise RuntimeError('nested frame timing')
        row = dict(sequence=str(sequence), frame=int(frame), core_seconds=None,
                   status='failed')
        self.synchronize()
        start = self.clock()
        self.current = row
        try:
            yield
            row['status'] = 'returned'
        finally:
            try:
                self.synchronize()
                row['frame_seconds'] = self.clock() - start
                if not math.isfinite(row['frame_seconds']) or row['frame_seconds'] < 0:
                    raise RuntimeError('invalid frame timer')
                if row['status'] == 'returned' and row['core_seconds'] is None:
                    raise RuntimeError('frame did not execute tracking core')
                self.output.write(json.dumps(row, allow_nan=False) + '\n')
                self.output.flush()
                if row['status'] == 'returned':
                    self.completed += 1
                else:
                    self.failed += 1
            finally:
                self.current = None

    @contextlib.contextmanager
    def core(self):
        if self.current is None or self.current['core_seconds'] is not None:
            raise RuntimeError('core outside frame or duplicate core')
        self.synchronize()
        start = self.clock()
        try:
            yield
        finally:
            self.synchronize()
            elapsed = self.clock() - start
            if not math.isfinite(elapsed) or elapsed < 0:
                raise RuntimeError('invalid core timer')
            self.current['core_seconds'] = elapsed


def _scope(method, arguments, body):
    call = ast.Call(func=ast.Attribute(value=ast.Name(id='_rbf_native_timer', ctx=ast.Load()),
                                      attr=method, ctx=ast.Load()),
                    args=arguments, keywords=[])
    return ast.With(items=[ast.withitem(context_expr=call, optional_vars=None)], body=body)


def instrument_author_source(source, filename, expected_sha256):
    """Compile the full source with two declared timing scopes; change no files.

    The frame scope includes native input loading, diagnostics, tracking and
    native output writes. Sequence setup and end-of-run author evaluation are
    outside that scope and must be reported separately by the host controller.
    """
    if hashlib.sha256(source).hexdigest() != expected_sha256:
        raise ValueError('author source SHA mismatch')
    tree = ast.parse(source, filename=filename)
    if any(isinstance(n, ast.Name) and n.id == '_rbf_native_timer' for n in ast.walk(tree)):
        raise ValueError('reserved timer name present in author source')
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train_one_epoch']
    if len(functions) != 1:
        raise ValueError('unique author train_one_epoch required')
    loops = [n for n in ast.walk(functions[0]) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'frame']
    if len(loops) != 1:
        raise ValueError('unique native frame loop required')
    loop = loops[0]
    expected_range = ast.parse('range(min_frame, max_frame + 1)', mode='eval').body
    if ast.dump(loop.iter) != ast.dump(expected_range):
        raise ValueError('unexpected author frame range')
    core = [n for n in loop.body if isinstance(n, ast.Assign)
            and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
            and n.value.func.attr == 'track_multi_sensor_differentiable_kalman_filter'
            and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'tracker']
    if len(core) != 1:
        raise ValueError('unique top-level native tracking assignment required')
    original = ast.dump(tree, include_attributes=False)
    body = list(loop.body)
    index = body.index(core[0])
    wrapped_core = _scope('core', [], [core[0]])
    body[index] = wrapped_core
    wrapped_frame = _scope('frame', [ast.Name(id='seq_name', ctx=ast.Load()),
                                   ast.Name(id='frame', ctx=ast.Load())], body)
    loop.body = [wrapped_frame]
    code = compile(ast.fix_missing_locations(tree), filename, 'exec')
    # Structural readback: removing precisely our scopes must recover every
    # original statement and expression. This is not full scientific parity.
    loop.body = list(wrapped_frame.body)
    loop.body[index:index + 1] = wrapped_core.body
    if ast.dump(tree, include_attributes=False) != original:
        raise RuntimeError('instrumentation changed native computation AST')
    return code


def execute_author_source(path, expected_sha256, timer):
    """Execute the instrumented file with the original run_path main semantics.

    Preserve the temporary __main__ module and argv[0] behavior so author
    imports and command parsing see the same native entry. The caller still
    owns argv[1:], working directory, dependency path and runtime admission.
    """
    filename = str(path)
    code = instrument_author_source(pathlib.Path(path).read_bytes(), filename, expected_sha256)
    module = types.ModuleType('__main__')
    namespace = module.__dict__
    namespace.update(__file__=filename, __package__='', __spec__=None,
                     __cached__=None, __loader__=None, _rbf_native_timer=timer)
    previous_module = sys.modules.get('__main__')
    previous_argv0 = sys.argv[0]
    sys.modules['__main__'] = module
    sys.argv[0] = filename
    try:
        exec(code, namespace)
        return namespace.copy()
    finally:
        sys.argv[0] = previous_argv0
        if previous_module is None:
            sys.modules.pop('__main__', None)
        else:
            sys.modules['__main__'] = previous_module
