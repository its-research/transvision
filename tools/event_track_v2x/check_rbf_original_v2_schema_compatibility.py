"""Compare original cache schema definitions after the documented extraction."""
import ast
import copy
import hashlib
import json
from pathlib import Path

R = Path('/Volumes/Data/test/recover-before-fuse')
MODEL = R / 'artifacts/rbf-original-joint-identity-source-byte-readback-20261001/source'
BUILDER = R / 'artifacts/rbf-original-nested-cache-builder-source-byte-readback-20261004/code-unpack'


class StripDocs(ast.NodeTransformer):
    def generic_visit(self, node):
        node = super().generic_visit(node)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:]
        return node


def compare():
    relative = 'transvision/models/event_track_v2x/detection_cache_v2.py'
    old_path, new_path = MODEL / relative, BUILDER / relative
    old_sha, new_sha = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (old_path, new_path)]
    assert old_sha == '9578ffd664c530c6436a791d6fb2bdf606d739b916216a0d09253a5b92efc8f0'
    assert new_sha == '227b909430eabbb2b35870b4b9f97e357a07d74389da471c535bee68394988ed'
    old, new = [StripDocs().visit(ast.parse(path.read_text())) for path in (old_path, new_path)]
    cls = next(node for node in old.body if isinstance(node, ast.ClassDef) and node.name == 'DetectionCacheV2')
    helper = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_validate_dataset_feature')
    post = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__post_init__')
    calls = [i for i, node in enumerate(post.body) if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == '_validate_dataset_feature']
    assert len(calls) == 1
    post.body[calls[0]:calls[0]+1] = copy.deepcopy(helper.body)
    cls.body.remove(helper)

    def definitions(module):
        result = {}
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign)):
                key = node.name if hasattr(node, 'name') else ','.join(ast.dump(target) for target in node.targets)
                result[key] = ast.dump(node, include_attributes=False)
        return result

    assert definitions(old) == definitions(new)
    feature = 'transvision/models/event_track_v2x/prediction_features.py'
    assert (MODEL / feature).read_bytes() == (BUILDER / feature).read_bytes()
    return dict(kind='rbf_published_builder_and_model_V2_schema_normalized_AST_compatibility_v1',
        model_schema_file_sha256=old_sha, builder_schema_file_sha256=new_sha,
        source_bytes_identical=False, source_files_not_modified=True,
        normalization=['remove docstrings', 'inline original dataset/feature validator call in __post_init__'],
        all_schema_constants_functions_and_class_definitions_identical_after_normalization=True,
        prediction_features_file_byte_identical=True,
        actual_val_cache_full_model_loader_admission=False,
        full_online_RBF_accepted=False, paper_performance_complete=False)


if __name__ == '__main__':
    print(json.dumps(compare()), flush=True)
