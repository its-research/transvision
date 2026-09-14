from types import SimpleNamespace
import json

from tools.event_track_v2x.convert_cooptrack_fit_inputs import configure_converter, write_location_tables


def test_disables_frame_removal_and_occlusion_interpolation():
    converter = SimpleNamespace(to_remove_list_veh=["1"], to_remove_list_coop=["2"])
    configure_converter(converter)
    annotations = {"1": {"original": {"x": 1.0}}}
    assert converter.to_remove_list_veh == []
    assert converter.to_remove_list_coop == []
    assert converter._generate_unvisible_annotations(
        "vehicle-side", {}, {}, {}, annotations, {}) is annotations


def test_unreported_location_is_preserved_without_inventing_junction(tmp_path):
    (tmp_path / "v1.0-trainval").mkdir()
    samples = {"1": {"token": "1", "scene_token": "0000", "location": ""},
               "2": {"token": "2", "scene_token": "0000", "location": ""}}
    assert write_location_tables(samples, tmp_path) == 2
    logs = json.loads((tmp_path / "v1.0-trainval/log.json").read_text())
    scenes = json.loads((tmp_path / "v1.0-trainval/scene.json").read_text())
    assert len(logs) == 1 and logs[0]["location"] == ""
    assert logs[0]["date_captured"] == ""
    assert scenes[0]["log_token"] == logs[0]["token"]
    assert scenes[0]["nbr_samples"] == 2


def test_mixed_scene_locations_are_recorded_without_selecting_one(tmp_path):
    (tmp_path / "v1.0-trainval").mkdir()
    samples = {"1": {"token": "1", "scene_token": "0000", "location": ""},
               "2": {"token": "2", "scene_token": "0000", "location": "yizhuang06"}}
    assert write_location_tables(samples, tmp_path) == 1
    logs = json.loads((tmp_path / "v1.0-trainval/log.json").read_text())
    assert logs[0]["location"] == ""
    assert logs[0]["source_locations"] == ["", "yizhuang06"]
    assert samples["2"]["location"] == "yizhuang06"
