"""ComfyClient.run with the in-graph upscale node: picks that node's 1080p file, downloads the raw take, reads olv_upscale_s."""
import pytest

from onegpu_longvideo.comfy import ComfyClient

G = {"save": {"class_type": "OLVUpscaleVideoSave", "inputs": {}}, "prev": {"class_type": "PreviewImage", "inputs": {}}}
UP = {"filename": "r_1080p_00001_.mp4", "subfolder": "onegpu", "type": "output"}
RAW = {"filename": "r_1080p_00001_raw.mp4", "subfolder": "onegpu", "type": "output"}
MSG = [["execution_start", {"timestamp": 1000}], ["execution_success", {"timestamp": 151000}]]


def client(monkeypatch, h):
    c = ComfyClient("http://x")
    calls = []
    monkeypatch.setattr(c, "queue", lambda g: "pid")
    monkeypatch.setattr(c, "wait", lambda pid, t: h)
    monkeypatch.setattr(c, "download", lambda item, dest: (calls.append((item["filename"], str(dest))), dest)[1])
    return c, calls


def test_run_in_graph_outputs(tmp_path, monkeypatch):
    h = {"status": {"completed": True, "messages": MSG},
         "outputs": {"prev": {"images": [{"filename": "preview.png", "subfolder": "", "type": "temp"}]},
                     "save": {"images": [UP, RAW], "animated": [True], "olv_upscale_s": [7.25]}}}
    c, calls = client(monkeypatch, h)
    res = c.run(G, tmp_path / "r_1080p.mp4", raw_dest=tmp_path / "r_raw.mp4")
    assert res["remote_file"] == UP["filename"] and res["exec_s"] == 150.0
    assert res["upscale_in_graph_s"] == 7.25 and res["raw_file"].endswith("r_raw.mp4")
    assert calls == [(UP["filename"], str(tmp_path / "r_1080p.mp4")), (RAW["filename"], str(tmp_path / "r_raw.mp4"))]


def test_cached_node_reports_no_upscale_time(tmp_path, monkeypatch):
    h = {"status": {"completed": True, "messages": MSG + [["execution_cached", {"nodes": ["save"]}]]},
         "outputs": {"save": {"images": [UP, RAW], "olv_upscale_s": [7.25]}}}
    c, _ = client(monkeypatch, h)
    res = c.run(G, tmp_path / "a.mp4", raw_dest=tmp_path / "a_raw.mp4")
    assert "upscale_in_graph_s" not in res


def test_missing_raw_take_fails(tmp_path, monkeypatch):
    h = {"status": {"completed": True, "messages": MSG}, "outputs": {"save": {"images": [UP]}}}
    c, calls = client(monkeypatch, h)
    with pytest.raises(RuntimeError, match="raw take"):
        c.run(G, tmp_path / "a.mp4", raw_dest=tmp_path / "a_raw.mp4")
    assert calls == []  # nothing written before the check


def test_plain_graph_without_raw_dest(tmp_path, monkeypatch):
    h = {"status": {"completed": True, "messages": []},
         "outputs": {"save": {"images": [{"filename": "a.mp4", "subfolder": "", "type": "output"},
                                         {"filename": "a_raw.mp4", "subfolder": "", "type": "output"}]}}}
    c, calls = client(monkeypatch, h)
    res = c.run({"save": {"class_type": "SaveVideo", "inputs": {}}}, tmp_path / "a.mp4")
    assert "raw_file" not in res and res["remote_file"] == "a.mp4"
    assert calls == [("a.mp4", str(tmp_path / "a.mp4"))]


def test_raw_dest_needs_the_upscale_node(tmp_path, monkeypatch):
    h = {"status": {"completed": True, "messages": []}, "outputs": {"save": {"images": [UP, RAW]}}}
    c, calls = client(monkeypatch, h)
    with pytest.raises(RuntimeError, match="raw take"):
        c.run({"save": {"class_type": "SaveVideo", "inputs": {}}}, tmp_path / "a.mp4", raw_dest=tmp_path / "a_raw.mp4")
    assert calls == []
