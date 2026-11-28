"""ComfyClient.run with the in-graph upscale node: picks the 1080p file, downloads the raw take, reads olv_upscale_s."""
from onegpu_longvideo.comfy import ComfyClient


def test_run_in_graph_outputs(tmp_path, monkeypatch):
    h = {"status": {"completed": True, "messages": [["execution_start", {"timestamp": 1000}], ["execution_success", {"timestamp": 151000}]]},
         "outputs": {"save": {"images": [{"filename": "r_1080p_00001_.mp4", "subfolder": "onegpu", "type": "output"},
                                         {"filename": "r_1080p_00001_raw.mp4", "subfolder": "onegpu", "type": "output"}],
                              "animated": [True], "olv_upscale_s": [7.25]}}}
    c = ComfyClient("http://x")
    got = []
    monkeypatch.setattr(c, "queue", lambda g: "pid")
    monkeypatch.setattr(c, "wait", lambda pid, t: h)
    monkeypatch.setattr(c, "download", lambda item, dest: (got.append((item["filename"], str(dest))), dest)[1])
    res = c.run({}, tmp_path / "r_1080p.mp4", raw_dest=tmp_path / "r_raw.mp4")
    assert res["remote_file"] == "r_1080p_00001_.mp4" and res["exec_s"] == 150.0
    assert res["upscale_in_graph_s"] == 7.25 and res["raw_file"].endswith("r_raw.mp4")
    assert got == [("r_1080p_00001_.mp4", str(tmp_path / "r_1080p.mp4")), ("r_1080p_00001_raw.mp4", str(tmp_path / "r_raw.mp4"))]


def test_run_without_raw_dest_skips_raw(tmp_path, monkeypatch):
    h = {"status": {"completed": True, "messages": []},
         "outputs": {"save": {"images": [{"filename": "a.mp4", "subfolder": "", "type": "output"}, {"filename": "a_raw.mp4", "subfolder": "", "type": "output"}]}}}
    c = ComfyClient("http://x")
    monkeypatch.setattr(c, "queue", lambda g: "pid")
    monkeypatch.setattr(c, "wait", lambda pid, t: h)
    calls = []
    monkeypatch.setattr(c, "download", lambda item, dest: (calls.append((item["filename"], str(dest))), dest)[1])
    res = c.run({}, tmp_path / "a.mp4")
    assert "raw_file" not in res and res["remote_file"] == "a.mp4"
    assert calls == [("a.mp4", str(tmp_path / "a.mp4"))]
