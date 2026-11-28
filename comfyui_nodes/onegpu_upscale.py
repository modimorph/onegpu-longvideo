"""onegpu-longvideo: x2 upscale + encode INSIDE the ComfyUI graph (no raw-mp4 round trip, no second process).

OLVUpscaleVideoSave(images, audio, upscale_model, width, height, fps, quality, filename_prefix)
  images: the VAE decode output (IMAGE, float 0..1, [T, H, W, 3]); audio: the VAE audio decode (AUDIO).
  Same math as src/onegpu_longvideo/upscale.py: SPAN (spandrel) fp16 -> bicubic antialias resize in fp32 ->
  clamp*255, round, uint8 -> h264_nvenc p5 vbr cq <quality>. Audio is trimmed to the video length and muxed as AAC 192k.
  Input frames are cut to rgb24 levels the way ComfyUI's encoder does (truncation); save_raw also writes the 720p take.
  The SPAN weights stay cached on the GPU between runs (a few MB; ComfyUI's /free does not release them).
  Batches of 8 frames; D2H through pinned slots on a copy stream while the next batch computes; a writer
  thread feeds ffmpeg. Output: <ComfyUI output>/<prefix>_NNNNN_.mp4, reported like SaveVideo.
Install: copy into ComfyUI/custom_nodes/. Inert unless a graph uses it.
"""
import os, queue, subprocess, tempfile, threading, wave

import numpy as np
import torch
import torch.nn.functional as F

_MODEL = {}


def _load(name):
    import folder_paths
    path = folder_paths.get_full_path("upscale_models", name)
    if path not in _MODEL:
        from spandrel import ModelLoader
        _MODEL.clear()
        _MODEL[path] = ModelLoader().load_from_file(path).model.eval().cuda().half()
    return _MODEL[path]


def _write_wav(audio, path):
    wf = audio["waveform"][0].float().cpu().clamp(-1, 1).numpy()  # [C, N]
    pcm = (wf.T * 32767.0).round().astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1]); w.setsampwidth(2); w.setframerate(int(audio["sample_rate"])); w.writeframes(pcm.tobytes())


class OLVUpscaleVideoSave:
    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        return {"required": {"images": ("IMAGE",), "audio": ("AUDIO",),
                             "upscale_model": (folder_paths.get_filename_list("upscale_models"),),
                             "width": ("INT", {"default": 1920, "min": 64, "max": 8192}),
                             "height": ("INT", {"default": 1080, "min": 64, "max": 8192}),
                             "fps": ("FLOAT", {"default": 24.0}),
                             "quality": ("INT", {"default": 18, "min": 0, "max": 51}),
                             "filename_prefix": ("STRING", {"default": "onegpu/up"}),
                             "save_raw": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "onegpu"

    def run(self, images, audio, upscale_model, width, height, fps, quality, filename_prefix, save_raw=True):
        import math, time as _time
        import folder_paths
        import comfy.model_management as mm
        t_start = _time.perf_counter()
        out_dir, fname, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        os.makedirs(out_dir, exist_ok=True)
        file, raw_file = f"{fname}_{counter:05}_.mp4", f"{fname}_{counter:05}_raw.mp4"
        out = os.path.join(out_dir, file)
        T, H0, W0 = images.shape[0], images.shape[1], images.shape[2]
        # audio trimmed to the video length, as ComfyUI's SaveVideo does before the separate upscale copies the track
        sr = int(audio["sample_rate"]); n_keep = int(math.ceil(T * sr / float(fps)))
        aud = {"waveform": audio["waveform"][..., :n_keep], "sample_rate": sr}
        wav = os.path.join(tempfile.gettempdir(), f"olv_up_{os.getpid()}_{counter}.wav")
        raw = enc = rt = wt = None
        rawq: queue.Queue = queue.Queue(8)
        pins: queue.Queue = queue.Queue()
        qout: queue.Queue = queue.Queue()
        werr, rawerr = [], []
        try:  # one cleanup scope for the wav, both encoders and both threads
            _write_wav(aud, wav)
            mm.unload_all_models(); mm.soft_empty_cache()  # same as the separate-process path's /free: SPAN gets the VRAM
            model = _load(upscale_model)
            if save_raw:  # 720p take for the raw-take QA of the required flow (steps 6-7), from the same uint8 frames
                raw = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W0}x{H0}", "-r", str(fps), "-i", "-",
                                        "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr", "-cq", "19",
                                        "-b:v", "0", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", os.path.join(out_dir, raw_file)],
                                       stdin=subprocess.PIPE)
            enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                                    "-i", "-", "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
                                    "-cq", str(quality), "-b:v", "0", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                                    "-map_metadata", "-1", "-metadata", "comment=AI-generated", "-movflags", "+faststart", out],
                                   stdin=subprocess.PIPE)

            def raw_writer():
                while True:
                    fr = rawq.get()
                    if fr is None:
                        break
                    try:
                        if not rawerr:
                            raw.stdin.write(fr.numpy().tobytes())
                    except Exception as e:  # noqa: BLE001
                        rawerr.append(e)

            def writer():
                while True:
                    it = qout.get()
                    if it is None:
                        break
                    ev, pin, n = it
                    try:
                        ev.synchronize()  # the slot is free only after its copy finished (also on the error path)
                        if not werr:
                            enc.stdin.write(pin[:n].numpy().tobytes())
                    except Exception as e:  # noqa: BLE001
                        werr.append(e)
                    finally:
                        pins.put(pin)

            B = 8
            stream = torch.cuda.Stream()
            for _ in range(3):
                pins.put(torch.empty((B, height, width, 3), dtype=torch.uint8, pin_memory=True))
            if raw is not None:
                t_ = threading.Thread(target=raw_writer); t_.start(); rt = t_  # assigned only once started (cleanup joins it)
            t_ = threading.Thread(target=writer); t_.start(); wt = t_
            with torch.inference_mode():
                for i in range(0, T, B):
                    # rgb24 levels exactly as ComfyUI's video encoder makes them ((x*255).clamp(0,255).to(uint8): truncation)
                    x8 = images[i:i + B].mul(255).clamp_(0, 255).to(torch.uint8)
                    if raw is not None:
                        rawq.put(x8.cpu())  # bounded queue: backpressure instead of an unbounded backlog
                    t = x8.to("cuda", non_blocking=True).permute(0, 3, 1, 2).half().div_(255)
                    y = model(t)
                    y = F.interpolate(y.float(), size=(height, width), mode="bicubic", antialias=True)
                    ob = y.clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1).contiguous()
                    pin = pins.get()
                    done = torch.cuda.current_stream().record_event()
                    with torch.cuda.stream(stream):
                        stream.wait_event(done)
                        pin[: len(x8)].copy_(ob, non_blocking=True)
                        ob.record_stream(stream)
                        ev = stream.record_event()
                    qout.put((ev, pin, len(x8)))
                    if werr or rawerr:
                        break
        finally:
            if wt is not None:
                qout.put(None); wt.join()
            if rt is not None:
                rawq.put(None); rt.join()
            for p_ in (enc, raw):
                if p_ is None:
                    continue
                try:
                    p_.stdin.close()
                except OSError as e:
                    werr.append(e)
                p_.wait()
            try:
                os.remove(wav)
            except OSError:
                pass
        if werr:
            raise RuntimeError(f"OLVUpscaleVideoSave writer failed: {werr[0]!r}")
        if rawerr:
            raise RuntimeError(f"OLVUpscaleVideoSave raw writer failed: {rawerr[0]!r}")
        if enc.returncode or (raw is not None and raw.returncode):
            raise RuntimeError(f"OLVUpscaleVideoSave: ffmpeg exited with {enc.returncode} / raw {raw.returncode if raw else None}")
        items = [{"filename": file, "subfolder": subfolder, "type": "output"}]
        if raw is not None:
            items.append({"filename": raw_file, "subfolder": subfolder, "type": "output"})
        return {"ui": {"images": items, "animated": (True,), "olv_upscale_s": [round(_time.perf_counter() - t_start, 2)]}}


NODE_CLASS_MAPPINGS = {"OLVUpscaleVideoSave": OLVUpscaleVideoSave}
NODE_DISPLAY_NAME_MAPPINGS = {"OLVUpscaleVideoSave": "OLV x2 upscale + save video (in graph)"}
