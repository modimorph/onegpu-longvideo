"""onegpu-longvideo: x2 upscale + encode INSIDE the ComfyUI graph (no raw-mp4 round trip, no second process).

OLVUpscaleVideoSave(images, audio, upscale_model, width, height, fps, quality, filename_prefix)
  images: the VAE decode output (IMAGE, float 0..1, [T, H, W, 3]); audio: the VAE audio decode (AUDIO).
  Same math as src/onegpu_longvideo/upscale.py: SPAN (spandrel) fp16 -> bicubic antialias resize in fp32 ->
  clamp*255, round, uint8 -> h264_nvenc p5 vbr cq <quality>. Audio is muxed as AAC 192k.
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
                             "filename_prefix": ("STRING", {"default": "onegpu/up"})}}

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "onegpu"

    def run(self, images, audio, upscale_model, width, height, fps, quality, filename_prefix):
        import folder_paths
        out_dir, fname, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        os.makedirs(out_dir, exist_ok=True)
        file = f"{fname}_{counter:05}_.mp4"
        out = os.path.join(out_dir, file)
        wav = os.path.join(tempfile.gettempdir(), f"olv_up_{os.getpid()}_{counter}.wav")
        _write_wav(audio, wav)
        model = _load(upscale_model)
        enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps),
                                "-i", "-", "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
                                "-cq", str(quality), "-b:v", "0", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest",
                                "-map_metadata", "-1", "-metadata", "comment=AI-generated", "-movflags", "+faststart", out],
                               stdin=subprocess.PIPE)
        B = 8
        T = images.shape[0]
        pins: queue.Queue = queue.Queue()
        qout: queue.Queue = queue.Queue()
        werr = []
        try:
            stream = torch.cuda.Stream()
            for _ in range(3):
                pins.put(torch.empty((B, height, width, 3), dtype=torch.uint8, pin_memory=True))

            def writer():
                while True:
                    it = qout.get()
                    if it is None:
                        break
                    ev, pin, n = it
                    try:
                        ev.synchronize()
                        if not werr:
                            enc.stdin.write(pin[:n].numpy().tobytes())
                    except Exception as e:  # noqa: BLE001
                        werr.append(e)
                    finally:
                        pins.put(pin)

            wt = threading.Thread(target=writer); wt.start()
            try:
                with torch.inference_mode():
                    for i in range(0, T, B):
                        x = images[i:i + B]
                        t = x.to("cuda", non_blocking=True).permute(0, 3, 1, 2).half()
                        y = model(t)
                        y = F.interpolate(y.float(), size=(height, width), mode="bicubic", antialias=True)
                        ob = y.clamp_(0, 1).mul_(255).round_().byte().permute(0, 2, 3, 1).contiguous()
                        pin = pins.get()
                        done = torch.cuda.current_stream().record_event()
                        with torch.cuda.stream(stream):
                            stream.wait_event(done)
                            pin[: len(x)].copy_(ob, non_blocking=True)
                            ob.record_stream(stream)
                            ev = stream.record_event()
                        qout.put((ev, pin, len(x)))
                        if werr:
                            break
            finally:
                qout.put(None)
                wt.join()
        finally:
            try:
                enc.stdin.close()
            except OSError as e:
                werr.append(e)
            enc.wait()
            try:
                os.remove(wav)
            except OSError:
                pass
        if werr:
            raise RuntimeError(f"OLVUpscaleVideoSave writer failed: {werr[0]!r}")
        if enc.returncode:
            raise RuntimeError(f"OLVUpscaleVideoSave: ffmpeg exited with {enc.returncode}")
        return {"ui": {"images": [{"filename": file, "subfolder": subfolder, "type": "output"}], "animated": (True,)}}


NODE_CLASS_MAPPINGS = {"OLVUpscaleVideoSave": OLVUpscaleVideoSave}
NODE_DISPLAY_NAME_MAPPINGS = {"OLVUpscaleVideoSave": "OLV x2 upscale + save video (in graph)"}
