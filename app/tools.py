"""工具层：工位的真实产出动作——文件落盘、小红书文字卡图、TTS 配音、ffmpeg 合成视频。

产物统一落盘到 outputs/<task_id>/，由 main.py 以 /outputs 静态托管，结果面板直接给链接。
"""
from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent / "outputs"
FONT_TITLE = "/System/Library/Fonts/STHeiti Medium.ttc"
FONT_BODY = "/System/Library/Fonts/Hiragino Sans GB.ttc"


def task_dir(task_id: str) -> Path:
    d = WORKSPACE / task_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def rel(path: Path) -> str:
    return f"outputs/{path.parent.name}/{path.name}"


def write_file(task_id: str, name: str, content: str) -> str:
    p = task_dir(task_id) / name
    p.write_text(content, encoding="utf-8")
    return rel(p)


# ---------------------------------------------------------------- 文字卡图（PIL）
def text_card(out: Path, title: str, lines: list[str], w: int = 1080, h: int = 1440,
              accent: tuple = (255, 79, 121), badge: str = "") -> Path:
    """小红书风竖版文字卡：渐变底 + 大标题 + 要点行 + 底部标签。"""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (w, h), (16, 18, 28))
    dr = ImageDraw.Draw(img)
    # 顶部渐变色块
    for y in range(int(h * .42)):
        k = y / (h * .42)
        dr.line([(0, y), (w, y)], fill=tuple(int(c * (1 - k) + 26 * k) for c in accent))
    dr.rectangle([0, int(h * .42), w, h], fill=(16, 18, 28))

    def font(size: int, bold: bool = False):
        return ImageFont.truetype(FONT_TITLE if bold else FONT_BODY, size, index=0)

    def wrap(text: str, f, max_w: int) -> list[str]:
        rows, cur = [], ""
        for ch in text:
            if dr.textlength(cur + ch, font=f) > max_w or ch == "\n":
                rows.append(cur)
                cur = "" if ch == "\n" else ch
            else:
                cur += ch
        if cur:
            rows.append(cur)
        return rows or [""]

    pad = 84
    y = int(h * .13)
    for row in wrap(title, font(96, True), w - pad * 2):
        dr.text((pad, y), row, font=font(96, True), fill=(255, 255, 255))
        y += 128
    if badge:
        y += 26
        tw = dr.textlength(badge, font=font(34, True))
        dr.rounded_rectangle([pad, y, pad + tw + 44, y + 62], 31, fill=accent)
        dr.text((pad + 22, y + 12), badge, font=font(34, True), fill=(255, 255, 255))
        y += 100
    y = max(y + 40, int(h * .5))
    for line in lines:
        for row in wrap(line, font(44), w - pad * 2):
            dr.text((pad, y), row, font=font(44), fill=(214, 224, 240))
            y += 66
        y += 22
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=92)
    return out


def _load_font(size: int, bold: bool = False):
    from PIL import ImageFont
    return ImageFont.truetype(FONT_TITLE if bold else FONT_BODY, size, index=0)


def _wrap(dr, text: str, f, max_w: int) -> list[str]:
    rows, cur = [], ""
    for ch in text:
        if dr.textlength(cur + ch, font=f) > max_w or ch == "\n":
            rows.append(cur)
            cur = "" if ch == "\n" else ch
        else:
            cur += ch
    if cur:
        rows.append(cur)
    return rows or [""]


def overlay_card(bg: Path, out: Path, title: str, lines: list[str], w: int, h: int,
                 accent: tuple = (255, 79, 121), badge: str = "") -> Path:
    """AI 生图作背景 + 底部渐变暗板叠标题/要点（小红书封面/分镜的真实排版）。"""
    from PIL import Image, ImageDraw
    img = Image.open(bg).convert("RGB")
    # 等比裁切到目标宽高
    rw, rh = img.size
    scale = max(w / rw, h / rh)
    img = img.resize((int(rw * scale) + 1, int(rh * scale) + 1))
    x0 = (img.width - w) // 2; y0 = (img.height - h) // 2
    img = img.crop((x0, y0, x0 + w, y0 + h))
    dr = ImageDraw.Draw(img, "RGBA")
    # 底部渐变暗板
    gh = int(h * .5)
    for y in range(gh):
        a = int(215 * (y / gh) ** 0.85)
        dr.line([(0, h - gh + y), (w, h - gh + y)], fill=(8, 10, 18, a))
    pad = 84
    y = h - gh + 40
    if badge:
        f = _load_font(34, True)
        tw = dr.textlength(badge, font=f)
        dr.rounded_rectangle([pad, y, pad + tw + 44, y + 60], 30, fill=accent + (255,))
        dr.text((pad + 22, y + 11), badge, font=f, fill=(255, 255, 255, 255))
        y += 92
    if title:
        for row in _wrap(dr, title, _load_font(84, True), w - pad * 2):
            dr.text((pad, y), row, font=_load_font(84, True), fill=(255, 255, 255, 255))
            y += 112
        y += 14
    for line in lines:
        for row in _wrap(dr, line, _load_font(42), w - pad * 2):
            dr.text((pad, y), row, font=_load_font(42), fill=(226, 234, 246, 255))
            y += 62
        y += 16
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, quality=92)
    return out


# ---------------------------------------------------------------- TTS（edge-tts，免费）
async def tts(text: str, out: Path, voice: str = "zh-CN-XiaoxiaoNeural") -> Path:
    import edge_tts
    out.parent.mkdir(parents=True, exist_ok=True)
    last = None
    for attempt in range(3):                       # 网络抖动重试
        try:
            await edge_tts.Communicate(text, voice, rate="+8%").save(str(out))
            if out.exists() and out.stat().st_size > 0:
                return out
        except Exception as e:
            last = e
        await asyncio.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"tts 失败: {last}")


def audio_dur(path: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return max(1.0, float(r.stdout.strip()))
    except ValueError:
        return 4.0


# ---------------------------------------------------------------- 视频合成（ffmpeg）
async def render_video(task_id: str, segments: list[dict], out_name: str = "video.mp4") -> dict:
    """segments: [{img: Path, audio: Path, text: str}] → 逐段合成 + 拼接 + srt 字幕。"""
    d = task_dir(task_id)
    clips, srt, t0 = [], [], 0.0
    for i, seg in enumerate(segments):
        seg_out = d / f"_seg{i}.mp4"
        audio = Path(seg["audio"])
        if not audio.exists() or audio.stat().st_size == 0:
            # 配音缺失兜底：按文本长度生成静音音轨，段时长仍成立
            est = max(2.0, len(seg.get("text", "")) / 4.5)
            silent = d / f"_sil{i}.mp3"
            await asyncio.to_thread(subprocess.run, [
                "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                "-i", "anullsrc=r=24000:cl=mono", "-t", f"{est:.2f}", str(silent)], capture_output=True)
            audio = silent
        dur = audio_dur(audio)
        await asyncio.to_thread(subprocess.run, [
            "ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(seg["img"]),
            "-i", str(seg["audio"]), "-c:v", "libx264", "-tune", "stillimage", "-r", "30",
            "-c:a", "aac", "-b:a", "128k", "-pix_fmt", "yuv420p", "-shortest",
            "-t", f"{dur:.2f}", str(seg_out)], capture_output=True)
        clips.append(seg_out)
        srt.append({"t0": t0, "t1": t0 + dur, "text": seg["text"]})
        t0 += dur
    lst = d / "_concat.txt"
    lst.write_text("".join(f"file '{c.name}'\n" for c in clips), encoding="utf-8")
    final = d / out_name
    await asyncio.to_thread(subprocess.run, [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
        "-c", "copy", str(final)], capture_output=True)
    srt_path = d / "字幕.srt"

    def fmt(t: float) -> str:
        ms = int(t * 1000)
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    srt_path.write_text("".join(f"{i + 1}\n{fmt(s['t0'])} --> {fmt(s['t1'])}\n{s['text']}\n\n"
                                for i, s in enumerate(srt)), encoding="utf-8")
    for c in clips:
        c.unlink(missing_ok=True)
    lst.unlink(missing_ok=True)
    return {"video": rel(final), "srt": rel(srt_path), "dur": round(t0, 1)}
