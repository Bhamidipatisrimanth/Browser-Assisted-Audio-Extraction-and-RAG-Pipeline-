import subprocess
from pathlib import Path
import asyncio
import yt_dlp
from playwright.async_api import async_playwright

import torch
from faster_whisper import WhisperModel

AUDIO_DIR = Path("audio")
AUDIO_DIR.mkdir(exist_ok=True)

COOKIE_PATH = Path("cookies.txt")


async def save_youtube_cookies(video_url):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True, args=["--no-sandbox"])
        context = await browser.new_context()
        page = await context.new_page()

        await page.goto(video_url, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(5000)

        title = await page.title()
        print("opened page:", title)
        cookies = await context.cookies()

        with COOKIE_PATH.open("w", encoding="utf-8", newline="\n") as f:
            f.write("# Netscape HTTP Cookie File\n")
            for cookie in cookies:
                domain = cookie.get("domain", "")
                if not domain.startswith("."):
                    domain = "." + domain
                flag = "TRUE"
                path = cookie.get("path", "/")
                secure = "TRUE" if cookie.get("secure") else "FALSE"
                
                name = cookie.get("name", "")
                value = cookie.get("value", "")
                flag = "TRUE" if domain.startswith(".") else "FALSE"
                expiry = cookie.get("expires", 0)
                try:
                    expiry = int(expiry)
                except Exception:
                    expiry = 0
                if expiry < 0:
                    expiry = 0
                line = "\t".join([domain, flag, path, secure, str(expiry), name, value])

                f.write(line + "\n")
    await browser.close()
    print("cookies.txt created")


def download_youtube_audio(url, output_name="test_audio"):
    url = url.strip().strip('"').strip("'")
    final_path = AUDIO_DIR / f"{output_name}.mp3"

    if final_path.exists():
        print("Using existing audio:", final_path)
        return str(final_path)

    print("Creating fresh Youtube cookies....")
    asyncio.run(save_youtube_cookies(url))
    ydl_opts = {
        "format": "bestaudio/best",
        "cookiefile": str(COOKIE_PATH),
        "outtmpl": str(AUDIO_DIR / f"{output_name}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "quiet": False,
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print("Download complete:", final_path)
    return str(final_path)


def transcribe(audio_path, model_size="base"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print("Torch CUDA:", torch.cuda.is_available())
    print("Device:", device)
    print("Compute type:", compute_type)

    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
    )

    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=False,
    )

    print("Detected language:", info.language)
    print("Language probability:", round(info.language_probability, 3))

    transcript_parts = []
    
    segment_count = 0
    for segment in segments:
        segment_count +=1
        print(
            f"[{segment.start:2f}s -> {segment.end:.2f}s]"
        )
        print(segment.text)
        line = segment.text.strip()
        if line:
            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {line}")
            transcript_parts.append(line)

    print("Total segments:", segment_count)
    transcript = " ".join(transcript_parts)
    print("\nTranscript length:", len(transcript))
    return transcript


if __name__ == "__main__":
    url = input("Enter Youtube URL: ").strip().strip('"').strip("'")

    audio_path = download_youtube_audio(url)
    transcript = transcribe(audio_path, model_size="base")
    Path("transcript.txt").write_text(transcript, encoding="utf-8")

    print("\nSaved transcript to transcript.txt")
    print("\nPreview:")
    print(transcript[:1000])
