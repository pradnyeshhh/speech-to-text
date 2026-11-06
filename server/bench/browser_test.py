"""The real page in headless Chromium, with a fake microphone playing a WAV file."""
import sys, time
from playwright.sync_api import sync_playwright

s, wav = sys.argv[1], sys.argv[2]
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, chromium_sandbox=False, args=[
        "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={wav}", "--autoplay-policy=no-user-gesture-required",
    ])
    page = browser.new_page(viewport={"width": 900, "height": 1100})
    page.on("console", lambda m: m.type in ("error", "warning") and errors.append(f"console {m.type}: {m.text}"))
    page.on("pageerror", lambda e: errors.append(f"page error: {e}"))
    page.goto("http://127.0.0.1:8096/")
    page.wait_for_function("document.getElementById('status').textContent.startsWith('Connected')", timeout=10000)
    print("status:", page.inner_text("#status"))
    page.click("#record")
    page.wait_for_function("document.getElementById('record').classList.contains('recording')", timeout=10000)
    print("recording; hint:", page.inner_text("#hint"))
    time.sleep(9)
    page.screenshot(path=f"{s}/shot-recording.png")
    print("after 9 s, final:", repr(page.inner_text("#final")[:80]), "| partial:", repr(page.inner_text("#partial")[:80]))
    time.sleep(10)
    page.click("#record")  # stop
    page.wait_for_function("document.getElementById('transcript-meta').textContent.includes('saved as')", timeout=30000)
    print("meta:", page.inner_text("#transcript-meta"))
    print("final:", repr(page.inner_text("#final")[:160]))
    page.wait_for_function("!document.getElementById('summary').classList.contains('pending')", timeout=30000)
    print("summary meta:", page.inner_text("#summary-meta"))
    print("button free:", not page.is_disabled("#record"), "| history items:", page.locator("#history li").count())
    page.screenshot(path=f"{s}/shot-done.png", full_page=True)
    page.set_input_files("#file", "/root/voice_to_text/bench/data/clips/short.wav")
    page.wait_for_function("document.getElementById('transcript-meta').textContent.includes('transcribed in')", timeout=30000)
    print("upload meta:", page.inner_text("#transcript-meta"), "| text:", repr(page.inner_text("#final")))
    browser.close()
print("\n".join(errors) or "no console errors")
