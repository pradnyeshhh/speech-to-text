"""The redesigned page in headless Chromium: a live recording, history, a summary, light/dark/phone screenshots."""
import json, sys, time
from playwright.sync_api import sync_playwright
s, wav = sys.argv[1], sys.argv[2]
haiku = json.load(open(f"{s}/haiku-summary.json"))
errors = []
args = ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", f"--use-file-for-fake-audio-capture={wav}"]
def watch(page):
    page.on("console", lambda m: m.type in ("error", "warning") and errors.append(f"console {m.type}: {m.text}"))
    page.on("pageerror", lambda e: errors.append(f"page error: {e}"))
def inject_summary(page):
    page.evaluate("t => { shownId = t.id; showSummary(t); }", {**haiku, "id": page.evaluate("shownId")})
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, chromium_sandbox=False, args=args)
    page = browser.new_page(viewport={"width": 1180, "height": 900}, color_scheme="light")
    watch(page)
    page.goto("http://127.0.0.1:8096/")
    page.wait_for_selector("#status.ok", timeout=10000)
    print("status:", page.inner_text("#status-text"))
    page.keyboard.press("Space")  # the shortcut
    page.wait_for_selector("#recorder.is-recording", timeout=10000)
    time.sleep(6)
    page.screenshot(path=f"{s}/ui-recording.png")
    print("while recording: final", repr(page.inner_text("#final")[:60]), "| partial", repr(page.inner_text("#partial")[:60]))
    time.sleep(4)
    page.click("#record")
    page.wait_for_function("document.getElementById('transcript-meta').textContent.includes('saved as')", timeout=30000)
    print("after stop:", page.inner_text("#transcript-meta").replace("\n", " | "))
    inject_summary(page)
    print("summary sections:", page.eval_on_selector_all("#summary h3", "hs => hs.map(h => h.textContent)"), "| items:", page.locator("#summary li").count())
    page.screenshot(path=f"{s}/ui-light.png", full_page=True)
    page.set_input_files("#file", "/root/voice_to_text/bench/data/clips/short.wav")
    page.wait_for_function("document.getElementById('transcript-meta').textContent.includes('transcribed in')", timeout=30000)
    print("upload:", page.inner_text("#final"), "| history items:", page.locator("#history button").count())
    page.fill("#search", "contrivance")
    print("search 'contrivance' ->", page.locator("#history button").count(), "item(s)")
    page.fill("#search", "")
    page.locator("#history button").nth(1).click()
    print("clicked history item 2, now showing:", repr(page.inner_text("#final")[:50]), "| selected:", page.locator("#history button.selected").count())
    dark = browser.new_page(viewport={"width": 1180, "height": 900}, color_scheme="dark")
    watch(dark)
    dark.goto("http://127.0.0.1:8096/"); dark.wait_for_selector("#history button", timeout=10000)
    dark.locator("#history button").nth(1).click(); inject_summary(dark)
    dark.screenshot(path=f"{s}/ui-dark.png", full_page=True)
    phone = browser.new_page(viewport={"width": 390, "height": 844}, color_scheme="light", device_scale_factor=2)
    watch(phone)
    phone.goto("http://127.0.0.1:8096/"); phone.wait_for_selector("#history button", timeout=10000)
    phone.screenshot(path=f"{s}/ui-phone.png", full_page=True)
    print("phone: page wider than screen?", phone.evaluate("document.documentElement.scrollWidth > window.innerWidth"))
    browser.close()
print("\n".join(errors) or "no console errors")
