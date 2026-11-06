"""Meeting mode in headless Chromium: a fake meeting tab plays a clip, the fake microphone plays another."""
import base64, sys, time
from playwright.sync_api import sync_playwright
s, mic_wav, meeting_wav = sys.argv[1], sys.argv[2], sys.argv[3]
audio64 = base64.b64encode(open(meeting_wav, "rb").read()).decode()
errors = []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, chromium_sandbox=False, args=[
        "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={mic_wav}",
        "--auto-select-tab-capture-source-by-title=Fake Meeting", "--autoplay-policy=no-user-gesture-required",
    ])
    context = browser.new_context(viewport={"width": 1180, "height": 1000})
    # Stand in for the tab share: the meeting clip, played in the page and captured as a stream.
    context.add_init_script("""
      navigator.mediaDevices.getDisplayMedia = async () => {
        const a = new Audio("data:audio/wav;base64,%s"); a.loop = true; await a.play();
        const stream = a.captureStream();
        await new Promise(r => setTimeout(r, 300));
        return stream;
      };""" % audio64)
    site = context.new_page()
    site.on("console", lambda m: m.type in ("error", "warning") and errors.append(f"console {m.type}: {m.text}"))
    site.on("pageerror", lambda e: errors.append(f"page error: {e}"))
    site.goto("http://127.0.0.1:8096/")
    site.wait_for_selector("#status.ok")
    site.click("#mode-meeting")
    print("mode:", site.get_attribute("#mode-meeting", "aria-checked"), "| hint:", site.inner_text("#hint"), "| speaker dots shown:", site.is_visible("#speakers"))
    site.click("#record")
    try:
        site.wait_for_selector("#recorder.is-recording", timeout=15000)
    except Exception:
        print("did not start; hint:", site.inner_text("#hint"))
        print("\n".join(errors)); sys.exit(1)
    time.sleep(17)
    site.screenshot(path=f"{s}/ui-meeting-live.png")
    print("live rows:", site.locator("#final .turn").count(), "| interim rows:", site.locator("#partial .turn").count())
    site.click("#record")
    site.wait_for_function("document.getElementById('transcript-meta').textContent.includes('saved as')", timeout=30000)
    turns = site.eval_on_selector_all("#final .turn", "ts => ts.map(t => t.querySelector('.who').childNodes[0].textContent + ' @' + t.querySelector('.at').textContent + ': ' + t.querySelector('.said').textContent.slice(0, 50))")
    print("final turns:"); [print("  ", t) for t in turns]
    print("meta:", site.inner_text("#transcript-meta").replace("\n", " | "))
    print("history tag:", site.inner_text("#history button .h-tag"))
    site.screenshot(path=f"{s}/ui-meeting-done.png", full_page=True)
    browser.close()
print("\n".join(errors) or "no console errors")
