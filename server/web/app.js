// The site: record a dictation or a meeting in the browser, stream it to the server, show the text,
// the summary and the history.
"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  status: $("status"), statusText: $("status-text"), recorder: $("recorder"), record: $("record"),
  modeButtons: [$("mode-dictation"), $("mode-meeting")],
  timer: $("timer"), wave: $("wave"), speakers: $("speakers"), spkYou: $("spk-you"), spkMeeting: $("spk-meeting"),
  hint: $("hint"), file: $("file"),
  transcriptCard: $("transcript-card"), liveBadge: $("live-badge"), transcript: $("transcript"),
  final: $("final"), partial: $("partial"), transcriptMeta: $("transcript-meta"), deleteShown: $("delete-shown"),
  summaryCard: $("summary-card"), summaryModel: $("summary-model"), summary: $("summary"), summaryMeta: $("summary-meta"),
  history: $("history"), historyEmpty: $("history-empty"), historyCount: $("history-count"), search: $("search"),
  toast: $("toast"),
};

let server = null; // /health
let mode = loadMode(); // "dictation" or "meeting"
let rec = null; // the recording in progress
let shownId = null; // the transcript the cards show, so a late summary of an older one stays out
let shownText = ""; // what Copy copies: the transcript as saved
let transcripts = []; // the history, newest first
let levels = []; // recent loudness, for the waveform

const HINTS = {
  dictation: "Press the button and start talking",
  meeting: "Press the button, pick the meeting's tab and tick “Share tab audio”",
};

// --- helpers ---------------------------------------------------------------

const ICON = {
  clock: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="8"/><path d="M12 8v4l2.5 2"/></svg>',
  bolt: '<svg viewBox="0 0 24 24"><path d="M13 3 5 13h6l-1 8 8-10h-6z"/></svg>',
  mic: '<svg viewBox="0 0 24 24"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M6 11a6 6 0 0 0 12 0M12 17v4"/></svg>',
  people: '<svg viewBox="0 0 24 24"><circle cx="9" cy="8" r="3"/><circle cx="17" cy="9" r="2.5"/><path d="M3 19c0-3 2.7-5 6-5s6 2 6 5M15 14.5c3 0 6 1.6 6 4.5"/></svg>',
  file: '<svg viewBox="0 0 24 24"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/></svg>',
  cal: '<svg viewBox="0 0 24 24"><rect x="4" y="5" width="16" height="15" rx="2"/><path d="M8 3v4M16 3v4M4 10h16"/></svg>',
  warn: '<svg viewBox="0 0 24 24"><path d="M12 4 2.5 20h19z"/><path d="M12 10v4M12 17.5v.01"/></svg>',
};
const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const secs = (s) => (s < 10 ? s.toFixed(1) : Math.round(s)) + " s";
const clock = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const modelName = (m) => ({ "parakeet-tdt-0.6b-v2": "Parakeet v2", "parakeet-tdt-0.6b-v3": "Parakeet v3", "large-v3-turbo": "Whisper turbo" })[m] || m;
const summaryName = (m) => (m || "").replace(/^claude-/, "Claude ").replace(/-(\d+)-(\d+)$/, " $1.$2").replace(/\b[a-z]/g, (c) => c.toUpperCase());

function loadMode() {
  try {
    return localStorage.getItem("mode") === "meeting" ? "meeting" : "dictation";
  } catch {
    return "dictation";
  }
}

function when(t) {
  const d = new Date(t * 1000);
  const now = new Date();
  const time = d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
  const days = Math.round((new Date(now.getFullYear(), now.getMonth(), now.getDate()) - new Date(d.getFullYear(), d.getMonth(), d.getDate())) / 86400000);
  if (days === 0) return `Today, ${time}`;
  if (days === 1) return `Yesterday, ${time}`;
  return `${d.toLocaleDateString(undefined, { month: "short", day: "numeric" })}, ${time}`;
}

function toast(message) {
  els.toast.textContent = message;
  els.toast.classList.add("show");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => els.toast.classList.remove("show"), 2200);
}

async function copy(text, what) {
  try {
    await navigator.clipboard.writeText(text);
    toast(`${what} copied`);
  } catch {
    toast("Couldn't copy");
  }
}

function setMeta(el, pieces, warn = false) {
  el.className = "meta" + (warn ? " warn" : "");
  el.innerHTML = pieces.filter(Boolean).map(([icon, text]) => `<span>${ICON[icon] || ""}${escapeHtml(text)}</span>`).join("");
}

function setHint(text, warn = false) {
  els.hint.textContent = text;
  els.hint.className = "hint" + (warn ? " warn" : "");
}

// --- server state ----------------------------------------------------------

async function loadHealth() {
  try {
    server = await (await fetch("/health")).json();
    const summaries = server.summaries ? `${summaryName(server.summary_model)} summaries` : "summaries off";
    els.status.className = "status ok";
    els.statusText.textContent = `${modelName(server.model)} · ${summaries}`;
    $("signout").hidden = !server.public; // signed in with Google on the subdomain
    $("user").textContent = server.user || "";
  } catch {
    els.status.className = "status bad";
    els.statusText.textContent = "Can't reach the server. Is the SSH tunnel open?";
  }
}

// --- dictation or meeting --------------------------------------------------

function setMode(next) {
  mode = next;
  try { localStorage.setItem("mode", mode); } catch { /* a private window: just not remembered */ }
  els.modeButtons.forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === mode)));
  els.speakers.hidden = mode !== "meeting";
  if (!rec) setHint(HINTS[mode]);
}
els.modeButtons.forEach((b) => b.addEventListener("click", () => { if (!rec) setMode(b.dataset.mode); }));

// --- the button and the waveform ------------------------------------------

function setButton(state) {
  const icon = { idle: "mic", recording: "stop", busy: "wait" }[state];
  els.record.innerHTML = `<span class="icon ${icon}" aria-hidden="true"></span>`;
  els.recorder.classList.toggle("is-recording", state === "recording");
  els.record.disabled = state === "busy";
  els.modeButtons.forEach((b) => (b.disabled = state !== "idle"));
  els.record.setAttribute("aria-label", state === "recording" ? "Stop recording" : "Start recording");
  if (state !== "recording") showLevels([0, 0]);
}

function showLevels(perChannel) {
  els.record.style.setProperty("--level", Math.max(...perChannel));
  els.spkYou.style.setProperty("--l", perChannel[0] || 0);
  els.spkMeeting.style.setProperty("--l", perChannel[1] || 0);
}

function drawWave() {
  const canvas = els.wave;
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w) return;
  if (canvas.width !== Math.round(w * dpr)) {
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const css = getComputedStyle(document.documentElement);
  const active = els.recorder.classList.contains("is-recording");
  ctx.fillStyle = css.getPropertyValue(active ? "--record" : "--line").trim();
  const gap = 3, bars = Math.max(16, Math.floor((w + gap) / 7));
  const bw = (w - gap * (bars - 1)) / bars;
  for (let i = 0; i < bars; i++) {
    const v = levels[levels.length - bars + i] || 0;
    const bh = Math.max(4, v * h);
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(i * (bw + gap), (h - bh) / 2, bw, bh, bw / 2);
    else ctx.rect(i * (bw + gap), (h - bh) / 2, bw, bh);
    ctx.fill();
  }
}
window.addEventListener("resize", drawWave);

// --- the transcript --------------------------------------------------------

// A meeting's saved text: one "[m:ss] You: ..." or "[m:ss] Meeting: ..." line per turn.
function parseTurns(text) {
  return text.split("\n").map((line) => {
    const m = line.match(/^\[(\d+:\d\d)\]\s+(You|Meeting):\s?(.*)$/);
    return m ? { at: m[1], who: m[2], text: m[3] } : { at: "", who: "", text: line };
  }).filter((t) => t.text);
}

function turnHtml(turn, partial = false) {
  const kind = turn.who === "You" ? "you" : "meeting";
  return `<div class="turn ${kind}${partial ? " partial" : ""}">
    <span class="who">${escapeHtml(turn.who || "")}<span class="at">${escapeHtml(turn.at || "")}</span></span>
    <span class="said">${escapeHtml(turn.text)}</span></div>`;
}

// Live turns: segments come in as their parts finish, a little out of order across the two streams.
function renderLiveMeeting(r) {
  const sorted = [...r.segments].sort((a, b) => a.start - b.start);
  const turns = [];
  for (const s of sorted) {
    const last = turns[turns.length - 1];
    if (last && last.who === s.speaker) last.text += " " + s.text;
    else turns.push({ who: s.speaker, at: clock(s.start), text: s.text });
  }
  els.final.innerHTML = turns.map((t) => turnHtml(t)).join("");
  els.partial.innerHTML = Object.entries(r.partials).filter(([, text]) => text)
    .map(([who, text]) => turnHtml({ who, at: "", text }, true)).join("");
}

function showText(text, textMode) {
  shownText = text;
  els.transcript.classList.toggle("meeting", textMode === "meeting");
  els.partial.textContent = "";
  if (textMode === "meeting") els.final.innerHTML = parseTurns(text).map((t) => turnHtml(t)).join("");
  else els.final.textContent = text;
}

// --- recording -------------------------------------------------------------

async function start() {
  const meeting = mode === "meeting";
  setButton("busy");
  shownId = null;
  markSelected();
  levels = [];
  drawWave();
  els.timer.textContent = "0:00";
  els.transcriptCard.hidden = false;
  els.transcriptCard.classList.add("recording");
  els.transcript.classList.toggle("meeting", meeting);
  els.liveBadge.hidden = false;
  els.deleteShown.hidden = true;
  els.final.textContent = "";
  els.partial.textContent = "";
  shownText = "";
  setMeta(els.transcriptMeta, []);
  els.summaryCard.hidden = true;

  const fail = (message, streams = []) => {
    streams.forEach((s) => s && s.getTracks().forEach((t) => t.stop()));
    setButton("idle");
    resetLive();
    els.transcriptCard.hidden = true;
    setHint(message, true);
  };

  // A meeting: the tab first, while the click still counts as the user's gesture.
  let display = null;
  if (meeting) {
    if (!navigator.mediaDevices.getDisplayMedia) return fail("This browser can't share a tab's audio. Use Chrome or Edge on a computer.");
    setHint("Pick the meeting's tab, and tick “Share tab audio”…");
    try {
      display = await navigator.mediaDevices.getDisplayMedia({
        video: true, // browsers won't share audio alone
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
        systemAudio: "include",
        selfBrowserSurface: "exclude",
      });
    } catch {
      return fail("Sharing was cancelled. Press the button again and pick the meeting's tab.");
    }
    if (!display.getAudioTracks().length) {
      return fail("No audio came with that share. Share a browser tab and tick “Share tab audio” (Chrome or Edge).", [display]);
    }
  }

  setHint("Connecting…");
  let mic;
  try {
    mic = await navigator.mediaDevices.getUserMedia({
      // In a meeting, echo cancellation keeps some of the speakers' sound out of the microphone.
      audio: { channelCount: 1, echoCancellation: meeting, noiseSuppression: false, autoGainControl: true },
    });
  } catch {
    return fail("No microphone. Allow it for this page in the browser's site settings, then try again.", [display]);
  }

  // 16 kHz is what the model wants; a browser that can't record at that rate sends its own and the server resamples.
  const channels = meeting ? 2 : 1;
  let ctx, sources;
  const connect = (context) => [context.createMediaStreamSource(mic), ...(display ? [context.createMediaStreamSource(display)] : [])];
  try {
    ctx = new AudioContext({ sampleRate: 16000 });
    sources = connect(ctx);
  } catch {
    if (ctx) ctx.close();
    ctx = new AudioContext();
    sources = connect(ctx);
  }
  await ctx.audioWorklet.addModule("/static/worklet.js?v=5");
  const node = new AudioWorkletNode(ctx, "capture", {
    channelCount: channels, channelCountMode: "explicit", processorOptions: { channels },
  });
  let merger = null;
  if (meeting) { // the microphone on channel 0, the tab on channel 1
    merger = ctx.createChannelMerger(2);
    sources[0].connect(merger, 0, 0);
    sources[1].connect(merger, 0, 1);
    merger.connect(node);
  } else {
    sources[0].connect(node);
  }
  node.connect(ctx.destination); // outputs silence; keeps the worklet running

  const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/live`);
  ws.binaryType = "arraybuffer";
  const r = {
    ws, mic, display, ctx, node, sources, merger, meeting,
    started: 0, timer: 0, segments: [], partials: {}, stopping: false, pending: [],
  };
  rec = r;
  if (display) display.getAudioTracks()[0].addEventListener("ended", () => { if (rec === r) stop(); }); // "Stop sharing"

  ws.onopen = () => ws.send(JSON.stringify({ type: "start", rate: ctx.sampleRate, mode }));
  ws.onmessage = (event) => onMessage(r, JSON.parse(event.data));
  ws.onclose = () => {
    if (rec && rec.ws === ws) { // closed before "done": while recording, or while finishing
      if (!rec.stopping) stopAudio();
      rec = null;
      setButton("idle");
      resetLive();
      setHint("The connection to the server dropped. What was said so far is saved.", true);
      loadHistory();
    }
  };
  ws.onerror = () => {
    els.status.className = "status bad";
    els.statusText.textContent = "Can't reach the server. Is the SSH tunnel open?";
  };

  node.port.onmessage = (event) => {
    const blocks = event.data; // one Float32Array per channel, the same moment
    const n = blocks[0].length;
    const pcm = new Int16Array(n * blocks.length); // planar: the microphone, then the tab
    const perChannel = blocks.map((block, c) => {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const v = Math.max(-1, Math.min(1, block[i]));
        sum += v * v;
        pcm[c * n + i] = v < 0 ? v * 0x8000 : v * 0x7fff;
      }
      return Math.min(1, Math.sqrt(Math.sqrt(sum / n)) * 1.7);
    });
    levels.push(Math.max(...perChannel));
    if (levels.length > 400) levels = levels.slice(-200);
    showLevels(perChannel);
    drawWave();
    if (r.stopping) return;
    // Until the server says it's ready, keep the audio: people start talking as they press the button.
    if (ws.readyState === WebSocket.OPEN && r.started) ws.send(pcm.buffer);
    else r.pending.push(pcm.buffer);
  };
}

function onMessage(r, msg) {
  if (msg.type === "summary") {
    if (msg.id === shownId) showSummary(msg);
    loadHistory();
    r.ws.close();
    return;
  }
  if (rec !== r) return;
  if (msg.type === "started") {
    r.pending.forEach((buffer) => r.ws.send(buffer)); // what was said while connecting
    r.pending = [];
    r.started = performance.now();
    r.timer = setInterval(() => (els.timer.textContent = clock((performance.now() - r.started) / 1000)), 250);
    setButton("recording");
    setHint(r.meeting
      ? "Recording the meeting. Headphones keep the speakers out of your microphone."
      : "Listening. Each sentence settles when you pause.");
  } else if (msg.type === "partial") {
    if (r.meeting) {
      r.partials[msg.speaker] = msg.text;
      renderLiveMeeting(r);
    } else {
      els.partial.textContent = (r.segments.length ? " " : "") + msg.text;
    }
  } else if (msg.type === "segment") {
    r.segments.push(msg);
    if (r.meeting) {
      r.partials[msg.speaker] = "";
      renderLiveMeeting(r);
    } else {
      els.final.textContent = r.segments.map((s) => s.text).join(" ");
      els.partial.textContent = "";
    }
  } else if (msg.type === "revise") { // part of that "You" segment was the meeting, heard through the speakers
    const existing = r.segments.find((s) => s.speaker === msg.speaker && s.start === msg.start);
    if (existing) existing.text = msg.text;
    else if (msg.text) r.segments.push({ speaker: msg.speaker, start: msg.start, text: msg.text });
    r.segments = r.segments.filter((s) => s.text);
    renderLiveMeeting(r);
  } else if (msg.type === "done") {
    rec = null; // a new recording can start while this one's summary is written
    shownId = msg.id;
    resetLive();
    setButton("idle");
    setHint(HINTS[mode]);
    const warnings = [];
    if (msg.silent_channels.includes("Meeting")) warnings.push(["warn", "No sound came from the shared tab. Was “Share tab audio” ticked?"]);
    if (msg.silent_channels.includes("You") || msg.silent_channels.includes("microphone")) warnings.push(["warn", "The microphone sent only silence. Check the browser's microphone access."]);
    if (!msg.text) {
      showText("", msg.mode);
      setMeta(els.transcriptMeta, warnings.length ? warnings : [["warn", "No speech heard."]], true);
    } else {
      showText(msg.text, msg.mode);
      els.deleteShown.hidden = false;
      setMeta(els.transcriptMeta, [
        ["clock", clock(msg.audio_s)], ["bolt", `ready ${secs(msg.seconds)} after you stopped`],
        [msg.mode === "meeting" ? "people" : "mic", `saved as #${msg.id}`], ...warnings,
      ], warnings.length > 0);
      showSummary(msg);
    }
    loadHistory();
    if (msg.summary_status !== "pending") r.ws.close();
  } else if (msg.type === "error") {
    setHint(msg.error, true);
  }
}

function stop() {
  if (!rec || rec.stopping) return;
  rec.stopping = true;
  stopAudio();
  setButton("busy");
  setHint("Finishing the last sentence…");
  rec.ws.send(JSON.stringify({ type: "stop" }));
}

function stopAudio() {
  clearInterval(rec.timer);
  [rec.mic, rec.display].forEach((s) => s && s.getTracks().forEach((t) => t.stop()));
  rec.sources.forEach((s) => s.disconnect());
  if (rec.merger) rec.merger.disconnect();
  rec.node.disconnect();
  rec.ctx.close();
  levels = [];
  showLevels([0, 0]);
  drawWave();
}

function resetLive() {
  els.transcriptCard.classList.remove("recording");
  els.liveBadge.hidden = true;
  els.partial.textContent = "";
}

els.record.addEventListener("click", () => (rec ? stop() : start()));
document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat || e.target.closest("input, textarea, button, label")) return;
  e.preventDefault();
  if (!els.record.disabled) els.record.click();
});

// --- showing a saved transcript and its summary ---------------------------

function show(t) {
  if (rec) return; // not while recording
  shownId = t.id;
  markSelected();
  els.transcriptCard.hidden = false;
  resetLive();
  showText(t.text, t.mode);
  els.deleteShown.hidden = false;
  setMeta(els.transcriptMeta, [
    ["cal", when(t.created_at)],
    ["clock", clock(t.audio_s)],
    t.source === "live"
      ? [t.mode === "meeting" ? "people" : "mic", t.mode === "meeting" ? "meeting, recorded live" : "recorded live"]
      : ["file", `uploaded, transcribed in ${secs(t.seconds)}`],
    ["mic", `#${t.id}`],
  ]);
  showSummary(t);
  if (window.matchMedia("(max-width: 880px)").matches) els.transcriptCard.scrollIntoView({ behavior: "smooth" });
}

function renderSummary(text) {
  // The prompts ask for sentences, then "Key points:", "Decisions:" and "Action items:" lines with "- " items.
  const TITLES = { points: "Key points", decisions: "Decisions", actions: "Action items" };
  let html = "", kind = "points", inList = false;
  const close = () => { if (inList) { html += "</ul>"; inList = false; } };
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    const head = line.match(/^(key points|decisions|action items)\s*:?$/i);
    if (head) {
      close();
      kind = /action/i.test(head[1]) ? "actions" : /decision/i.test(head[1]) ? "decisions" : "points";
      html += `<h3>${TITLES[kind]}</h3>`;
      continue;
    }
    const item = line.match(/^[-•*]\s+(.*)$/);
    if (item) {
      if (!inList) { html += `<ul class="${kind}">`; inList = true; }
      html += `<li>${escapeHtml(item[1])}</li>`;
      continue;
    }
    close();
    html += `<p>${escapeHtml(line)}</p>`;
  }
  close();
  return html;
}

function showSummary(t) {
  els.summaryCard.hidden = false;
  els.summaryModel.hidden = true;
  els.summary.dataset.text = "";
  setMeta(els.summaryMeta, []);
  if (t.summary_status === "pending") {
    els.summary.innerHTML = '<div class="skeleton" aria-label="Summarizing"><i></i><i></i><i></i></div>';
    setMeta(els.summaryMeta, [["bolt", "Summarizing with Claude…"]]);
  } else if (t.summary_status === "done") {
    els.summary.innerHTML = renderSummary(t.summary);
    els.summary.dataset.text = t.summary;
    if (t.summary_model) {
      els.summaryModel.textContent = summaryName(t.summary_model);
      els.summaryModel.hidden = false;
    }
  } else if (t.summary_status === "off") {
    els.summary.innerHTML = '<p class="notice">Summaries are off: the server has no Anthropic API key yet.</p>';
  } else if (t.summary_status === "error") {
    els.summary.innerHTML = `<p class="notice">${escapeHtml(t.summary_error || "The summary failed.")}</p>`;
  } else {
    els.summaryCard.hidden = true;
  }
}

$("copy-transcript").addEventListener("click", () => copy(shownText || els.final.textContent, "Transcript"));
$("copy-summary").addEventListener("click", () => copy(els.summary.dataset.text || els.summary.textContent, "Summary"));
els.deleteShown.addEventListener("click", async () => {
  if (!shownId || !confirm("Delete this transcript and its summary from the server?")) return;
  await fetch(`/api/transcripts/${shownId}`, { method: "DELETE" });
  shownId = null;
  els.transcriptCard.hidden = true;
  els.summaryCard.hidden = true;
  toast("Deleted");
  loadHistory();
});

// --- uploading a file ------------------------------------------------------

async function upload(file) {
  if (!file || rec) return;
  setButton("busy");
  shownId = null;
  markSelected();
  els.transcriptCard.hidden = false;
  resetLive();
  showText("", "dictation");
  els.deleteShown.hidden = true;
  els.summaryCard.hidden = true;
  setMeta(els.transcriptMeta, [["file", `Transcribing ${file.name}…`]]);
  try {
    // A file in meeting mode is a recorded meeting: one stream, so no speaker labels, but a meeting summary.
    const response = await fetch(`/transcribe${mode === "meeting" ? "?mode=meeting" : ""}`, { method: "POST", body: file });
    const t = await response.json();
    if (!response.ok) throw new Error(t.error || response.statusText);
    shownId = t.id;
    showText(t.text, "dictation");
    if (t.text) {
      els.deleteShown.hidden = false;
      setMeta(els.transcriptMeta, [["clock", clock(t.audio_s)], ["bolt", `transcribed in ${secs(t.seconds)}`], ["file", `saved as #${t.id}`]]);
      showSummary(t);
      loadHistory();
      if (t.summary_status === "pending") pollSummary(t.id);
    } else {
      setMeta(els.transcriptMeta, [["warn", "No speech heard in that file."]], true);
    }
  } catch (e) {
    setMeta(els.transcriptMeta, [["warn", `Couldn't transcribe ${file.name}: ${e.message}`]], true);
  } finally {
    setButton("idle");
  }
}

els.file.addEventListener("change", () => {
  const file = els.file.files[0];
  els.file.value = "";
  upload(file);
});
let dragDepth = 0;
els.recorder.addEventListener("dragenter", (e) => { e.preventDefault(); dragDepth++; els.recorder.classList.add("dragging"); });
els.recorder.addEventListener("dragover", (e) => e.preventDefault());
els.recorder.addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; els.recorder.classList.remove("dragging"); } });
els.recorder.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  els.recorder.classList.remove("dragging");
  upload(e.dataTransfer.files[0]);
});

async function pollSummary(id) {
  for (let i = 0; i < 90; i++) {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    const t = await (await fetch(`/api/transcripts/${id}`)).json();
    if (t.summary_status !== "pending") {
      if (t.id === shownId) showSummary(t);
      loadHistory();
      return;
    }
  }
}

// --- history ---------------------------------------------------------------

async function loadHistory() {
  try {
    transcripts = (await (await fetch("/api/transcripts")).json()).transcripts;
  } catch {
    return;
  }
  renderHistory();
}

function preview(t) {
  return t.mode === "meeting" ? parseTurns(t.text).map((turn) => (turn.who ? `${turn.who}: ` : "") + turn.text).join(" · ") : t.text;
}

function renderHistory() {
  const query = els.search.value.trim().toLowerCase();
  const list = query
    ? transcripts.filter((t) => t.text.toLowerCase().includes(query) || (t.summary || "").toLowerCase().includes(query))
    : transcripts;
  els.historyCount.textContent = transcripts.length ? String(transcripts.length) : "";
  els.historyEmpty.hidden = list.length > 0;
  els.historyEmpty.textContent = transcripts.length ? "Nothing matches that search." : "Your recordings will show up here.";
  els.history.innerHTML = list.map((t) => `
    <li><button type="button" data-id="${t.id}">
      <div class="h-top">
        <span>${escapeHtml(when(t.created_at))}</span>
        <span class="h-tags">
          <span class="h-tag">${t.mode === "meeting" ? "meeting" : t.source === "live" ? "live" : "file"}</span>
          <span class="h-tag">${clock(t.audio_s)}</span>
          ${t.summary_status === "done" ? '<span class="h-tag summary-on">summary</span>' : ""}
        </span>
      </div>
      <div class="h-text">${escapeHtml(preview(t))}</div>
    </button></li>`).join("");
  els.history.querySelectorAll("button[data-id]").forEach((b) =>
    b.addEventListener("click", () => show(transcripts.find((t) => String(t.id) === b.dataset.id))));
  markSelected();
}

function markSelected() {
  els.history.querySelectorAll("button[data-id]").forEach((b) => b.classList.toggle("selected", b.dataset.id === String(shownId)));
}

els.search.addEventListener("input", renderHistory);

setMode(mode);
setButton("idle");
drawWave();
loadHealth();
loadHistory();
