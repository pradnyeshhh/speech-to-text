"""Google sign-in (auth.py) against a fake Cloudflare: our own RSA key, key list and tokens."""
import asyncio, json, os, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import aiohttp, jwt
from aiohttp import web
from cryptography.hazmat.primitives.asymmetric import rsa
sys.path.insert(0, "/root/voice_to_text/server")

TEAM, AUD = "test.cloudflareaccess.com", "test-aud-tag"
key, other_key = (rsa.generate_private_key(public_exponent=65537, key_size=2048) for _ in range(2))
jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key())); jwk.update(kid="k1", alg="RS256", use="sig")
class Certs(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"keys": [jwk]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
certs = ThreadingHTTPServer(("127.0.0.1", 0), Certs); threading.Thread(target=certs.serve_forever, daemon=True).start()

os.environ.update(CF_ACCESS_TEAM_DOMAIN=TEAM, CF_ACCESS_AUD=AUD, VOICE_ALLOWED_EMAILS="@sphere-software.com")
import auth, server
from store import Store

def token(email, *, k=key, aud=AUD, iss=f"https://{TEAM}", exp=3600, kid="k1"):
    now = int(time.time())
    return jwt.encode({"email": email, "aud": [aud], "iss": iss, "iat": now, "exp": now + exp, "type": "app"}, k,
                      algorithm="RS256", headers={"kid": kid})

class Stub:  # just enough of server.App for the routes under test
    def __init__(self, store):
        self.store = store
        self.transcriber = type("T", (), {"model_name": "test-model", "short_windows": None})()
        self.summarizer = type("S", (), {"enabled": False})()

async def main():
    tmp = Path(tempfile.mkdtemp())
    store = Store(tmp / "t.db")
    a = store.add(source="live", mode="dictation", owner="alice@sphere-software.com", model="m", audio_s=1, speech_s=1, seconds=0, text="alice's note", summary_status="none")
    b = store.add(source="live", mode="dictation", owner="bob@sphere-software.com", model="m", audio_s=1, speech_s=1, seconds=0, text="bob's note", summary_status="none")
    store.add(source="live", mode="dictation", owner=None, model="m", audio_s=1, speech_s=1, seconds=0, text="made over ssh", summary_status="none")
    site_auth = auth.SiteAuth(None)
    site_auth._jwks = jwt.PyJWKClient(f"http://127.0.0.1:{certs.server_port}/cdn-cgi/access/certs", cache_keys=True)
    app = server.make_app(Stub(store), site_auth)
    runner = web.AppRunner(app); await runner.setup()
    tcp = web.TCPSite(runner, "127.0.0.1", 0); await tcp.start()
    port = tcp._server.sockets[0].getsockname()[1]; site_auth.public_port = port
    sock = str(tmp / "s.sock"); await web.UnixSite(runner, sock).start()
    base = f"http://127.0.0.1:{port}"
    alice, bob = token("alice@sphere-software.com"), token("bob@sphere-software.com")
    H = lambda t, **extra: {"Cf-Access-Jwt-Assertion": t, **extra}
    async with aiohttp.ClientSession() as http:
        async def st(method, path, headers=None, cookies=None):
            async with http.request(method, base + path, headers=headers or {}, cookies=cookies, allow_redirects=False) as r:
                body = await r.text()
                return r.status, body
        checks = [
            ("no token: the page", await st("GET", "/"), 401),
            ("no token: the API", await st("GET", "/api/transcripts"), 401),
            ("no token: the app's script", await st("GET", "/static/app.js"), 401),
            ("alice: the page", await st("GET", "/", H(alice)), 200),
            ("alice via the CF_Authorization cookie", await st("GET", "/health", cookies={"CF_Authorization": alice}), 200),
            ("expired token", await st("GET", "/health", H(token("alice@sphere-software.com", exp=-120))), 401),
            ("token for another Access app", await st("GET", "/health", H(token("alice@sphere-software.com", aud="other-app"))), 401),
            ("token from another team", await st("GET", "/health", H(token("alice@sphere-software.com", iss="https://evil.cloudflareaccess.com"))), 401),
            ("forged with another key", await st("GET", "/health", H(token("alice@sphere-software.com", k=other_key))), 401),
            ("unsigned ('none') token", await st("GET", "/health", H(jwt.encode({"email": "alice@sphere-software.com", "aud": [AUD], "iss": f"https://{TEAM}", "iat": int(time.time()), "exp": int(time.time()) + 600}, None, algorithm="none", headers={"kid": "k1"}))), 401),
            ("gmail account, not on the list", await st("GET", "/", H(token("someone@gmail.com"))), 403),
            ("alice deletes bob's transcript", await st("DELETE", f"/api/transcripts/{b}", H(alice)), 404),
            ("alice reads bob's transcript", await st("GET", f"/api/transcripts/{b}", H(alice)), 404),
            ("alice deletes, from another website", await st("DELETE", f"/api/transcripts/{a}", H(alice, Origin="https://evil.example")), 403),
        ]
        ok = True
        for name, (status, body), want in checks:
            good = status == want; ok &= good
            print(f"{'ok ' if good else 'BAD'} {name:42} -> {status} (want {want})")
        _, health = await st("GET", "/health", H(alice)); print("    alice's /health user:", json.loads(health)["user"])
        _, listing = await st("GET", "/api/transcripts", H(alice)); print("    alice's history:", [t["text"] for t in json.loads(listing)["transcripts"]])
        _, listing = await st("GET", "/api/transcripts", H(bob)); print("    bob's history:  ", [t["text"] for t in json.loads(listing)["transcripts"]])
    async with aiohttp.ClientSession(connector=aiohttp.UnixConnector(path=sock)) as http:
        async with http.get("http://x/api/transcripts") as r:
            print("    over the SSH socket (no sign-in):", r.status, [t["text"] for t in (await r.json())["transcripts"]])
    print("ALL GOOD" if ok else "SOMETHING FAILED")
    await runner.cleanup()

asyncio.run(main())
