"""Google sign-in for the subdomain, through Cloudflare Access.

The public listener (--public-port) is what the Cloudflare tunnel publishes. Cloudflare
Access signs people in with Google before a request reaches this server and adds a token
(an RS256 JWT) to it. Every request on the public port must carry one that verifies against
the team's published keys, for this Access application, unexpired. That check is what stops
the other accounts on this server, which can reach 127.0.0.1 too, from going around Google.
The token's email is who the request is from: each person sees only their own transcripts.

Set in /etc/voice-to-text/env by server/set-google-access.sh: CF_ACCESS_TEAM_DOMAIN (e.g.
sphere.cloudflareaccess.com), CF_ACCESS_AUD (the application's AUD tag) and, optionally,
VOICE_ALLOWED_EMAILS (comma-separated addresses or "@domain"), a second check behind
Cloudflare's own policy. Without the first two the public port answers 503 to everything.

Requests on the root-only Unix socket (the SSH tunnel) need no sign-in and see every transcript.
"""

import asyncio
import html
import logging
import os
from urllib.parse import urlsplit

import jwt
from aiohttp import web

log = logging.getLogger("voice_to_text")


class SiteAuth:
    def __init__(self, public_port: int | None):
        self.public_port = public_port
        team = os.environ.get("CF_ACCESS_TEAM_DOMAIN", "").strip().removeprefix("https://").strip("/")
        self._aud = os.environ.get("CF_ACCESS_AUD", "").strip()
        self._issuer = f"https://{team}" if team else ""
        self._jwks = (jwt.PyJWKClient(f"{self._issuer}/cdn-cgi/access/certs", cache_keys=True, lifespan=3600)
                      if team and self._aud else None)
        self._allowed = [a.strip().lower() for a in os.environ.get("VOICE_ALLOWED_EMAILS", "").split(",") if a.strip()]

    @property
    def configured(self) -> bool:
        return self._jwks is not None

    def is_public(self, request: web.Request) -> bool:
        """Whether the request came in on the public port (a TCP socket name is (host, port); a Unix one a path)."""
        sockname = request.transport.get_extra_info("sockname") if request.transport else None
        return self.public_port is not None and isinstance(sockname, tuple) and sockname[1] == self.public_port

    def _verified_email(self, token: str) -> tuple[str | None, str]:
        """(email, "") for a valid token, else (None, why). Fetches the team's keys when needed (cached)."""
        try:
            key = self._jwks.get_signing_key_from_jwt(token).key
            claims = jwt.decode(token, key, algorithms=["RS256"], audience=self._aud, issuer=self._issuer,
                                options={"require": ["exp", "iat", "aud", "iss"]}, leeway=30)
        except jwt.PyJWTError as e:
            return None, f"invalid token ({type(e).__name__})"
        email = str(claims.get("email") or "").strip().lower()
        if not email:
            return None, "token without an email"
        if self._allowed and not any(email == a or (a.startswith("@") and email.endswith(a)) for a in self._allowed):
            return None, "not allowed"
        return email, ""

    async def who(self, request: web.Request) -> tuple[str | None, str]:
        """(email, "") of a signed-in request, else (None, why)."""
        token = request.headers.get("Cf-Access-Jwt-Assertion") or request.cookies.get("CF_Authorization")
        if not token:
            return None, "no token"
        return await asyncio.get_running_loop().run_in_executor(None, self._verified_email, token)


AUTH = web.AppKey("auth", SiteAuth)
OWNER = "owner"  # request[OWNER]: whose transcripts it sees and saves; None (the SSH tunnel) is everyone's


def _same_origin(request: web.Request) -> bool:
    origin = request.headers.get("Origin")
    return origin is None or urlsplit(origin).netloc == request.host


_REFUSED = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Voice to Text</title>
<style>
  :root { --bg: #f4f4f6; --surface: #fff; --text: #17171c; --muted: #6e6e78; --line: #e6e6eb; --accent: #4f46e5; color-scheme: light; }
  @media (prefers-color-scheme: dark) { :root { --bg: #0c0c10; --surface: #15151b; --text: #f2f2f5; --muted: #a0a0ab; --line: #272730; --accent: #8b8cff; color-scheme: dark; } }
  body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: var(--bg); color: var(--text);
         font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif; padding: 16px; }
  .box { max-width: 380px; background: var(--surface); border: 1px solid var(--line); border-radius: 18px; padding: 28px; display: grid; gap: 14px; }
  h1 { margin: 0; font-size: 20px; } p { margin: 0; color: var(--muted); }
  a { font-weight: 600; padding: 10px; border-radius: 10px; background: var(--accent); color: #fff; text-align: center; text-decoration: none; }
</style></head>
<body><div class="box"><h1>Voice to Text</h1><p>__TEXT__</p><a href="/cdn-cgi/access/logout">Sign in with a different Google account</a></div></body></html>
"""


def _refused(why: str) -> web.Response:
    text = ("Your Google account isn't on the list for this site. Ask its owner to add you."
            if why == "not allowed" else "This site signs people in with Google. Open it from its address to sign in.")
    return web.Response(text=_REFUSED.replace("__TEXT__", html.escape(text)), content_type="text/html",
                        status=403 if why == "not allowed" else 401, headers={"Cache-Control": "no-store"})


@web.middleware
async def guard(request: web.Request, handler):
    auth = request.app[AUTH]
    if not auth.is_public(request):
        request[OWNER] = None
        return await handler(request)
    if not auth.configured:
        return web.Response(status=503, text="Google sign-in isn't set up yet. On the server, run: "
                                             "bash /root/voice_to_text/server/set-google-access.sh\n")
    email, why = await auth.who(request)
    if email is None:
        if why != "no token":
            client = request.headers.get("Cf-Connecting-Ip") or request.remote or "?"
            log.warning("public: refused %s %s from %s: %s", request.method, request.path, client, why)
        if request.method == "GET" and request.path in ("/", "/index.html"):
            return _refused(why)
        return web.json_response({"error": "sign in with Google first"}, status=403 if why == "not allowed" else 401)
    # Another website can't make the browser act here: the WebSocket and every change must come from this site.
    if (request.method not in ("GET", "HEAD") or request.path == "/live") and not _same_origin(request):
        return web.json_response({"error": "wrong origin"}, status=403)
    request[OWNER] = email
    return await handler(request)


async def security_headers(request: web.Request, response: web.StreamResponse) -> None:
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")


async def logout(request: web.Request) -> web.Response:
    raise web.HTTPFound("/cdn-cgi/access/logout")  # Cloudflare ends the Google sign-in
