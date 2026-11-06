#!/bin/sh
# Turns on Google sign-in for the subdomain (Cloudflare Access), then restarts the server.
#
#   bash /root/voice_to_text/server/set-google-access.sh
#
# Asks for two values from the Cloudflare dashboard and who may sign in, and keeps them in
# /etc/voice-to-text/env (readable by root alone) beside the Anthropic API key, which this
# leaves as it is. See server/auth.py.
set -eu
umask 077
ENV=/etc/voice-to-text/env

echo "From Cloudflare Zero Trust:"
printf '  Team domain (Settings > Custom Pages, e.g. sphere.cloudflareaccess.com): '
read -r team
printf '  Application Audience (AUD) Tag (Access > Applications > your app > Edit > Additional settings): '
read -r aud
printf 'Who may sign in, as a second check behind Cloudflare (Enter for @sphere-software.com): '
read -r allowed

team=$(printf '%s' "$team" | sed -e 's#^https://##' -e 's#/*$##')
case "$team" in
  *.*) ;;
  *) team="$team.cloudflareaccess.com" ;;
esac
case "$aud" in
  ""|*" "*) echo "That AUD tag looks wrong (it's a long string without spaces). Nothing changed."; exit 1 ;;
esac
[ -n "$allowed" ] || allowed="@sphere-software.com"

mkdir -p /etc/voice-to-text
touch "$ENV"
grep -v -E '^(CF_ACCESS_TEAM_DOMAIN|CF_ACCESS_AUD|VOICE_ALLOWED_EMAILS)=' "$ENV" > "$ENV.new" || true
printf 'CF_ACCESS_TEAM_DOMAIN=%s\nCF_ACCESS_AUD=%s\nVOICE_ALLOWED_EMAILS=%s\n' "$team" "$aud" "$allowed" >> "$ENV.new"
chmod 600 "$ENV.new"
mv "$ENV.new" "$ENV"
echo "Saved: team $team, allowed $allowed."

systemctl restart voice-to-text
echo "Restarted voice-to-text. The subdomain asks for Google sign-in once it has loaded (about 20 seconds)."
