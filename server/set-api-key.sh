#!/bin/sh
# Saves the Anthropic API key the transcription server uses for summaries, then restarts it.
#
#   bash /root/voice_to_text/server/set-api-key.sh
#
# The key is read without being shown, and goes only into /etc/voice-to-text/env
# (readable by root alone), which the voice-to-text service loads.
set -eu
umask 077

printf 'Paste your Anthropic API key (it will not be shown), then press Enter: '
stty -echo
trap 'stty echo' EXIT
read -r key
stty echo
trap - EXIT
echo

case "$key" in
  sk-ant-*) ;;
  *) echo "That doesn't look like an Anthropic API key (they start with sk-ant-). Nothing saved."; exit 1 ;;
esac

mkdir -p /etc/voice-to-text
printf 'ANTHROPIC_API_KEY=%s\n' "$key" > /etc/voice-to-text/env
chmod 600 /etc/voice-to-text/env
echo "Saved to /etc/voice-to-text/env."

systemctl restart voice-to-text
echo "Restarted voice-to-text. Summaries are on once it has loaded (about 20 seconds)."
