#!/data/data/com.termux/files/usr/bin/bash
# Banku's voice on the phone: watch a directory, speak whatever appears in it.
#
# The laptop only has adb, and `adb shell` cannot run termux-tts-speak (the
# termux-* commands live in Termux's private PATH). So this tiny loop runs inside
# Termux and does the one thing the phone is still responsible for: talking.
#
# Protocol: the laptop pushes "<stamp>.txt" into say/. We speak it, then create
# "<stamp>.done", which is how the laptop knows the phone has finished talking and
# can start listening again without hearing itself. Files are handled in name
# order, and the laptop names them by timestamp.

WORK=/sdcard/Download/banku
Q="$WORK/say"
LOG="$WORK/say.log"

mkdir -p "$Q"
termux-wake-lock 2>/dev/null

LANG_CODE="${BANKU_TTS_LANG:-en}"
REGION="${BANKU_TTS_REGION:-IN}"
RATE="${BANKU_TTS_RATE:-1.0}"
ENGINE="${BANKU_TTS_ENGINE:-com.google.android.tts}"

log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }

log "=== speaker starting (lang=$LANG_CODE-$REGION rate=$RATE)"

while true; do
  spoke=0
  for f in "$Q"/*.txt; do
    [ -e "$f" ] || continue
    spoke=1
    base="${f%.txt}"

    text=$(cat "$f" 2>/dev/null)
    rm -f "$f"

    if [ -n "$text" ]; then
      log "say: $text"
      termux-tts-speak -e "$ENGINE" -l "$LANG_CODE" -n "$REGION" \
        -r "$RATE" -s MUSIC "$text" >> "$LOG" 2>&1
    fi

    : > "$base.done"
  done

  [ "$spoke" = 1 ] || sleep 0.15
done
