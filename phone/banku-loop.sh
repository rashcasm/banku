#!/data/data/com.termux/files/usr/bin/bash
# Banku on the phone: send short audio chunks, speak the answer.
#
# The phone does not decide when an utterance ended. It records ~2 s at a time
# and posts each chunk; the laptop runs voice activity detection and stitches the
# chunks into one utterance, replying as soon as it hears a pause. Before this,
# the phone recorded a fixed 7 s window, so a question finished at second one
# still waited six seconds for the clock — that was most of the "replies are
# pretty late".
#
# Two hard-won details:
#
# * `-l` must be a whole number of seconds. The Termux wrapper runs
#   `add_params --ei limit $((OPTARG * 1000))`, so `-l 1.5` is a bash arithmetic
#   error. It fails *silently*, leaves a 15-minute recording running, and every
#   chunk afterwards is a half-written mp4 ("moov atom not found").
#
# * The upload of the previous chunk happens *while* the next one records. The
#   first version recorded, then decoded the JSON with six `jq` calls, then
#   uploaded — about two seconds with the microphone switched off, so half of
#   everything said fell between chunks. Uploading in the background closes that
#   gap; only the recorder's own restart remains, a few milliseconds.

SERVER="${BANKU_SERVER:-http://127.0.0.1:8765}"
ENDPOINT="$SERVER/listen"
ASK=/sdcard/Download/banku-ask.py
SESSION="phone"
WORK=/sdcard/Download/banku
CHUNK="${BANKU_CHUNK_SECONDS:-2}"
LOG="$WORK/banku.log"
CUR="$WORK/chunk.m4a"
PENDING="$WORK/pending.m4a"
RESP="$WORK/resp.raw"

mkdir -p "$WORK"
termux-wake-lock 2>/dev/null

log() { echo "$(date +%H:%M:%S) $*" >> "$LOG"; }

# A previous run may have died mid-recording, which makes every new start fail
# with "Recording already in progress!" and record into the old file instead.
termux-microphone-record -q >/dev/null 2>&1
sleep 1
rm -f "$CUR" "$PENDING" "$RESP"

log "=== starting, endpoint=$ENDPOINT chunk=${CHUNK}s"

pending=""
cycles=0

while true; do
  cycles=$((cycles + 1))
  rm -f "$CUR"

  # Start this chunk first...
  termux-microphone-record -f "$CUR" -r 16000 -e aac -b 64 -c 1 -l "$CHUNK" >/dev/null 2>&1

  # ...then upload the previous one while this one is being recorded.
  upload_pid=""
  if [ -n "$pending" ] && [ -s "$pending" ]; then
    rm -f "$RESP"
    python3 "$ASK" "$ENDPOINT" "$pending" "$SESSION" > "$RESP" 2>>"$LOG" &
    upload_pid=$!
  fi

  sleep "$CHUNK"
  # The recorder writes the mp4 index when it stops; let it finish.
  sleep 0.35

  if [ -n "$upload_pid" ]; then
    wait "$upload_pid" 2>/dev/null

    ACTION=""; TRANSCRIPT=""; REPLY=""; MS=0; FOLLOWUP="false"
    TTS_LANG="en"; TTS_REGION="IN"; TTS_RATE="1.0"; TTS_ENGINE="com.google.android.tts"

    if [ -s "$RESP" ]; then
      IFS=$'\x1f' read -r ACTION TRANSCRIPT REPLY MS FOLLOWUP \
        TTS_LANG TTS_REGION TTS_RATE TTS_ENGINE < "$RESP"
    fi

    case "$ACTION" in
      error)
        log "server error: $TRANSCRIPT"
        sleep 1
        ;;
      reply)
        [ -n "$TRANSCRIPT" ] && log "heard: $TRANSCRIPT"
        [ "$FOLLOWUP" = "true" ] && log "(follow-up: no wake word needed)"
        log "reply (${MS}ms): $REPLY"
        termux-tts-speak -e "$TTS_ENGINE" -l "$TTS_LANG" -n "$TTS_REGION" \
          -r "$TTS_RATE" -s MUSIC "$REPLY" 2>>"$LOG"
        ;;
      idle)
        # Heard speech that was not for us. Worth a line so the log shows near
        # misses, but stay quiet for chunks that were simply silent.
        [ -n "$TRANSCRIPT" ] && log "heard: $TRANSCRIPT -> not addressed to Banku"
        ;;
      "")
        log "no response from server"
        sleep 1
        ;;
    esac
  fi

  rm -f "$pending"

  if [ -s "$CUR" ]; then
    mv -f "$CUR" "$PENDING"
    pending="$PENDING"
  else
    log "capture produced nothing; retrying"
    pending=""
    sleep 0.5
  fi

  if [ $((cycles % 20)) -eq 0 ]; then
    log "still listening (cycle $cycles)"
  fi
done
