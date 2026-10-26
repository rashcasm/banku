#!/data/data/com.termux/files/usr/bin/bash
# One-line launcher: `bash /sdcard/Download/go.sh &`
# Kept short on purpose. Driving a phone with a dead screen means typing blind
# through adb, and long commands get mangled or dropped. Everything real lives in
# banku-loop.sh; this just starts it and captures stray output.
mkdir -p /sdcard/Download/banku
exec bash /sdcard/Download/banku-loop.sh >> /sdcard/Download/banku/out.log 2>&1
