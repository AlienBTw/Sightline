#!/bin/bash
# parallel range download: pdl.sh URL OUT N
URL=$1; OUT=$2; N=${3:-16}
SIZE=$(curl -sIL "$URL" | grep -i '^content-length' | tail -1 | tr -dc 0-9)
echo "size=$SIZE"
CH=$(( (SIZE + N - 1) / N ))
mkdir -p "$OUT.parts"
for i in $(seq 0 $((N-1))); do
  S=$((i*CH)); E=$(( S+CH-1 )); [ $E -ge $SIZE ] && E=$((SIZE-1))
  P="$OUT.parts/$i"; HAVE=0; [ -f "$P" ] && HAVE=$(stat -c%s "$P")
  NEED=$((E-S+1))
  if [ $HAVE -lt $NEED ]; then
    ( until [ "$(stat -c%s "$P" 2>/dev/null || echo 0)" -ge $NEED ]; do H=$(stat -c%s "$P" 2>/dev/null || echo 0); curl -sL --retry 5 -r $((S+H))-$E "$URL" >> "$P"; done ) &
  fi
done
wait
cat $(for i in $(seq 0 $((N-1))); do echo "$OUT.parts/$i"; done) > "$OUT.tmp"
[ "$(stat -c%s "$OUT.tmp")" = "$SIZE" ] && mv "$OUT.tmp" "$OUT" && rm -rf "$OUT.parts" && echo PDL_OK || echo PDL_SIZE_MISMATCH