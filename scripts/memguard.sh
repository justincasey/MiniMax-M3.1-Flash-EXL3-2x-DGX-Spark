#!/usr/bin/env bash
# Fail-closed host memory guard for one serving container (run as root on a Spark).
#
#   memguard.sh watch CONTAINER LOG STAGE START_FLOOR_KIB STEADY_FLOOR_KIB SWAP_MAX_KIB DROP_EVERY_S [DROP_LOADING]
#   memguard.sh stop LOG
#
# The floor is START_FLOOR_KIB while STAGE says "loading" and STEADY_FLOOR_KIB
# once it says "ready" (m31.sh writes the stage file when /v1/models answers).
# Three consecutive 1 s samples below the floor, or swap growth above the cap,
# kill the container: on GB10 the GPU and the host share one memory pool, and a
# host that runs out takes the whole node down instead of failing a request.
#
# Page cache is dropped every DROP_EVERY_S seconds while the container runs:
# a 100+ GiB weight load leaves cache that the GPU allocator cannot reclaim in
# time (measured on the 2.15 bpw TP2 lane). DROP_LOADING=0 skips the drops
# until the stage file says "ready", so prefetched shards survive the load.
set -uo pipefail

mode=${1:-}
case "$mode" in
  watch)
    container=${2:?container}
    log=${3:?log}
    stage=${4:?stage}
    start_floor=${5:?start floor KiB}
    steady_floor=${6:?steady floor KiB}
    swap_max=${7:?swap max KiB}
    drop_every=${8:?drop every seconds}
    drop_loading=${9:-1}
    pidfile="${log}.pid"
    mkdir -p "$(dirname "$log")"
    echo $$ >"$pidfile"
    echo "$(date -Is) guard: armed pid=$$ floor=${start_floor}KiB steady=${steady_floor}KiB swap_max=${swap_max}KiB drop_every=${drop_every}s" >>"$log"

    swap_used_kib() {
      awk '/^SwapTotal:/{t=$2}/^SwapFree:/{f=$2}END{printf "%d", t-f}' /proc/meminfo
    }
    # The guard is armed before `docker run`: wait for the container to come up
    # (120 s), and only treat it as gone after it has been seen running.
    waited=0
    until docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -q true; do
      if (( waited >= 120 )); then
        echo "$(date -Is) guard: container $container never started; exiting" >>"$log"
        rm -f "$pidfile"
        exit 1
      fi
      sleep 1; waited=$((waited + 1))
    done
    echo "$(date -Is) guard: watching $container (started after ${waited}s)" >>"$log"
    base_swap=$(swap_used_kib)
    low=0
    ticks=0
    # Lowest MemAvailable seen per stage, rewritten on each new low: <log>.min
    # holds "loading_min_kib=N" and "ready_min_kib=N" for the harness.
    min_loading=-1 min_ready=-1
    while :; do
      if ! docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null | grep -q true; then
        echo "$(date -Is) guard: container $container is gone; exiting" >>"$log"
        break
      fi
      avail=$(awk '/^MemAvailable:/{print int($2)}' /proc/meminfo)
      floor=$start_floor
      ready=0
      if grep -q '^ready' "$stage" 2>/dev/null; then floor=$steady_floor; ready=1; fi
      if (( ready == 1 )); then
        if (( min_ready < 0 || avail < min_ready )); then min_ready=$avail; printf 'loading_min_kib=%s\nready_min_kib=%s\n' "$min_loading" "$min_ready" >"$log.min"; fi
      else
        if (( min_loading < 0 || avail < min_loading )); then min_loading=$avail; printf 'loading_min_kib=%s\nready_min_kib=%s\n' "$min_loading" "$min_ready" >"$log.min"; fi
      fi
      if (( avail < floor )); then low=$((low + 1)); else low=0; fi
      swap_now=$(swap_used_kib)
      growth=$((swap_now - base_swap))
      if (( growth > swap_max )); then
        echo "$(date -Is) GUARD KILL swap growth ${growth}KiB > ${swap_max}KiB (MemAvailable ${avail}KiB)" >>"$log"
        docker kill "$container" >/dev/null 2>&1 || true
        break
      fi
      if (( low >= 3 )); then
        echo "$(date -Is) GUARD KILL MemAvailable ${avail}KiB < floor ${floor}KiB for 3 samples" >>"$log"
        docker kill "$container" >/dev/null 2>&1 || true
        break
      fi
      ticks=$((ticks + 1))
      if (( drop_every > 0 && ticks % drop_every == 0 )) && (( ready == 1 || drop_loading == 1 )); then
        sync
        echo 1 >/proc/sys/vm/drop_caches 2>/dev/null || true
      fi
      sleep 1
    done
    rm -f "$pidfile"
    ;;
  stop)
    log=${2:?log}
    pidfile="${log}.pid"
    if [[ -f "$pidfile" ]]; then
      pid=$(cat "$pidfile" 2>/dev/null || true)
      if [[ -n "${pid:-}" ]]; then
        kill -TERM "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
        sleep 1
        kill -KILL "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
      fi
      rm -f "$pidfile"
      echo "$(date -Is) guard: stopped pid=${pid:-?}" >>"$log" 2>/dev/null || true
    fi
    ;;
  *)
    echo "usage: $0 watch CONTAINER LOG STAGE START_FLOOR_KIB STEADY_FLOOR_KIB SWAP_MAX_KIB DROP_EVERY_S | stop LOG" >&2
    exit 2
    ;;
esac
