#!/usr/bin/env python3
"""Read-only fabric inventory for one DGX Spark (JSON on stdout).

No writes, no network, no sudo. Reports every RDMA device with its RoCE v2
IPv4 GIDs, every carrier-up netdev with an IPv4 address, and the netdev -> RDMA
mapping derived from the GID address (the P2p function's netdev has no
device/infiniband symlink, so the sysfs link alone is not enough).

  python3 scripts/fabric_probe.py
"""

from __future__ import annotations

import glob
import ipaddress
import json
import os
import socket
import subprocess
import sys

SYS_IB = "/sys/class/infiniband"
EXCLUDE_PREFIX = ("lo", "docker", "br-", "veth", "tailscale", "wlP")


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def _ip_json(kind: str) -> list:
    try:
        out = subprocess.run(["ip", "-j", "-4", kind, "show"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return json.loads(out.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return []


def _gid_ipv4(gid: str) -> str | None:
    """IPv4 address of a RoCE v2 IPv4-mapped GID (::ffff:a.b.c.d)."""
    hexchars = gid.replace(":", "")
    if len(hexchars) != 32:
        return None
    # ::ffff:a.b.c.d -> 20 zero hex chars, "ffff", then four octets.
    if hexchars[:20] != "0" * 20 or hexchars[20:24].lower() != "ffff":
        return None
    octets = [int(hexchars[i : i + 2], 16) for i in range(24, 32, 2)]
    return ".".join(str(o) for o in octets)


def roce_devices() -> list[dict]:
    devices = []
    for dev_path in sorted(glob.glob(f"{SYS_IB}/*")):
        dev = os.path.basename(dev_path)
        port = f"{dev_path}/ports/1"
        state = _read(f"{port}/state")
        link_layer = _read(f"{port}/link_layer")
        gids = []
        for gid_path in sorted(glob.glob(f"{port}/gids/*"), key=lambda p: int(os.path.basename(p))):
            index = int(os.path.basename(gid_path))
            raw = _read(gid_path)
            kind = _read(f"{port}/gid_attrs/types/{index}")
            ipv4 = _gid_ipv4(raw)
            if ipv4:
                gids.append({"index": index, "type": kind, "ipv4": ipv4})
        devices.append(
            {
                "dev": dev,
                "state": state,
                "link_layer": link_layer,
                "gids": gids,
            }
        )
    return devices


def netdevs() -> list[dict]:
    links = {entry.get("ifname"): entry for entry in _ip_json("link") if entry.get("ifname")}
    addrs = _ip_json("addr")
    out = []
    for entry in addrs:
        name = entry.get("ifname")
        if not name or name.startswith(EXCLUDE_PREFIX):
            continue
        for info in entry.get("addr_info", []):
            if info.get("family") != "inet":
                continue
            link = links.get(name, {})
            out.append(
                {
                    "name": name,
                    "ipv4": info.get("local"),
                    "prefix": info.get("prefixlen"),
                    "mtu": link.get("mtu"),
                    "carrier": 1 if link.get("operstate") == "UP" else 0,
                    "operstate": link.get("operstate"),
                    "rdma": sorted(
                        os.path.basename(p)
                        for p in glob.glob(f"/sys/class/net/{name}/device/infiniband/*")
                    ),
                }
            )
    return out


def main() -> int:
    devices = roce_devices()
    nets = netdevs()
    # Map every netdev to the RDMA devices whose RoCE v2 IPv4 GID equals it.
    gid_by_ip: dict[str, list[str]] = {}
    for dev in devices:
        for gid in dev["gids"]:
            gid_by_ip.setdefault(gid["ipv4"], []).append(dev["dev"])
    for net in nets:
        net["rdma"] = sorted(set(net["rdma"]) | set(gid_by_ip.get(net["ipv4"], [])))
    print(
        json.dumps(
            {
                "node": socket.gethostname(),
                "roce": devices,
                "netdevs": nets,
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
