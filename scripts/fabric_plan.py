#!/usr/bin/env python3
"""Turn two fabric probes into a two-rank NCCL plan (JSON on stdout).

  python3 scripts/fabric_plan.py --rank0 head.json --rank1 worker.json

Selection rule: a netdev pair on the same IPv4 subnet, both carrier-up, each
mapped to at least one RDMA device. Every shared subnet is used (on GB10 a
single QSFP cable exposes two PCIe functions, hence two /24s and two HCAs);
the rendezvous address is rank 0's address on the first shared subnet.

The plan is fail-closed: no shared RoCE subnet, an RDMA device with no RoCE v2
IPv4 GID, or HCAs whose IPv4 GIDs do not share one GID index aborts.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import sys


def _shared(a: dict, b: dict) -> bool:
    try:
        na = ipaddress.ip_network(f"{a['ipv4']}/{a['prefix']}", strict=False)
        nb = ipaddress.ip_network(f"{b['ipv4']}/{b['prefix']}", strict=False)
    except (KeyError, ValueError):
        return False
    return na == nb and na.prefixlen < 32


def _gid_for(probe: dict, hca: str, ipv4: str) -> int:
    """Lowest RoCE v2 GID index on hca whose IPv4 is ipv4 (prefer 3 on GB10)."""
    for dev in probe["roce"]:
        if dev["dev"] != hca:
            continue
        indexes = sorted(
            g["index"] for g in dev["gids"] if g["ipv4"] == ipv4 and "v2" in g["type"].lower()
        )
        if not indexes:
            raise SystemExit(f"{hca}: no RoCE v2 GID for {ipv4}")
        return 3 if 3 in indexes else indexes[0]
    raise SystemExit(f"{hca}: not present in the probe")


def plan(head: dict, worker: dict) -> dict:
    pairs = []
    for a in head["netdevs"]:
        if not a["carrier"] or not a["rdma"]:
            continue
        for b in worker["netdevs"]:
            if not b["carrier"] or not b["rdma"]:
                continue
            if _shared(a, b):
                pairs.append((a, b))
    if not pairs:
        raise SystemExit(
            "no shared RoCE subnet between the nodes: check the QSFP cable, both ends' IPv4 addresses and carrier"
        )
    # One pair per shared subnet, sorted by network then address.
    by_net = {}
    for a, b in pairs:
        net = ipaddress.ip_network(f"{a['ipv4']}/{a['prefix']}", strict=False)
        by_net.setdefault(str(net), (a, b))
    nets = sorted(by_net)
    gids = []
    hcas = [[], []]
    for net in nets:
        a, b = by_net[net]
        for rank, side in enumerate((a, b)):
            for hca in side["rdma"]:
                if hca not in hcas[rank]:
                    hcas[rank].append(hca)
                    gids.append(_gid_for(head if rank == 0 else worker, hca, side["ipv4"]))
    if len(set(gids)) != 1:
        raise SystemExit(
            f"the shared HCAs report different RoCE v2 GID indexes {sorted(set(gids))}; "
            "pin NCCL_IB_GID_INDEX after checking /sys/class/infiniband/*/ports/1/gid_attrs"
        )
    first = by_net[nets[0]]
    return {
        "subnets": nets,
        "gid": gids[0],
        "rank": [
            {
                "node": head["node"],
                "ip": first[0]["ipv4"],
                "ifname": first[0]["name"],
                "mtu": first[0]["mtu"],
                "hcas": hcas[0],
            },
            {
                "node": worker["node"],
                "ip": first[1]["ipv4"],
                "ifname": first[1]["name"],
                "mtu": first[1]["mtu"],
                "hcas": hcas[1],
            },
        ],
        "rendezvous": first[0]["ipv4"],
        "mtu_uniform": first[0]["mtu"] == first[1]["mtu"],
        "warnings": (
            [] if first[0]["mtu"] == first[1]["mtu"] == 9000
            else [f"fabric MTU is {first[0]['mtu']}/{first[1]['mtu']}, expected 9000/9000"]
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rank0", required=True)
    ap.add_argument("--rank1", required=True)
    ap.add_argument("-o", "--out")
    args = ap.parse_args()
    with open(args.rank0) as f:
        head = json.load(f)
    with open(args.rank1) as f:
        worker = json.load(f)
    out = plan(head, worker)
    payload = json.dumps(out, indent=1, sort_keys=True) + "\n"
    if args.out:
        with open(args.out, "w") as f:
            f.write(payload)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
