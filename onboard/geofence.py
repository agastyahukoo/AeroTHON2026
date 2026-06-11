"""
geofence.py -- Dual-geofence architecture per Design Report Safety B.

  1. Pixhawk geofence ...... hard boundary, configured in FC params on site
                             (FENCE_ENABLE / FENCE_ACTION / FENCE_ALT_MAX).
  2. OBC geofence .......... THIS module. A custom point-in-polygon check,
                             deliberately TIGHTER (inset by obc_margin_m) than
                             the official fence so course correction happens
                             BEFORE the mandated, mission-ending RTL/LAND.
  3. Mission geofence ...... Pi-bounded delivery search area used to generate
                             the S4 grid search pattern after QR decoding.

Coordinates arrive on site via the GCS mission-planning page and are written
to mission_geofence.json (this file is what "Export package" ships).
"""

from __future__ import annotations

import json
import math
import os

try:
    from . import config
except ImportError:
    import config


def _point_in_polygon(lat, lon, poly):
    """Ray-casting point-in-polygon; poly = [[lat, lon], ...]."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        yi, xi = poly[i]
        yj, xj = poly[j]
        if ((xi > lon) != (xj > lon)) and \
           (lat < (yj - yi) * (lon - xi) / (xj - xi + 1e-12) + yi):
            inside = not inside
        j = i
    return inside


def _inset_polygon(poly, margin_m):
    """Shrink polygon toward its centroid by ~margin_m (adequate for the
    convex AeroTHON field boundaries)."""
    if len(poly) < 3:
        return poly
    clat = sum(p[0] for p in poly) / len(poly)
    clon = sum(p[1] for p in poly) / len(poly)
    out = []
    for lat, lon in poly:
        dn = (lat - clat) * 111_111.0
        de = (lon - clon) * 111_111.0 * math.cos(math.radians(clat))
        d = math.hypot(dn, de)
        k = max(d - margin_m, 0.0) / (d + 1e-9)
        out.append([clat + (lat - clat) * k, clon + (lon - clon) * k])
    return out


def _dist_to_edge_m(lat, lon, poly):
    """Min distance from point to polygon edges, metres (local flat earth)."""
    clat = math.radians(lat)
    px, py = lon * 111_111.0 * math.cos(clat), lat * 111_111.0
    best = float("inf")
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ax, ay = a[1] * 111_111.0 * math.cos(clat), a[0] * 111_111.0
        bx, by = b[1] * 111_111.0 * math.cos(clat), b[0] * 111_111.0
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) /
                         (dx * dx + dy * dy + 1e-12)))
        best = min(best, math.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


class GeofenceManager:
    def __init__(self, path: str | None = None):
        self.path = path or config.GEOFENCE["file"]
        self.data = {"official_fence": [], "delivery_zone": [],
                     "corridor_path": [], "takeoff_point": None,
                     "fence_alt_max_m": 18.0}
        self.load()

    # ----------------------------------------------------------------- io
    def load(self):
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data.update(json.load(f))
        return self.data

    def save(self, data: dict | None = None):
        if data:
            self.data.update(data)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)
        return self.data

    # ------------------------------------------------------------- fences
    @property
    def obc_fence(self):
        """Tighter inner fence (Safety B)."""
        return _inset_polygon(self.data["official_fence"],
                              config.GEOFENCE["obc_margin_m"])

    def check(self, lat, lon, alt_m=0.0) -> dict:
        official = self.data["official_fence"]
        out = {"configured": len(official) >= 3,
               "inside_official": True, "inside_obc": True,
               "inside_delivery": False, "edge_dist_m": None,
               "proximity_warn": False, "alt_ok": alt_m <= self.data.get(
                   "fence_alt_max_m", 18.0)}
        if not out["configured"]:
            return out
        out["inside_official"] = _point_in_polygon(lat, lon, official)
        out["inside_obc"] = _point_in_polygon(lat, lon, self.obc_fence)
        out["edge_dist_m"] = _dist_to_edge_m(lat, lon, official)
        out["proximity_warn"] = (out["edge_dist_m"] is not None and
                                 out["edge_dist_m"] <
                                 config.GEOFENCE["proximity_warn_m"])
        dz = self.data["delivery_zone"]
        if len(dz) >= 3:
            out["inside_delivery"] = _point_in_polygon(lat, lon, dz)
        return out

    # ------------------------------------------- S4 grid search generator
    def delivery_search_pattern(self, lane_spacing_m=2.0) -> list:
        """Pre-generated lawnmower grid inside the delivery-zone polygon
        (Geofence-Constrained Search Pattern, algorithm C)."""
        dz = self.data["delivery_zone"]
        if len(dz) < 3:
            return []
        lats = [p[0] for p in dz]
        lons = [p[1] for p in dz]
        dlat = lane_spacing_m / 111_111.0
        wpts, lat, flip = [], min(lats), False
        while lat <= max(lats):
            row = sorted([min(lons), max(lons)], reverse=flip)
            for lon in row:
                # nudge toward zone interior until inside
                for k in range(20):
                    t = k / 20.0
                    clat = sum(lats) / len(lats)
                    clon = sum(lons) / len(lons)
                    li, lo = lat + (clat - lat) * t, lon + (clon - lon) * t
                    if _point_in_polygon(li, lo, dz):
                        wpts.append([li, lo])
                        break
            lat += dlat
            flip = not flip
        return wpts
