"""Frozen station sets S_z.

Rule (pre-registered): ERCOT does not publish which NOAA stations it uses per weather zone (Load Profiling Guide
§13.2.2.2 and the 2025 LTLF report say only "two or three" first/second-order NOAA stations per zone; searched
2026-10-14, see lit/query_log.csv). So, per §P4.1's fallback, S_z = the major airport ASOS stations (NWS
first-order sites at commercial-service airports) inside the zone, 2-3 per zone, each verified to lie in the zone
by the ERCOT ZIP-to-Weather-Zone table (LPG Appendix D, 2024-05-01 version) via the nearest 2020 ZCTA centroids,
and each with a GFS MOS (MAV) archive at IEM. Stations whose surrounding load was not in ERCOT before the
validation period (Lubbock, KLBB) or whose zone ZIP serves mostly non-ERCOT load (Beaumont, KBPT) are excluded.
"""

STATIONS: dict[str, list[str]] = {
    "COAST": ["IAH", "HOU", "GLS"],
    "EAST": ["TYR", "LFK", "CLL"],
    "FWEST": ["MAF", "INK"],
    "NORTH": ["SPS", "PRX"],
    "NCENT": ["DFW", "DAL", "ACT"],
    "SOUTH": ["CRP", "BRO", "LRD"],
    "SCENT": ["AUS", "SAT"],
    "WEST": ["ABI", "SJT"],
}

ALL_STATIONS: list[str] = [s for z in STATIONS.values() for s in z]
