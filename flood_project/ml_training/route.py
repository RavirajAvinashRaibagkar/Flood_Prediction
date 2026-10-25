"""
route.py  --  FloodSentinel AI  (Phase 5: Evacuation Routing)
=============================================================
Converts predicted flood masks into a flood-risk cost layer,
builds a weighted road network from OSMnx, and computes
safest evacuation routes to shelters/high ground.

Pipeline:
  1. Load flood probability mask (raster or numpy array)
  2. Vectorize flood mask -> GeoDataFrame of flooded polygons
  3. Buffer by 100m (uncertainty margin)
  4. Download road network with OSMnx for the bounding box
  5. Tag road edges as flooded/dry using spatial join
  6. Compute risk-weighted shortest path (Dijkstra)
  7. Output: route GeoJSON + folium interactive map

Usage:
  python route.py --lat 25.0 --lon 85.0 --radius 20  # in km
  python route.py --mask flood_mask.npy --crs EPSG:4326

Datasets compatible:
  - Any numpy array of shape (H, W) with values in [0, 1]
  - Sen1Floods11 segmentation output
  - U-Net segmentation mask

Dependencies:
  pip install osmnx shapely geopandas folium networkx scipy
"""

import os
import sys
import json
import argparse
import numpy as np
import warnings

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))

# Optional imports - warn gracefully
try:
    import osmnx as ox
    import networkx as nx
    HAS_OSM = True
except ImportError:
    HAS_OSM = False
    print("[WARN] osmnx not installed. Run: pip install osmnx")

try:
    import geopandas as gpd
    from shapely.geometry import Point, Polygon, MultiPolygon, LineString, mapping
    from shapely.ops import unary_union
    HAS_GEO = True
except ImportError:
    HAS_GEO = False
    print("[WARN] geopandas/shapely not installed. Run: pip install geopandas shapely")

try:
    import folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    print("[WARN] folium not installed. Run: pip install folium")


# ============================================================
# Step 1: Vectorize flood mask
# ============================================================

def mask_to_flood_polygons(
    flood_mask: np.ndarray,
    threshold: float = 0.5,
    bbox: tuple = None,
    crs: str = "EPSG:4326"
) -> "gpd.GeoDataFrame":
    """
    Convert a flood probability raster to a GeoDataFrame of flood polygons.

    Args:
        flood_mask : (H, W) float32 array of flood probability in [0, 1]
        threshold  : probability threshold for binary classification
        bbox       : (min_lon, min_lat, max_lon, max_lat) geographic bounds
        crs        : coordinate reference system

    Returns:
        GeoDataFrame with geometry=flooded polygons and 'flood_prob' attribute
    """
    if not HAS_GEO:
        raise ImportError("geopandas and shapely required.")

    from rasterio import transform as rio_transform
    from rasterio.features import shapes
    import rasterio

    binary = (flood_mask >= threshold).astype(np.uint8)
    H, W = binary.shape

    if bbox is not None:
        min_lon, min_lat, max_lon, max_lat = bbox
        tfm = rio_transform.from_bounds(min_lon, min_lat, max_lon, max_lat, W, H)
    else:
        # Default: assume 1km resolution centered at (0, 0)
        tfm = rio_transform.from_bounds(-0.1, -0.1, 0.1, 0.1, W, H)

    polys = []
    for shp, val in shapes(binary, transform=tfm):
        if val == 1:
            polys.append(Polygon(shp["coordinates"][0]))

    if not polys:
        print("  [WARN] No flooded pixels above threshold.")
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame(
        {"geometry": polys, "flood_prob": threshold},
        crs=crs
    )
    print(f"  Vectorized {len(gdf)} flood polygons")
    return gdf


def buffer_flood_polygons(flood_gdf: "gpd.GeoDataFrame", buffer_m: float = 100) -> "gpd.GeoDataFrame":
    """Buffer flood polygons by `buffer_m` meters (uncertainty margin)."""
    if flood_gdf.crs and flood_gdf.crs.is_geographic:
        # Convert to UTM for metric buffering
        flood_utm = flood_gdf.to_crs(flood_gdf.estimate_utm_crs())
        buffered  = flood_utm.copy()
        buffered["geometry"] = flood_utm.geometry.buffer(buffer_m)
        return buffered.to_crs(flood_gdf.crs)
    else:
        flood_gdf = flood_gdf.copy()
        flood_gdf["geometry"] = flood_gdf.geometry.buffer(buffer_m)
        return flood_gdf


# ============================================================
# Step 2: Build risk-weighted road network
# ============================================================

def build_flood_risk_network(
    center_lat: float,
    center_lon: float,
    radius_km: float,
    flood_gdf: "gpd.GeoDataFrame",
    flood_penalty: float = 1000.0,
) -> "nx.MultiDiGraph":
    """
    Download road network and assign flood-risk weights to edges.

    Args:
        center_lat    : center latitude
        center_lon    : center longitude
        radius_km     : network radius in km
        flood_gdf     : flooded area polygons (GeoDataFrame)
        flood_penalty : edge weight multiplier for flooded roads (default 1000x)

    Returns:
        networkx.MultiDiGraph with 'risk_weight' edge attribute
    """
    if not HAS_OSM:
        raise ImportError("osmnx required. Run: pip install osmnx")

    print(f"  Downloading road network (r={radius_km}km, center=({center_lat},{center_lon}))...")
    G = ox.graph_from_point(
        (center_lat, center_lon),
        dist=radius_km * 1000,
        network_type="drive",
        simplify=True,
    )
    print(f"  Network: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # Add base weight = travel time (seconds)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)

    # Tag flooded edges via spatial join
    edges_gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)[["geometry", "travel_time"]].copy()
    edges_gdf = edges_gdf.reset_index()
    edges_gdf.crs = "EPSG:4326"

    if len(flood_gdf) > 0:
        flood_union = unary_union(flood_gdf.to_crs("EPSG:4326").geometry)
        flooded_mask = edges_gdf.geometry.intersects(flood_union)
        print(f"  Flooded edges: {flooded_mask.sum()} / {len(edges_gdf)}")
    else:
        flooded_mask = edges_gdf.geometry.apply(lambda _: False)

    # Assign risk weight
    for u, v, k, data in G.edges(data=True, keys=True):
        base_time = data.get("travel_time", 60)
        data["risk_weight"] = base_time * flood_penalty if flooded_mask.any() else base_time

    return G


# ============================================================
# Step 3: Evacuation routing
# ============================================================

def find_safe_route(
    G: "nx.MultiDiGraph",
    origin_node: int,
    shelters: list,
    weight: str = "risk_weight",
) -> dict:
    """
    Find safest route from origin to nearest safe shelter.
    Uses Dijkstra with risk-weighted edge costs.

    Args:
        G            : risk-weighted road network
        origin_node  : OSMnx node ID of evacuation start point
        shelters     : list of (lat, lon) tuples for shelter locations
        weight       : edge weight attribute ('risk_weight' or 'travel_time')

    Returns:
        dict with route nodes, geometry, and route metrics
    """
    if not HAS_OSM:
        raise ImportError("osmnx required.")

    best_route = None
    best_cost  = float("inf")
    best_shelter = None

    for lat, lon in shelters:
        shelter_node = ox.nearest_nodes(G, lon, lat)
        try:
            route = nx.shortest_path(G, origin_node, shelter_node, weight=weight)
            cost  = nx.shortest_path_length(G, origin_node, shelter_node, weight=weight)
            if cost < best_cost:
                best_cost    = cost
                best_route   = route
                best_shelter = (lat, lon)
        except nx.NetworkXNoPath:
            print(f"  [WARN] No path to shelter at ({lat},{lon})")

    if best_route is None:
        print("  [ERR] No route found to any shelter!")
        return {}

    # Route analysis
    route_edges = list(zip(best_route[:-1], best_route[1:]))
    n_flooded_edges = sum(
        1 for u, v in route_edges
        if G[u][v][0].get("risk_weight", 60) > 100  # flooded threshold
    )
    total_edges = len(route_edges)
    flooded_pct = 100 * n_flooded_edges / max(total_edges, 1)

    print(f"  Route found:")
    print(f"    Nodes: {len(best_route)}")
    print(f"    Risk cost: {best_cost:.1f} seconds (risk-weighted)")
    print(f"    Flooded segments: {n_flooded_edges}/{total_edges} ({flooded_pct:.1f}%)")
    print(f"    Best shelter: {best_shelter}")

    # Get actual travel time (unpenalized)
    actual_time = sum(
        G[u][v][0].get("travel_time", 60) for u, v in route_edges
    )
    print(f"    Estimated drive time (clear roads): {actual_time/60:.1f} min")

    return {
        "route_nodes"       : best_route,
        "best_shelter"      : best_shelter,
        "risk_cost"         : float(best_cost),
        "actual_time_s"     : float(actual_time),
        "total_edges"       : total_edges,
        "flooded_edges"     : n_flooded_edges,
        "flooded_pct"       : float(flooded_pct),
    }


# ============================================================
# Step 4: Export GeoJSON + Folium map
# ============================================================

def export_geojson(G, route_nodes, flood_gdf, output_path):
    """Export route and flood areas to GeoJSON."""
    features = []

    # Route as linestring
    if HAS_GEO:
        coords = [
            (G.nodes[n]["x"], G.nodes[n]["y"])
            for n in route_nodes
        ]
        route_feature = {
            "type": "Feature",
            "properties": {"type": "evacuation_route"},
            "geometry": mapping(LineString(coords))
        }
        features.append(route_feature)

        # Flood polygons
        for _, row in flood_gdf.iterrows():
            features.append({
                "type": "Feature",
                "properties": {"type": "flood_zone"},
                "geometry": mapping(row.geometry)
            })

    geojson = {"type": "FeatureCollection", "features": features}
    with open(output_path, "w") as f:
        json.dump(geojson, f, indent=2)
    print(f"  GeoJSON saved: {output_path}")


def build_folium_map(G, route_result, flood_gdf, shelters, center_lat, center_lon, output_path):
    """Build interactive folium map with route and flood overlay."""
    if not HAS_FOLIUM:
        print("  [SKIP] folium not installed.")
        return

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles="OpenStreetMap")

    # Flood zones
    if HAS_GEO and len(flood_gdf) > 0:
        folium.GeoJson(
            flood_gdf.to_json(),
            style_function=lambda _: {
                "fillColor": "#ff4444",
                "color": "#cc0000",
                "weight": 1,
                "fillOpacity": 0.4,
            },
            name="Flood Zones",
            tooltip="Flood Zone"
        ).add_to(m)

    # Route
    if route_result and "route_nodes" in route_result and G is not None:
        route_latlons = [
            (G.nodes[n]["y"], G.nodes[n]["x"])
            for n in route_result["route_nodes"]
        ]
        folium.PolyLine(
            route_latlons, color="#00cc00", weight=4, opacity=0.9,
            tooltip=f"Safest Route ({route_result['actual_time_s']/60:.1f} min)"
        ).add_to(m)

    # Shelters
    for lat, lon in shelters:
        folium.Marker(
            location=[lat, lon],
            popup="Emergency Shelter",
            icon=folium.Icon(color="green", icon="home"),
        ).add_to(m)

    # Origin
    folium.Marker(
        location=[center_lat, center_lon],
        popup="Evacuation Start",
        icon=folium.Icon(color="red", icon="exclamation-sign"),
    ).add_to(m)

    folium.LayerControl().add_to(m)
    m.save(output_path)
    print(f"  Folium map saved: {output_path}")


# ============================================================
# Evaluation on flood events
# ============================================================

def evaluate_routing(G, flood_events, shelters, output_dir):
    """
    Evaluate routing on 3 held-out flood events.
    Reports route validity, % on flooded roads, detour length.
    """
    results = []
    for i, event in enumerate(flood_events[:3]):
        lat, lon = event["origin_lat"], event["origin_lon"]
        flood_gdf = event["flood_gdf"]

        print(f"\n  Event {i+1}: ({lat:.4f}, {lon:.4f})")
        origin_node = ox.nearest_nodes(G, lon, lat)

        # Safe route
        route_safe  = find_safe_route(G, origin_node, shelters, weight="risk_weight")
        # Baseline (fastest, ignores floods)
        route_fast  = find_safe_route(G, origin_node, shelters, weight="travel_time")

        detour_time = 0.0
        if route_safe and route_fast:
            detour_time = route_safe.get("actual_time_s", 0) - route_fast.get("actual_time_s", 0)
            print(f"    Detour vs fastest route: {detour_time/60:.1f} min")

        results.append({
            "event"         : i + 1,
            "origin"        : (lat, lon),
            "safe_route"    : route_safe,
            "fast_route"    : route_fast,
            "detour_s"      : float(detour_time),
        })

    # Summary
    report_path = os.path.join(output_dir, "routing_evaluation.json")
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Routing evaluation saved: {report_path}")

    valid_routes = sum(1 for r in results if r["safe_route"])
    avg_flood_pct = np.mean([r["safe_route"].get("flooded_pct", 0) for r in results if r["safe_route"]])
    avg_detour = np.mean([r["detour_s"] for r in results]) / 60

    print(f"\n  Routing Summary (3 events):")
    print(f"    Valid routes: {valid_routes}/3")
    print(f"    Avg flooded road %: {avg_flood_pct:.1f}%")
    print(f"    Avg detour time: {avg_detour:.1f} min")

    return results


# ============================================================
# Demo run
# ============================================================

def demo_route(lat=25.37, lon=84.99, radius_km=15.0, flood_mask=None):
    """
    Demo: Patna, Bihar, India flood evacuation routing.
    Uses simulated or user-provided real flood mask (from CNN/U-Net segmentation).
    """
    if not (HAS_OSM and HAS_GEO):
        print("[SKIP] Demo requires osmnx and geopandas.")
        print("  Install: pip install osmnx geopandas shapely folium")
        return

    print("=" * 60)
    print("  DEMO: Flood Evacuation Routing -- Patna, Bihar")
    print("=" * 60)

    if flood_mask is None:
        print("\n[1] Simulating flood mask (50x50 grid)...")
        H, W = 50, 50
        rng = np.random.RandomState(42)
        flood_mask = np.zeros((H, W), dtype=np.float32)
        # Simulate river flooding along bottom-left region
        flood_mask[30:, :20] = rng.uniform(0.6, 0.95, (20, 20))
        flood_mask[40:, :30] = rng.uniform(0.7, 0.99, (10, 30))
    else:
        print(f"\n[1] Using provided flood mask shape {flood_mask.shape}...")

    bbox = (lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1)

    try:
        from rasterio import transform as rio_transform
        from rasterio.features import shapes
        print("[2] Vectorizing flood mask...")
        flood_gdf = mask_to_flood_polygons(flood_mask, threshold=0.5, bbox=bbox)
        flood_gdf = buffer_flood_polygons(flood_gdf, buffer_m=100)
    except ImportError:
        print("  [SKIP] rasterio not installed. Using empty flood zone.")
        flood_gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    print("\n[3] Building risk-weighted road network...")
    try:
        G = build_flood_risk_network(lat, lon, radius_km, flood_gdf)
    except Exception as e:
        print(f"  [ERR] Could not download road network: {e}")
        print("  (Network access required for OSMnx)")
        return

    # Define shelters (example: high-ground points)
    shelters = [
        (lat + 0.08, lon + 0.05),  # shelter north-east
        (lat + 0.10, lon - 0.08),  # shelter north-west
        (lat + 0.05, lon + 0.10),  # shelter east
    ]

    origin_node = ox.nearest_nodes(G, lon, lat)

    print("\n[4] Computing safest evacuation route...")
    route_result = find_safe_route(G, origin_node, shelters, weight="risk_weight")

    output_dir = os.path.join(BASE, "saved_models", "routing")
    os.makedirs(output_dir, exist_ok=True)

    print("\n[5] Exporting outputs...")
    if route_result and "route_nodes" in route_result:
        export_geojson(
            G, route_result["route_nodes"], flood_gdf,
            os.path.join(output_dir, "route.geojson")
        )
        build_folium_map(
            G, route_result, flood_gdf, shelters, lat, lon,
            os.path.join(output_dir, "route_map.html")
        )

    print("\n[OK] Routing demo complete!")
    return route_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lat", type=float, default=25.37)
    parser.add_argument("--lon", type=float, default=84.99)
    parser.add_argument("--radius", type=float, default=15.0)
    parser.add_argument("--mask", type=str, default=None)
    args = parser.parse_args()

    if args.mask and os.path.exists(args.mask):
        print(f"Loading flood mask: {args.mask}")
        mask = np.load(args.mask)
        demo_route(args.lat, args.lon, args.radius, flood_mask=mask)
    else:
        demo_route(args.lat, args.lon, args.radius)
