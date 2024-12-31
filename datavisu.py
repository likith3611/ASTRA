from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
from av2.map.map_api import ArgoverseStaticMap
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from shapely.geometry import Point

SCENARIO_PATH = Path("fill path to scenario file here>")
MAP_PATH = Path("<fill path to map file here>")
scenario = load_argoverse_scenario_parquet(SCENARIO_PATH)
print(f"Scenario ID: {scenario.scenario_id}")
print(f"Number of Agents: {len(scenario.tracks)}")
print(f"Number of Timestamps: {len(scenario.timestamps_ns)}")

map_api = ArgoverseStaticMap.from_json(MAP_PATH)
fig, ax = plt.subplots(figsize=(12, 8))
for lane_id, lane_segment in map_api.vector_lane_segments.items():
    print(f"Lane ID: {lane_id}")
    polygon = lane_segment.polygon_boundary
    if polygon is not None:
        if isinstance(polygon, np.ndarray):
            xs = polygon[:, 0]
            ys = polygon[:, 1]
        elif isinstance(polygon, list):
            if all(isinstance(point, Point) for point in polygon):
                xs = [point.x for point in polygon]
                ys = [point.y for point in polygon]
            elif all(isinstance(point, (tuple, list)) and len(point) >= 2 for point in polygon):
                xs = [point[0] for point in polygon]
                ys = [point[1] for point in polygon]
            else:
                continue
        elif isinstance(polygon, Point):
            xs = [polygon.x]
            ys = [polygon.y]
        else:
            continue
        ax.plot(xs, ys, color='gray', linewidth=1, label='Lane Boundary' if lane_id == 0 else "")
    if hasattr(lane_segment, 'is_intersection') and lane_segment.is_intersection:
        ax.fill(xs, ys, color='orange', alpha=0.3, label='Intersection' if lane_id == 0 else "")

for area_id, drivable_area in map_api.vector_drivable_areas.items():
    boundary = drivable_area.area_boundary
    if isinstance(boundary, np.ndarray):
        xs = boundary[:, 0]
        ys = boundary[:, 1]
    elif isinstance(boundary, list):
        if all(isinstance(point, Point) for point in boundary):
            xs = [point.x for point in boundary]
            ys = [point.y for point in boundary]
        elif all(isinstance(point, (tuple, list)) and len(point) >= 2 for point in boundary):
            xs = [point[0] for point in boundary]
            ys = [point[1] for point in boundary]
        else:
            continue
    elif isinstance(boundary, Point):
        xs = [boundary.x]
        ys = [boundary.y]
    else:
        continue
    ax.fill(xs, ys, color='lightgray', alpha=0.5, label='Drivable Area' if area_id == 0 else "")

for crosswalk_id, crosswalk in map_api.vector_pedestrian_crossings.items():
    edge1_2d, edge2_2d = crosswalk.get_edges_2d()
    ax.plot([edge1_2d[0][0], edge1_2d[1][0]], [edge1_2d[0][1], edge1_2d[1][1]], color='green', linewidth=2, label='Crosswalk')
    ax.plot([edge2_2d[0][0], edge2_2d[1][0]], [edge2_2d[0][1], edge2_2d[1][1]], color='green', linewidth=2)

for track in scenario.tracks:
    x = [state.position[0] for state in track.object_states if state is not None]
    y = [state.position[1] for state in track.object_states if state is not None]
    ax.plot(x, y, marker='o', label=f"{track.object_type.name}")

for track in scenario.tracks:
    if track.object_states[0] is not None:
        state = track.object_states[0]
        ax.scatter(state.position[0], state.position[1], color='blue', marker='x', s=50, label='Agent Start Position')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Argoverse Scenario Visualization with Static and Dynamic Elements')
ax.legend(loc='upper right', fontsize='small')

plt.show()
