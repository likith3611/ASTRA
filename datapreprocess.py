import os
import torch
import pickle
import math
from typing import List
from av2.datasets.motion_forecasting.scenario_serialization import load_argoverse_scenario_parquet
from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path
from shapely.geometry import Point

DATASET_PATH = '/home/light/Documents/datasets/motion-prediction/argoverse/trail'
OUTPUT_PATH = './preprocessed_dataset/'
BATCH_SIZE = 5

os.makedirs(OUTPUT_PATH, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def load_dataset(dataset_path: str):
    dataset_info = []
    trajectories = []
    road_vectors = []
    agent_types = []
    timestamps = []
    orientations = []
    accelerations = []
    road_edges = []
    stop_signs = []
    crosswalks = []
    
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        
        scenario_file = None
        map_file = None
        
        for file in os.listdir(folder_path):
            if file.endswith('.parquet'):
                scenario_file = os.path.join(folder_path, file)
            if file.endswith('.json'):
                map_file = os.path.join(folder_path, file)
        
        if not scenario_file or not map_file:
            print(f"Skipping folder {folder}: Missing required files")
            continue
        
        scenario = load_argoverse_scenario_parquet(scenario_file)
        map_api = ArgoverseStaticMap.from_json(Path(map_file))
        
        scenario_info = {
            'folder': folder,
            'scenario_id': scenario.scenario_id,
            'num_agents': len(scenario.tracks),
            'num_timestamps': len(scenario.timestamps_ns)
        }
        
        agent_data = []
        type_data = []
        time_data = []
        orientation_data = []
        acceleration_data = []
        
        for track in scenario.tracks:
            prev_velocity = None
            prev_timestep = None
            for state in track.object_states:
                if state is not None:
                    agent_data.append([state.position[0], state.position[1], state.velocity[0], state.velocity[1]])
                    type_data.append(track.object_type.name)
                    time_data.append(state.timestep)
                    orientation = math.atan2(state.velocity[1], state.velocity[0]) if state.velocity[0] != 0 or state.velocity[1] != 0 else 0.0
                    orientation_data.append(orientation)
                    
                    if prev_velocity is not None and prev_timestep is not None:
                        delta_vx = state.velocity[0] - prev_velocity[0]
                        delta_vy = state.velocity[1] - prev_velocity[1]
                        delta_t = state.timestep - prev_timestep if state.timestep - prev_timestep > 0 else 1.0
                        ax = delta_vx / delta_t
                        ay = delta_vy / delta_t
                        acceleration_data.append([ax, ay])
                    else:
                        acceleration_data.append([0.0, 0.0])
                    
                    prev_velocity = state.velocity
                    prev_timestep = state.timestep
        
        if agent_data:
            traj_tensor = torch.tensor(agent_data, dtype=torch.float32, device=device)
            trajectories.append(traj_tensor)
            agent_types.extend(type_data)
            timestamps.extend(time_data)
            orientations.extend(orientation_data)
            accelerations.extend(acceleration_data)
            scenario_info['trajectory_rows'] = len(agent_data)
        
        road_vector_data = []
        road_attributes = []
        for lane_segment in map_api.vector_lane_segments.values():
            polygon = lane_segment.polygon_boundary
            start_point = polygon[0]
            end_point = polygon[-1]
            road_vector_data.append([start_point[0], start_point[1], end_point[0], end_point[1]])
            road_attributes.append({
                'lane_type': lane_segment.lane_type,
                'is_intersection': lane_segment.is_intersection
            })
        
        for edge_segment in map_api.vector_drivable_areas.values():
            boundary = edge_segment.area_boundary
            for i in range(len(boundary) - 1):
                start_point = boundary[i]
                end_point = boundary[i + 1]
                if isinstance(start_point, Point) and isinstance(end_point, Point):
                    road_edges.append([start_point.x, start_point.y, end_point.x, end_point.y])
        
        for crossing in map_api.vector_pedestrian_crossings.values():
            edge1_2d, edge2_2d = crossing.get_edges_2d()
            start_point1 = edge1_2d[0]
            end_point1 = edge1_2d[1]
            crosswalks.append([start_point1[0], start_point1[1], end_point1[0], end_point1[1]])
            start_point2 = edge2_2d[0]
            end_point2 = edge2_2d[1]
            crosswalks.append([start_point2[0], start_point2[1], end_point2[0], end_point2[1]])


        
        if road_vector_data:
            road_vector_tensor = torch.tensor(road_vector_data, dtype=torch.float32, device=device)
            road_vectors.append(road_vector_tensor)
            scenario_info['road_vector_segments'] = len(road_vector_data)
        
        scenario_info['road_attributes'] = road_attributes
        scenario_info['road_edges'] = len(road_edges)
        scenario_info['stop_signs'] = len(stop_signs)
        scenario_info['crosswalks'] = len(crosswalks)
        dataset_info.append(scenario_info)
        print(f"Processed scenario: {scenario.scenario_id}")
    
    trajectories_tensor = torch.cat(trajectories, dim=0) if trajectories else torch.empty((0,), device=device)
    road_vectors_tensor = torch.cat(road_vectors, dim=0) if road_vectors else torch.empty((0, 4), device=device)
    
    return trajectories_tensor, road_vectors_tensor, agent_types, timestamps, orientations, accelerations, road_edges, stop_signs, crosswalks, dataset_info

def save_combined_dataset(*args):
    (trajectories, road_vectors, agent_types, timestamps, orientations, accelerations, road_edges, stop_signs, crosswalks, dataset_info) = args
    for scenario in dataset_info:
        scenario_id = scenario['scenario_id']
        print(f"Saving scenario: {scenario_id}")
        combined_data = {
            'trajectories': trajectories.cpu().numpy(),
            'road_vectors': road_vectors.cpu().numpy(),
            'agent_types': agent_types,
            'timestamps': timestamps,
            'orientations': orientations,
            'accelerations': accelerations,
            'road_edges': road_edges,
            'stop_signs': stop_signs,
            'crosswalks': crosswalks,
            'metadata': scenario
        }
        with open(os.path.join(OUTPUT_PATH, f'{scenario_id}.pkl'), 'wb') as f:
            pickle.dump(combined_data, f)
        print(f"Saved {scenario_id}.pkl")

if __name__ == '__main__':
    save_combined_dataset(*load_dataset(DATASET_PATH))
