import os
import torch
import pickle
import math
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
    
    observed_array = []
    timestep_array = []
    position_array = []
    heading_array = []
    velocity_array = []
    acceleration_array = []
    
    for folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        
        scenario_file = next((os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')), None)
        map_file = next((os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')), None)
        
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
        
        for track in scenario.tracks:
            prev_velocity = None
            prev_timestep = None
            for state in track.object_states:
                if state is None:
                    continue
                
                observed_array.append(state.observed)
                timestep_array.append(state.timestep)
                position_array.append([state.position[0], state.position[1]])
                heading_array.append(math.atan2(state.velocity[1], state.velocity[0]) if state.velocity[0] != 0 or state.velocity[1] != 0 else 0.0)
                velocity_array.append([state.velocity[0], state.velocity[1]])
                
                if prev_velocity and prev_timestep:
                    delta_vx = state.velocity[0] - prev_velocity[0]
                    delta_vy = state.velocity[1] - prev_velocity[1]
                    delta_t = max(state.timestep - prev_timestep, 1.0)
                    acceleration_array.append([delta_vx / delta_t, delta_vy / delta_t])
                else:
                    acceleration_array.append([0.0, 0.0])
                
                prev_velocity = state.velocity
                prev_timestep = state.timestep
        
        road_vector_data = [[lane.polygon_boundary[0][0], lane.polygon_boundary[0][1],
                             lane.polygon_boundary[-1][0], lane.polygon_boundary[-1][1]]
                             for lane in map_api.vector_lane_segments.values()]
        road_attributes = [{'lane_type': lane.lane_type, 'is_intersection': lane.is_intersection}
                           for lane in map_api.vector_lane_segments.values()]
        
        for edge in map_api.vector_drivable_areas.values():
            for i in range(len(edge.area_boundary) - 1):
                start, end = edge.area_boundary[i], edge.area_boundary[i + 1]
                if isinstance(start, Point) and isinstance(end, Point):
                    road_edges.append([start.x, start.y, end.x, end.y])
        
        for crossing in map_api.vector_pedestrian_crossings.values():
            edge1_2d, edge2_2d = crossing.get_edges_2d()
            crosswalks.append([edge1_2d[0][0], edge1_2d[0][1], edge1_2d[1][0], edge1_2d[1][1]])
            crosswalks.append([edge2_2d[0][0], edge2_2d[0][1], edge2_2d[1][0], edge2_2d[1][1]])
        
        if road_vector_data:
            road_vectors.append(torch.tensor(road_vector_data, dtype=torch.float32, device=device))
        
        scenario_info.update({
            'road_vector_segments': len(road_vector_data),
            'road_attributes': road_attributes,
            'road_edges': len(road_edges),
            'stop_signs': len(stop_signs),
            'crosswalks': len(crosswalks)
        })
        dataset_info.append(scenario_info)
        print(f"Processed scenario: {scenario.scenario_id}")
    
    return observed_array, timestep_array, position_array, heading_array, velocity_array, acceleration_array, road_vectors, road_edges, stop_signs, crosswalks, dataset_info

def save_combined_dataset(*args):
    metadata = args[10]
    
    for scenario in metadata:
        scenario_id = scenario.get('scenario_id', 'default')
        combined_data = {
            'observed': args[0],
            'timestep': args[1],
            'position': args[2],
            'heading': args[3],
            'velocity': args[4],
            'acceleration': args[5],
            'road_vectors': args[6],
            'road_edges': args[7],
            'stop_signs': args[8],
            'crosswalks': args[9],
            'metadata': scenario
        }
        
        output_file = os.path.join(OUTPUT_PATH, f'{scenario_id}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(combined_data, f)
        print(f"Dataset saved successfully as {scenario_id}.pkl")


if __name__ == '__main__':
    save_combined_dataset(*load_dataset(DATASET_PATH))
