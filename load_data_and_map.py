import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

r = 3

def load_data(dataset_index):
    dataset_index = 6
    file_path = f'Congestion dynamics in single-file motion\PHAS_SEP\PHAS_SEP\PHASE_SEP_{dataset_index}_pos.csv'
    data = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=0, usecols=[0, 1, 2, 3])

    data.columns = ['id', 'frame', 'x', 'y']
    data = data.astype({'id': int, 'frame': int, 'x': float, 'y': float})

    return data

def map_to_path(x,y):
    r = 3
    if -2 <= y <= 2 and x >= 0:
        x_coord = r
        y_coord = y
        return x_coord, y_coord
    elif y > 2:
        x_coord = x
        y_coord = y - 2
        factor = r / np.sqrt(x_coord ** 2 + y_coord ** 2)
        x_coord *= factor
        y_coord *= factor
        return x_coord, y_coord+2
    else:
        x_temp, y_temp = map_to_path(-x, -y)
        return -x_temp, -y_temp


def map_data(data):
    x_mapped = []
    y_mapped = []
    data['y'] = data['y'] - 1
    for person_id in data['id'].unique():
        data_person = data[data['id'] == person_id]
        mapped_coords = data_person.apply(lambda row: map_to_path(row['x'], row['y']), axis=1)
        x_mapped_person, y_mapped_person = zip(*mapped_coords)
        x_mapped.append(x_mapped_person)
        y_mapped.append(y_mapped_person)

    data['x_mapped'] = np.concatenate(x_mapped)
    data['y_mapped'] = np.concatenate(y_mapped)

    return data

def compute_position(x,y):
    if x >= 0 and 0 <= y <= 2:
        return y
    elif y > 2:
        return 2 + np.arctan2(y-2, x) * r
    elif x < 0 and -2 <= y <= 2:
        return 2 + np.pi * r + (2 - y)
    elif y < -2:
        return 2 + np.pi * r + 4 + np.arctan2(-(y + 2), -x) * r
    else:
        return 2 + 2*np.pi * r + 4 + y + 2
    
def reset_laps(person_position):
    lap_positions = []
    for index in range(len(person_position)):
        lap_positions.append(person_position[index])
        if index < len(person_position) -1 and (person_position[index] > person_position[index + 1] + 20 or person_position[index] < person_position[index + 1] - 20):
            lap_positions.append(np.nan)
    return lap_positions

def plot_trajectory(dataset_index):
    data = load_data(dataset_index)
    data = map_data(data)

    data['position'] = data.apply(lambda row: compute_position(row['x_mapped'], row['y_mapped']), axis=1)
    plt.figure(figsize=(12, 8))
    for person_id, person_data in data.groupby('id'):
        position_list = reset_laps(person_data['position'].tolist())
        
        new_frame_list = list(range(len(position_list)))
        plt.plot(position_list, new_frame_list, label=f'Person {person_id}')

    plt.title(f'Dataset {dataset_index} 1D Trajectory: Position along the Trajectory vs. Frame')
    plt.xlabel('Position along the Trajectory (from (3, 0))')
    plt.ylabel('Frame')
    plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', title="Person ID")
    plt.tight_layout()


    plt.show()

if __name__ == "__main__":
    plot_trajectory(1)