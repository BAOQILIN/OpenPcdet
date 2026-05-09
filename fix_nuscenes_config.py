#!/usr/bin/env python
import yaml

# Read the config
with open('tools/cfgs/dataset_configs/nuscenes_dataset.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Fix DATA_PATH
config['DATA_PATH'] = 'data/nuscenes'

# Fix INFO_PATH
config['INFO_PATH'] = {
    'train': ['v1.0-trainval/nuscenes_infos_10sweeps_train.pkl'],
    'test': ['v1.0-trainval/nuscenes_infos_10sweeps_val.pkl']
}

# Fix gt_sampling in DATA_AUGMENTOR
for aug in config['DATA_AUGMENTOR']['AUG_CONFIG_LIST']:
    if aug['NAME'] == 'gt_sampling':
        aug['DB_INFO_PATH'] = ['v1.0-trainval/nuscenes_dbinfos_10sweeps_withvelo.pkl']
        aug['DB_DATA_PATH'] = ['v1.0-trainval/gt_database_10sweeps_withvelo']
        aug['BACKUP_DB_INFO'] = {
            'DB_INFO_PATH': 'v1.0-trainval/nuscenes_dbinfos_10sweeps_withvelo.pkl',
            'DB_DATA_PATH': 'v1.0-trainval/gt_database_10sweeps_withvelo',
            'NUM_POINT_FEATURES': 5
        }

# Write back
with open('tools/cfgs/dataset_configs/nuscenes_dataset.yaml', 'w') as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)

print("Config fixed successfully!")
