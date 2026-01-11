#!/usr/bin/env python3
"""
Regenerate visualizations with updated cost data
"""

import sys
sys.path.insert(0, '.')

import json
from pathlib import Path
from evaluation.visualization.report_generator import ReportGenerator

# Minimal config for visualization
config = {
    'visualization': {
        'style': 'seaborn-v0_8-darkgrid',
        'figure_size': [12, 8],
        'dpi': 300
    }
}

print('Loading evaluation results...')

# Load detailed results
with open('outputs/evaluation_results/detailed_per_template.json', 'r') as f:
    results = json.load(f)

print(f'Loaded {len(results)} results')

# Create report generator
generator = ReportGenerator(config)

# Generate visualizations
output_dir = Path('outputs/visualizations')
generator.generate_all_visualizations(results, output_dir)

print('\n✓ All visualizations regenerated successfully!')
print(f'\nVisualization files:')
for viz_file in sorted(output_dir.glob('*.png')):
    print(f'  - {viz_file.name}')
