#!/usr/bin/env python3
"""Merge original and retry datasets, keeping best examples for each task."""

import json
from pathlib import Path
from collections import defaultdict

def merge_datasets(original_path: Path, retry_path: Path, output_path: Path):
    """Merge two training datasets, prioritizing successful examples."""
    
    with open(original_path) as f:
        original_data = json.load(f)
    
    with open(retry_path) as f:
        retry_data = json.load(f)
    
    # Combine all examples
    all_examples = original_data['examples'] + retry_data['examples']
    
    # Group by (instance_id, sample_id) and keep successful ones
    example_map = {}
    for ex in all_examples:
        key = (ex['instance_id'], ex['sample_id'])
        # Keep the example if it's the first one or if it's successful
        if key not in example_map or ex['success']:
            example_map[key] = ex
    
    merged_examples = list(example_map.values())
    
    # Calculate statistics
    task_stats = defaultdict(lambda: {'success': 0, 'failed': 0})
    for ex in merged_examples:
        task = ex['instance_id']
        if ex['success']:
            task_stats[task]['success'] += 1
        else:
            task_stats[task]['failed'] += 1
    
    # Create merged dataset
    merged_data = {
        'metadata': {
            **original_data['metadata'],
            'merged_from': [str(original_path), str(retry_path)],
            'merge_timestamp': None,  # Could add timestamp
        },
        'examples': merged_examples,
        'summary': {
            'total_examples': len(merged_examples),
            'successful': sum(1 for ex in merged_examples if ex['success']),
            'failed': sum(1 for ex in merged_examples if not ex['success']),
            'average_reward': sum(ex['reward'] for ex in merged_examples) / len(merged_examples) if merged_examples else 0,
            'total_model_calls': sum(ex['model_calls'] for ex in merged_examples),
        }
    }
    
    # Save merged dataset
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
    
    print(f"Merged dataset saved to {output_path}")
    print(f"\nMerge Statistics:")
    print(f"  Original examples: {len(original_data['examples'])}")
    print(f"  Retry examples: {len(retry_data['examples'])}")
    print(f"  Merged examples: {len(merged_examples)}")
    print(f"  Successful: {merged_data['summary']['successful']}")
    print(f"  Failed: {merged_data['summary']['failed']}")
    print(f"  Success rate: {merged_data['summary']['successful']/len(merged_examples)*100:.1f}%")
    
    print(f"\nTask-level summary:")
    samples_per_task = original_data['metadata']['samples_per_task']
    complete_tasks = 0
    for task, stats in sorted(task_stats.items()):
        total = stats['success'] + stats['failed']
        if stats['success'] == samples_per_task:
            complete_tasks += 1
            status = "✓"
        else:
            status = "✗"
        print(f"  {status} {task}: {stats['success']}/{total} successful")
    
    print(f"\nComplete tasks: {complete_tasks}/{len(task_stats)}")

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 4:
        print("Usage: merge_datasets.py <original.json> <retry.json> <output.json>")
        sys.exit(1)
    
    original = Path(sys.argv[1])
    retry = Path(sys.argv[2])
    output = Path(sys.argv[3])
    
    merge_datasets(original, retry, output)

