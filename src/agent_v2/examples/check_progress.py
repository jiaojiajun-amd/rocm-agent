#!/usr/bin/env python3
"""Check training data generation progress and create resume dataset."""

import json
from pathlib import Path
from collections import defaultdict
import sys

def check_progress(dataset_path: Path, results_path: Path, samples_per_task: int = 32, max_tasks: int = None):
    """Check progress and return tasks that need more samples."""
    
    # Load original dataset
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    if max_tasks:
        dataset = dataset[:max_tasks]
        print(f"Limiting to first {max_tasks} tasks")
    
    # Try to load existing results
    task_success_count = defaultdict(int)
    total_examples = 0
    successful_examples = 0
    failed_examples = 0
    
    if results_path.exists():
        print(f"Loading existing results from {results_path}")
        try:
            # Try to load the full JSON
            with open(results_path) as f:
                results = json.load(f)
            
            for ex in results['examples']:
                total_examples += 1
                if ex['success']:
                    task_success_count[ex['instance_id']] += 1
                    successful_examples += 1
                else:
                    failed_examples += 1
            
            print(f"Successfully loaded {total_examples} examples")
            
        except json.JSONDecodeError as e:
            print(f"JSON file is corrupted: {e}")
            print("Attempting to parse line by line...")
            
            # Try to salvage data by parsing line by line
            with open(results_path) as f:
                content = f.read()
            
            # Try to find complete example entries
            import re
            # Look for complete example objects
            example_pattern = r'\{[^{}]*"instance_id":\s*"([^"]+)"[^{}]*"success":\s*(true|false)[^{}]*\}'
            
            for match in re.finditer(example_pattern, content):
                instance_id = match.group(1)
                success = match.group(2) == 'true'
                total_examples += 1
                if success:
                    task_success_count[instance_id] += 1
                    successful_examples += 1
                else:
                    failed_examples += 1
            
            if total_examples > 0:
                print(f"Salvaged {total_examples} examples from corrupted file")
            else:
                print("Could not parse any examples from corrupted file")
    else:
        print(f"No existing results found at {results_path}")
    
    # Print current progress
    print(f"\n{'='*70}")
    print(f"PROGRESS SUMMARY")
    print(f"{'='*70}")
    print(f"Total examples processed: {total_examples}")
    print(f"  Successful: {successful_examples}")
    print(f"  Failed: {failed_examples}")
    
    # Find tasks that need more samples (only counting successful ones)
    tasks_to_resume = []
    complete_tasks = []
    
    print(f"\n{'='*70}")
    print(f"TASK STATUS (target: {samples_per_task} successful samples per task)")
    print(f"{'='*70}")
    
    for task in dataset:
        task_id = task['instance_id']
        success_count = task_success_count[task_id]
        
        if success_count >= samples_per_task:
            complete_tasks.append(task)
            status = "✓ COMPLETE"
        else:
            tasks_to_resume.append(task)
            status = "✗ INCOMPLETE"
        
        print(f"{status:15} {task_id:60} {success_count:2}/{samples_per_task}")
    
    print(f"\n{'='*70}")
    print(f"Complete tasks: {len(complete_tasks)}/{len(dataset)}")
    print(f"Tasks to resume: {len(tasks_to_resume)}")
    print(f"Samples needed: {len(tasks_to_resume) * samples_per_task}")
    print(f"{'='*70}\n")
    
    return tasks_to_resume, task_success_count

def main():
    if len(sys.argv) < 3:
        print("Usage: check_progress.py <dataset.json> <results.json> [max_tasks] [samples_per_task]")
        print("\nExample:")
        print("  check_progress.py data/rocprim_v5.json training_data/large_dataset.json 29 32")
        sys.exit(1)
    
    dataset_path = Path(sys.argv[1])
    results_path = Path(sys.argv[2])
    max_tasks = int(sys.argv[3]) if len(sys.argv) > 3 else None
    samples_per_task = int(sys.argv[4]) if len(sys.argv) > 4 else 32
    
    tasks_to_resume, task_success_count = check_progress(
        dataset_path, results_path, samples_per_task, max_tasks
    )
    
    if tasks_to_resume:
        # Create resume dataset
        output_path = dataset_path.parent / f"{dataset_path.stem}_resume.json"
        with open(output_path, 'w') as f:
            json.dump(tasks_to_resume, f, indent=2)
        
        print(f"Resume dataset saved to: {output_path}")
        print(f"\nTo continue generation, use:")
        print(f"  --dataset {output_path}")
    else:
        print("All tasks are complete! No need to resume.")

if __name__ == '__main__':
    main()

