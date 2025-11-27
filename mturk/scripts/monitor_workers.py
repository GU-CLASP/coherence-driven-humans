#!/usr/bin/env python3
"""
Simple monitoring script that checks all runs every 30 seconds and blocks workers with 5+ tasks.

How it works:
1. Every 30 seconds, scans ALL run folders (False-publish-*)
2. Counts total tasks per worker across ALL runs
3. Blocks any worker who has completed 5 or more tasks
4. Keeps running continuously
"""

import argparse
import os
import sys
import time
import glob
import pandas as pd
import simpleamt
from xml.etree import ElementTree as ET

QUALIFICATION_TYPE_ID = '3VIW7BLKQFCIG5E6CXLL1HDYHN07PT'
MAX_TASKS_PER_WORKER = 6
CHECK_INTERVAL = 30  # seconds
RUNS_DIR = '../data/results'

def get_all_run_folders(runs_dir):
    """Find all run folders matching the pattern."""
    pattern = os.path.join(runs_dir, 'False-publish-*')
    folders = [f for f in glob.glob(pattern) if os.path.isdir(f)]
    return sorted(folders)

def process_assignments(mtc, hit_id, status):
    """Get all assignments for a HIT from MTurk."""
    results = []
    paginator = mtc.get_paginator('list_assignments_for_hit')
    try:
        for a_page in paginator.paginate(HITId=hit_id, PaginationConfig={'PageSize': 100}):
            for a in a_page['Assignments']:
                if a['AssignmentStatus'] not in status:
                    continue
                
                # Just extract worker_id for counting
                results.append({
                    'worker_id': a['WorkerId'],
                    'assignment_id': a['AssignmentId'],
                    'hit_id': hit_id
                })
    except Exception as e:
        print(f'  Warning: Error processing HIT {hit_id}: {e}', file=sys.stderr)
        return results
    
    return results

def count_all_workers(mtc, status, runs_dir):
    """Count total tasks per worker across ALL runs by querying MTurk directly."""
    worker_counts = {}
    
    run_folders = get_all_run_folders(runs_dir)
    print(f'Found {len(run_folders)} run folders')
    
    for folder in run_folders:
        hit_ids_file = os.path.join(folder, 'hit_ids.txt')
        
        if not os.path.exists(hit_ids_file):
            folder_name = os.path.basename(folder)
            print(f'  {folder_name}: No hit_ids.txt found, skipping')
            continue
        
        # Read HIT IDs from file
        with open(hit_ids_file, 'r') as f:
            hit_ids = [line.strip() for line in f if line.strip()]
        
        if not hit_ids:
            continue
        
        folder_name = os.path.basename(folder)
        print(f'  {folder_name}: Processing {len(hit_ids)} HITs...')
        
        # Get assignments for all HITs in this folder
        folder_assignments = []
        for hit_id in hit_ids:
            assignments = process_assignments(mtc, hit_id, status)
            folder_assignments.extend(assignments)
        
        # Count workers in this folder
        folder_worker_counts = {}
        for assignment in folder_assignments:
            worker_id = assignment['worker_id']
            folder_worker_counts[worker_id] = folder_worker_counts.get(worker_id, 0) + 1
        
        print(f'    Found {len(folder_assignments)} assignments from {len(folder_worker_counts)} workers')
        
        # Add to overall counts
        for worker_id, count in folder_worker_counts.items():
            worker_counts[worker_id] = worker_counts.get(worker_id, 0) + count
    
    return worker_counts

def block_worker(mtc, worker_id):
    """Assign blocking qualification to a worker. Returns True if newly blocked."""
    try:
        # Check if already has qualification
        response = mtc.get_qualification_score(
            QualificationTypeId=QUALIFICATION_TYPE_ID,
            WorkerId=worker_id
        )
        if 'Qualification' in response and response['Qualification'].get('IntegerValue') == 1:
            return False  # Already blocked
    except:
        pass  # Doesn't have it yet
    
    try:
        mtc.associate_qualification_with_worker(
            QualificationTypeId=QUALIFICATION_TYPE_ID,
            WorkerId=worker_id,
            IntegerValue=1,
            SendNotification=False
        )
        return True
    except Exception as e:
        print(f'    Error blocking {worker_id}: {e}', file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Monitor all runs and block workers with 5+ tasks',
        parents=[simpleamt.get_parent_parser()]
    )
    parser.add_argument('--max-tasks', type=int, default=MAX_TASKS_PER_WORKER,
                       help=f'Maximum tasks before blocking (default: {MAX_TASKS_PER_WORKER})')
    parser.add_argument('--runs-dir', default=RUNS_DIR,
                       help=f'Directory containing run folders (default: {RUNS_DIR})')
    parser.add_argument('--interval', type=int, default=CHECK_INTERVAL,
                       help=f'Check interval in seconds (default: {CHECK_INTERVAL})')
    parser.add_argument('--rejected', action='store_true', default=False,
                       help='Include rejected assignments in counts')
    
    args = parser.parse_args()
    
    # Use local variables instead of modifying globals
    runs_dir = args.runs_dir
    check_interval = args.interval
    max_tasks = args.max_tasks
    
    # Assignment statuses to include
    status = ['Approved', 'Submitted']
    if args.rejected:
        status.append('Rejected')
    
    # Connect to MTurk
    mtc = simpleamt.get_mturk_connection_from_args(args)
    
    print(f'\n{"="*70}')
    print(f'Simple Worker Monitor')
    print(f'{"="*70}')
    print(f'Runs directory: {runs_dir}')
    print(f'Max tasks per worker: {max_tasks}')
    print(f'Check interval: {check_interval} seconds')
    print(f'Assignment statuses: {", ".join(status)}')
    print(f'Blocking qualification: {QUALIFICATION_TYPE_ID}')
    print(f'{"="*70}\n')
    
    blocked_workers = set()  # Track workers we've already blocked
    
    print('Starting monitoring loop... (Press Ctrl+C to stop)\n')
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f'{"="*70}')
            print(f'Check #{iteration} - {time.strftime("%Y-%m-%d %H:%M:%S")}')
            print(f'{"="*70}')
            
            # Count all workers across all runs
            worker_counts = count_all_workers(mtc, status, runs_dir)
            
            if not worker_counts:
                print('No workers found yet')
            else:
                print(f'\nTotal unique workers: {len(worker_counts)}')
                
                # Find workers who need to be blocked
                workers_needing_block = {
                    wid: count for wid, count in worker_counts.items() 
                    if count >= max_tasks
                }
                
                print(f'Workers with {max_tasks}+ tasks: {len(workers_needing_block)}')
                print(f'Workers with < {max_tasks} tasks: {len(worker_counts) - len(workers_needing_block)}')
                
                # Block workers who need it
                if workers_needing_block:
                    print(f'\nProcessing workers with {max_tasks}+ tasks:')
                    
                    newly_blocked = 0
                    already_blocked = 0
                    
                    for worker_id, count in sorted(workers_needing_block.items(), 
                                                   key=lambda x: x[1], reverse=True):
                        if worker_id in blocked_workers:
                            # We already blocked this worker in a previous iteration
                            already_blocked += 1
                            continue
                        
                        if block_worker(mtc, worker_id):
                            print(f'  ✓ BLOCKED {worker_id} ({count} tasks)')
                            blocked_workers.add(worker_id)
                            newly_blocked += 1
                        else:
                            print(f'  ✓ Already blocked: {worker_id} ({count} tasks)')
                            blocked_workers.add(worker_id)
                            already_blocked += 1
                    
                    print(f'\n  Summary:')
                    print(f'    Newly blocked: {newly_blocked}')
                    print(f'    Already blocked: {already_blocked}')
                    print(f'    Total blocked workers: {len(blocked_workers)}')
                else:
                    print('\nNo workers need blocking yet')
                    
                # Show top workers (for monitoring)
                print(f'\nTop 10 workers by task count:')
                top_workers = sorted(worker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                for worker_id, count in top_workers:
                    status_icon = '🚫' if worker_id in blocked_workers else '✅'
                    print(f'  {status_icon} {worker_id}: {count} tasks')
            
            print(f'\nNext check in {check_interval} seconds...\n')
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print(f'\n\n{"="*70}')
        print('Monitoring stopped by user')
        print(f'{"="*70}')
        print(f'Final stats:')
        print(f'  Total workers monitored: {len(worker_counts) if worker_counts else 0}')
        print(f'  Workers blocked: {len(blocked_workers)}')
        print(f'{"="*70}\n')
        sys.exit(0)

if __name__ == '__main__':
    main()
