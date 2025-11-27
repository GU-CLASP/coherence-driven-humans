#!/usr/bin/env python3
"""
Assign blocking qualifications to specific workers.
Edit the WORKERS_TO_BLOCK list below with worker IDs you want to block from all HITs.
"""

import argparse
import sys
import simpleamt

WORKERS_TO_BLOCK = [
  'A35BY30TC8WCL4',
  'A26UIS59SY4NM6',
  'A3PJ51GS2AKBO6',
  'A1IZ4NX41GKU4X',
  'A3O81LHBBI8NPK',
  'A3PYB8Z6FFWSOV',
  'A2EI075XZT9Y2S',
  'A2VE5IV9OD2SK1',
  'AMG9Y1YLBTKIV',
  'A2WCCV1W8UE8ED',
  'A29VL3MZE7YPBZ',
  'AVPKE76DJLWK6'
]

QUALIFICATION_TYPE_ID = '3VIW7BLKQFCIG5E6CXLL1HDYHN07PT'

def block_worker(mtc, worker_id, dry_run=False):
    """
    Assign blocking qualification to a worker.
    Returns True if newly blocked, False if already blocked.
    """
    # Check if already blocked
    try:
        response = mtc.get_qualification_score(
            QualificationTypeId=QUALIFICATION_TYPE_ID,
            WorkerId=worker_id
        )
        if 'Qualification' in response and response['Qualification'].get('IntegerValue') == 1:
            print(f'  ✓ {worker_id} - Already blocked')
            return False
    except:
        pass  # Doesn't have qualification yet
    
    if dry_run:
        print(f'  [DRY RUN] Would block: {worker_id}')
        return True
    
    try:
        mtc.associate_qualification_with_worker(
            QualificationTypeId=QUALIFICATION_TYPE_ID,
            WorkerId=worker_id,
            IntegerValue=1,
            SendNotification=False
        )
        print(f'  ✓ {worker_id} - BLOCKED')
        return True
    except Exception as e:
        print(f'  ✗ {worker_id} - Error: {e}', file=sys.stderr)
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        parents=[simpleamt.get_parent_parser()],
        description='Block workers listed in WORKERS_TO_BLOCK'
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    # Validate
    if not WORKERS_TO_BLOCK:
        print('ERROR: WORKERS_TO_BLOCK list is empty!')
        print('Edit this script and add worker IDs to the WORKERS_TO_BLOCK list.')
        sys.exit(1)
    
    # Connect to MTurk
    print(f'\n{"="*70}')
    print(f'Qualification ID: {QUALIFICATION_TYPE_ID}')
    print(f'Workers to process: {len(WORKERS_TO_BLOCK)}')
    
    if args.dry_run:
        print(f'MODE: DRY RUN (no changes will be made)')
    else:
        print(f'MODE: LIVE (will assign blocking qualifications)')
    print(f'{"="*70}\n')
    
    mtc = simpleamt.get_mturk_connection_from_args(args)
    
    # Block workers
    print('Processing workers:\n')
    blocked_count = 0
    already_blocked_count = 0
    
    for worker_id in WORKERS_TO_BLOCK:
        if block_worker(mtc, worker_id, dry_run=args.dry_run):
            blocked_count += 1
        else:
            already_blocked_count += 1
    
    # Summary
    print(f'\n{"="*70}')
    print(f'SUMMARY:')
    print(f'  Newly blocked: {blocked_count}')
    print(f'  Already blocked: {already_blocked_count}')
    print(f'  Total processed: {len(WORKERS_TO_BLOCK)}')
    print(f'{"="*70}\n')
