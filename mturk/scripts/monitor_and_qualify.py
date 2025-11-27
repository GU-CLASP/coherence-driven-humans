import argparse, json, os, sys, time, glob
import pandas as pd
import simpleamt
from xml.etree import ElementTree as ET

QUALIFICATION_TYPE_ID = '3VIW7BLKQFCIG5E6CXLL1HDYHN07PT'
MAX_TASKS_PER_WORKER = 6
CHECK_INTERVAL = 30

def parse_xml_answer(xml_str):
    ns = {'ns': 'http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2005-10-01/QuestionFormAnswers.xsd'}
    try:
        root = ET.fromstring(xml_str)
        parsed = {}
        for ans in root.findall('ns:Answer', ns):
            key = ans.find('ns:QuestionIdentifier', ns).text
            value = ans.find('ns:FreeText', ns).text or ''
            parsed[key] = value.strip()
        return parsed
    except ET.ParseError as e:
        print(f'XML ParseError: {e}', file=sys.stderr)
        return {}

def process_assignments(mtc, hit_id, status):
    results = []
    paginator = mtc.get_paginator('list_assignments_for_hit')
    try:
        for a_page in paginator.paginate(HITId=hit_id, PaginationConfig={'PageSize': 100}):
            for a in a_page['Assignments']:
                if a['AssignmentStatus'] not in status:
                    continue

                answers = parse_xml_answer(a['Answer'])
                story_id = answers.get('storyId', '')
                described_dialogue = answers.get('describedDialogue', '')
                image_familiarity = answers.get('imageFamiliarity', '')
                skipped_images = answers.get('skippedImages', '')
                story_ease = answers.get('storyEase', '')
                descs = [answers.get(f'desc{i}', '') for i in range(10)]

                result = {
                    'assignment_id': a['AssignmentId'],
                    'hit_id': hit_id,
                    'worker_id': a['WorkerId'],
                    'story_id': story_id,
                    'describedDialogue': described_dialogue,
                    'imageFamiliarity': image_familiarity,
                    'skippedImages': skipped_images,
                    'storyEase': story_ease
                }

                for i, desc in enumerate(descs):
                    result[f'text{i}'] = desc

                results.append(result)
    except mtc.exceptions.RequestError:
        print(f'Bad hit_id {hit_id}', file=sys.stderr)
        return results

    return results

def get_all_worker_counts(previous_run_folders, current_hit_ids, mtc, status):
    """Get total worker counts from all previous runs + current HITs"""
    all_worker_counts = {}
    
    # Load previous runs
    for folder in previous_run_folders:
        counts_file = os.path.join(folder, 'worker_counts.csv')
        if os.path.exists(counts_file):
            df = pd.read_csv(counts_file)
            print(f'Loaded {len(df)} workers from {folder}')
            for _, row in df.iterrows():
                worker_id = row['worker_id']
                count = row['count']
                all_worker_counts[worker_id] = all_worker_counts.get(worker_id, 0) + count
    
    # Add current HITs
    for hit_id in current_hit_ids:
        results = process_assignments(mtc, hit_id, status)
        for result in results:
            worker_id = result['worker_id']
            all_worker_counts[worker_id] = all_worker_counts.get(worker_id, 0) + 1

    #print(all_worker_counts)
    
    return all_worker_counts

def assign_blocking_qualification(mtc, worker_id):
    """Assign blocking qualification to a worker"""
    try:
        # Check if already has qualification
        response = mtc.get_qualification_score(
            QualificationTypeId=QUALIFICATION_TYPE_ID,
            WorkerId=worker_id
        )
        if 'Qualification' in response and response['Qualification'].get('IntegerValue') == 1:
            return False  # Already has it
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
        print(f'Error assigning qualification to {worker_id}: {e}', file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(parents=[simpleamt.get_parent_parser()])
    parser.add_argument('--rejected', action='store_true', default=False)
    parser.add_argument('--previous-runs', nargs='*', help='Previous run folders (will be filtered to False-publish-2025* pattern)')
    parser.add_argument('--runs-dir', help='Directory containing the current run')
    parser.add_argument('--max-tasks', type=int, default=MAX_TASKS_PER_WORKER)
    args = parser.parse_args()

    mtc = simpleamt.get_mturk_connection_from_args(args)
    status = ['Approved', 'Submitted']
    if args.rejected:
        status.append('Rejected')

    # Get previous run folders
    previous_run_folders = []
    if args.previous_runs:
        for folder in args.previous_runs:
            if os.path.isdir(folder):
                # If it's a directory that might contain run folders, scan it
                if folder.endswith('runs') or folder.endswith('runs/'):
                    # Scan for False-publish-2025-10* folders within this directory
                    pattern = os.path.join(folder, 'False-publish-2025*')
                    discovered = [f for f in glob.glob(pattern) if os.path.isdir(f)]
                    previous_run_folders.extend(discovered)
                    print(f"Auto-discovered {len(discovered)} folders in {folder}: {discovered}")
                else:
                    # Treat as individual run folder
                    previous_run_folders.append(folder)
    
    # Get current run directory (to exclude from previous runs)
    hit_ids_path = args.hit_ids_file
    if not hit_ids_path:
        print("Error: --hit_ids_file is required")
        sys.exit(1)
    
    current_run_dir = os.path.dirname(hit_ids_path)
    current_run_dir = os.path.abspath(current_run_dir)  # Make absolute for comparison
    
    # Filter to only include folders matching the False-publish-2025-10* pattern
    # AND exclude the current run directory
    filtered_folders = []
    for folder in previous_run_folders:
        folder_abs = os.path.abspath(folder)  # Make absolute for comparison
        folder_name = os.path.basename(folder)
        
        if folder_abs == current_run_dir:
            print(f"Skipping current run folder: {folder}")
        elif folder_name.startswith('False-publish-2025'):
            filtered_folders.append(folder)
        else:
            print(f"Skipping folder (doesn't match pattern): {folder}")
    
    previous_run_folders = filtered_folders
    
    # Remove current folder from previous runs (double-check)
    if current_run_dir in previous_run_folders:
        previous_run_folders.remove(current_run_dir)
    
    print(f"Monitoring HITs from: {hit_ids_path}")
    print(f"Current run directory: {current_run_dir}")
    print(f"Previous runs: {previous_run_folders}")
    print(f"Max tasks per worker: {args.max_tasks}")

    seen_assignments = set()
    qualified_workers = set()

    while True:
        print('\nChecking HIT results...')
        
        # Read current HITs
        with open(hit_ids_path, 'r') as f:
            hit_ids = [line.strip() for line in f if line.strip()]
        
        # Get all worker counts (previous + current)
        all_worker_counts = get_all_worker_counts(previous_run_folders, hit_ids, mtc, status)
        
        # Process current assignments for results file
        all_results = []
        for hit_id in hit_ids:
            results = process_assignments(mtc, hit_id, status)
            all_results.extend(results)
        
        if all_results:
            df = pd.DataFrame(all_results)
            new_assignments = df[~df['assignment_id'].isin(seen_assignments)]
            
            if not new_assignments.empty:
                seen_assignments.update(new_assignments['assignment_id'].tolist())
                
                # Save new results
                results_path = os.path.join(current_run_dir, 'results.csv')
                if os.path.exists(results_path):
                    existing = pd.read_csv(results_path)
                    combined = pd.concat([existing, new_assignments], ignore_index=True)
                    combined.drop_duplicates(subset='assignment_id', inplace=True)
                else:
                    combined = new_assignments
                
                combined.to_csv(results_path, index=False)
                print(f'Saved {len(new_assignments)} new assignments')
                
                # Update current run worker counts
                current_counts = df['worker_id'].value_counts().reset_index()
                current_counts.columns = ['worker_id', 'count']
                counts_path = os.path.join(current_run_dir, 'worker_counts.csv')
                current_counts.to_csv(counts_path, index=False)
        
        # Check for workers who need blocking
        workers_to_block = [worker_id for worker_id, count in all_worker_counts.items() 
                           if count >= args.max_tasks and worker_id not in qualified_workers]
        
        print(f"Total workers: {len(all_worker_counts)}")
        print(f"Workers with {args.max_tasks}+ tasks: {len([c for c in all_worker_counts.values() if c >= args.max_tasks])}")
        
        if workers_to_block:
            print("Blocking workers:")
            for worker_id in workers_to_block:
                count = all_worker_counts[worker_id]
                if assign_blocking_qualification(mtc, worker_id):
                    print(f"  BLOCKED {worker_id} ({count} tasks)")
                    qualified_workers.add(worker_id)
                else:
                    print(f"  Already blocked: {worker_id} ({count} tasks)")
                    qualified_workers.add(worker_id)
        
        print(f'Waiting {CHECK_INTERVAL} seconds...\n')
        time.sleep(CHECK_INTERVAL)

if __name__ == '__main__':
    main()