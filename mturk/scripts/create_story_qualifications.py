#!/usr/bin/env python3
"""
Create MTurk qualifications for blocking workers from specific stories.
Edit the STORY_IDS list below with the stories that need qualifications.
"""

import argparse
import sys
import simpleamt

STORY_IDS = [
    1214,
    5047,
    6111,
    6729,
    6847,
    6878,
    7702,
    8551,
    11260,
    11340,
    12358,
    13683,
]

def create_qualification(mtc, story_id, dry_run=False):
    """
    Create a blocking qualification for a specific story.
    Returns the qualification type ID if successful.
    """
    qual_name = f'BlockStory{story_id}'
    qual_description = f'Workers who cannot work on story {story_id} (already completed or have 5+ HITs)'
    
    if dry_run:
        print(f'  [DRY RUN] Would create qualification: {qual_name}')
        return None
    
    try:
        response = mtc.create_qualification_type(
            Name=qual_name,
            Description=qual_description,
            QualificationTypeStatus='Active',
            AutoGranted=False
        )
        
        qual_type_id = response['QualificationType']['QualificationTypeId']
        print(f'  ✓ Created: {qual_name}')
        print(f'    ID: {qual_type_id}')
        return qual_type_id
        
    except Exception as e:
        print(f'  ✗ Error creating qualification for story {story_id}: {e}', file=sys.stderr)
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        parents=[simpleamt.get_parent_parser()],
        description='Create MTurk qualifications for story-specific worker blocking'
    )
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    # Validate
    if not STORY_IDS:
        print('ERROR: STORY_IDS list is empty!')
        print('Edit this script and add story IDs to the STORY_IDS list.')
        sys.exit(1)
    
    # Connect to MTurk
    print(f'\n{"="*70}')
    print(f'Stories to create qualifications for: {len(STORY_IDS)}')
    
    if args.dry_run:
        print(f'MODE: DRY RUN (no qualifications will be created)')
    else:
        print(f'MODE: LIVE (will create qualifications)')
    print(f'{"="*70}\n')
    
    mtc = simpleamt.get_mturk_connection_from_args(args)
    
    # Create qualifications
    print('Creating qualifications:\n')
    created_qualifications = {}
    
    for story_id in STORY_IDS:
        qual_id = create_qualification(mtc, story_id, dry_run=args.dry_run)
        if qual_id:
            created_qualifications[story_id] = qual_id
    
    # Summary
    print(f'\n{"="*70}')
    print(f'SUMMARY:')
    print(f'  Qualifications created: {len(created_qualifications)}')
    print(f'  Failed: {len(STORY_IDS) - len(created_qualifications)}')
    
    if created_qualifications and not args.dry_run:
        print(f'\n  Created qualification IDs:')
        for story_id, qual_id in sorted(created_qualifications.items()):
            print(f'    Story {story_id}: {qual_id}')
        
        # Save to file for later use
        output_file = 'story_qualification_ids.txt'
        with open(output_file, 'w') as f:
            for story_id, qual_id in sorted(created_qualifications.items()):
                f.write(f'{story_id},{qual_id}\n')
        print(f'\n  Saved qualification IDs to: {output_file}')
    
    print(f'{"="*70}\n')
