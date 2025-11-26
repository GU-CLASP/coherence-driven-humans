import argparse
import json
import os
import simpleamt
import sys
import time
from datetime import datetime

INTERVAL = 1800  # publish one HIT every 30 minutes

def create_output_dir(sandbox_true, base_dir='runs'):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M')
    output_dir = os.path.join(base_dir, f'{sandbox_true}-publish-{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def prompt_confirmation(num_hits, sandbox):
    env_name = 'SANDBOX' if sandbox else 'PRODUCTION'
    prompt = f'You are about to publish {num_hits} HIT(s) to the {env_name} environment. One HIT will be launched every 10 minutes. Continue? (Y/N): '
    confirm = input(prompt)
    return confirm.strip().lower() == 'y'

def main():
    parser = argparse.ArgumentParser(parents=[simpleamt.get_parent_parser()])
    parser.add_argument('--hit_properties_file', type=argparse.FileType('r'), required=True)
    parser.add_argument('--html_dir', type=str, required=True)
    args = parser.parse_args()

    mtc = simpleamt.get_mturk_connection_from_args(args)

    hit_properties = json.load(args.hit_properties_file)
    hit_properties['Reward'] = str(hit_properties['Reward'])
    frame_height = hit_properties.pop('FrameHeight')
    env = simpleamt.get_jinja_env(args.config)

    html_files = sorted([
        f for f in os.listdir(args.html_dir)
        if f.endswith('.html') and not f.startswith('.')
    ])

    if not html_files:
        print('No HTML files found in the directory.', file=sys.stderr)
        sys.exit(1)

    if not prompt_confirmation(len(html_files), args.sandbox):
        print('Aborted by user.')
        sys.exit(0)

    output_dir = create_output_dir(args.sandbox)
    hit_ids_path = os.path.join(output_dir, 'hit_ids.txt')
    print(f'Saving HIT IDs to: {hit_ids_path}')

    # Create the file immediately so monitor can access it
    open(hit_ids_path, 'a').close()

    for i, file in enumerate(html_files):
        html_path = os.path.join(args.html_dir, file)
        with open(html_path, 'r') as html_file:
            html_content = html_file.read()

        html_question = f'''
            <HTMLQuestion xmlns='http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd'>
              <HTMLContent><![CDATA[
                <!DOCTYPE html>
                {html_content}
              ]]></HTMLContent>
              <FrameHeight>{frame_height}</FrameHeight>
            </HTMLQuestion>
        '''
        hit_properties['Question'] = html_question

        print(f'[{datetime.now().strftime("%H:%M:%S")}] Publishing HIT {i+1}/{len(html_files)}: {file}')

        launched = False
        
        while not launched:
            try:
                boto_hit = mtc.create_hit(**hit_properties)
                launched = True
                hit_id = boto_hit['HIT']['HITId']
            except Exception as e:
                print(f'Error launching HIT: {e}', file=sys.stderr)

        with open(hit_ids_path, 'a') as hit_ids_file:
            hit_ids_file.write(f'{hit_id}\n')

        print(f'Published HIT ID: {hit_id}')

        if i < len(html_files) - 1:
            print(f'Waiting {INTERVAL // 60} minutes before next HIT...\n')
            time.sleep(INTERVAL)

    print('All HITs published.')

if __name__ == '__main__':
    main()
