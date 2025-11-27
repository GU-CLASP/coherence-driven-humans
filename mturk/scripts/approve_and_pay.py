import argparse
import sys
import os
import pandas as pd
import simpleamt

def count_words(text_list):
    total = 0
    for text in text_list:
        if isinstance(text, str) and text.strip():
            total += len(text.strip().split())
    return total

def main():
    parser = argparse.ArgumentParser(parents=[simpleamt.get_parent_parser()])
    parser.add_argument('--results_file', required=True, help='Path to CSV file with assignment results')
    args = parser.parse_args()

    results_path = args.results_file
    results_dir = os.path.dirname(results_path)
    decisions_path = os.path.join(results_dir, 'decisions.csv')

    df = pd.read_csv(results_path)
    mtc = simpleamt.get_mturk_connection_from_args(args)

    decisions = []
    default_feedback = 'Thank you for your high-quality work!'

    for _, row in df.iterrows():
        assignment_id = row['assignment_id']
        worker_id = row['worker_id']

        try:
            assignment_data = mtc.get_assignment(AssignmentId=assignment_id)['Assignment']
            status = assignment_data['AssignmentStatus']
        except Exception as e:
            print(f'Failed to get status for assignment {assignment_id}: {e}', file=sys.stderr)
            continue

        if status in ['Approved', 'Rejected']:
            decision_type = 'approve' if status == 'Approved' else 'reject'
            feedback = assignment_data.get('RequesterFeedback', '')
            if not feedback and decision_type == 'approve':
                feedback = default_feedback
            print(f'\nAssignment {assignment_id} already {status}. Logging and skipping.')
            decisions.append({**row, 'decision': decision_type, 'bonus': 0.0, 'decision_feedback': feedback})
            # now move again to the tasks which were submitted, but did not get a decision yet
            continue

        text_cols = [f'text{i}' for i in range(10) if f'text{i}' in row and pd.notna(row[f'text{i}'])]
        texts = [row[col] for col in text_cols if isinstance(row[col], str) and row[col].strip()]
        word_count = count_words(texts)
        full_text = '\n'.join(texts)

        print('\n' + '-' * 60)
        print(f'Assignment ID: {assignment_id}')
        print(f'Worker ID: {worker_id}')
        print(f'Word count: {word_count}')
        print('Combined text:')
        print(full_text)
        print('-' * 60)
        print(f'Described dialogue: {row.get("describedDialogue", "")}')
        print(f'Image familiarity: {row.get("imageFamiliarity", "")}')
        print(f'Skipped images: {row.get("skippedImages", "")}')
        print(f'Story ease: {row.get("storyEase", "")}')
        print('-' * 60)

        approve = input('Approve this HIT? (Y/N): ').strip().lower()
        if approve == 'y':
            bonus = round((word_count - 85) * 0.04, 2) if word_count > 85 else 0.0
            if bonus > 0:
                print(f'Eligible bonus: ${bonus:.2f} for {word_count - 85} extra words.')
                send_bonus = input(f'Send bonus of ${bonus:.2f}? (Y/N): ').strip().lower()
            else:
                send_bonus = 'n'

            try:
                mtc.approve_assignment(AssignmentId=assignment_id)
                print(f'Approved assignment {assignment_id}')
                feedback = default_feedback
            except Exception as e:
                print(f'Failed to approve: {e}', file=sys.stderr)
                continue

            if bonus > 0 and send_bonus == 'y':
                try:
                    mtc.send_bonus(
                        WorkerId=worker_id,
                        BonusAmount=f'{bonus:.2f}',
                        AssignmentId=assignment_id,
                        Reason=feedback
                    )
                    print(f'Sent ${bonus:.2f} bonus to {worker_id}')
                except Exception as e:
                    print(f'Failed to send bonus: {e}', file=sys.stderr)
                    bonus = 0.0
            else:
                bonus = 0.0

            decisions.append({**row, 'decision': 'approve', 'bonus': bonus, 'decision_feedback': feedback})

        else:
            reject = input('Reject this HIT instead? (Y/N): ').strip().lower()
            if reject == 'y':
                reason = input('Enter rejection reason: ').strip()
                try:
                    mtc.reject_assignment(
                        AssignmentId=assignment_id,
                        RequesterFeedback=reason
                    )
                    print(f'Rejected assignment {assignment_id}')
                    decisions.append({**row, 'decision': 'reject', 'bonus': 0.0, 'decision_feedback': reason})
                except Exception as e:
                    print(f'Failed to reject: {e}', file=sys.stderr)
            else:
                print('Skipped.')

    if decisions:
        print(f'\nWriting decisions to: {decisions_path}')
        decisions_df = pd.DataFrame(decisions)
        decisions_df.to_csv(decisions_path, index=False)
        print('Done.')

if __name__ == '__main__':
    main()
