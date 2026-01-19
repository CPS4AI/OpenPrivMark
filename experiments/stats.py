# python examples/python/ml/watermarking_detection/stats.py --file examples/python/ml/watermarking_detection/resultsFalsePositive.jsonl

import json
import argparse


def proportionsFP(args):
    with open(args.file, 'r') as f:
        data = [json.loads(line) for line in f]

    total = len(data)
    detected_count = sum(1 for d in data if d["detected"])
    
    proportion_detected = detected_count / total if total > 0 else 0

    print(f"Detected: {detected_count}/{total}")
    print(f"Proportion detected: {proportion_detected*100}%")
    
def proportionsTP(args):
    with open(args.file, 'r') as f:
        data = [json.loads(line) for line in f]

    total = len(data)
    detected_original_count = sum(1 for d in data if d["detected_original"])
    detected_paraphrase_count = sum(1 for d in data if d["detected_paraphrase"])
    detected_removing_count = sum(1 for d in data if d["detected_removing"])
    
    proportion_detected_original = detected_original_count / total if total > 0 else 0
    proportion_detected_paraphrase = detected_paraphrase_count / total if total > 0 else 0
    proportion_detected_removing = detected_removing_count / total if total > 0 else 0
    
    print(f"Detected original: {detected_original_count}/{total}")
    print(f"Proportion detected original: {proportion_detected_original*100}%")
    print(f"Detected paraphrase: {detected_paraphrase_count}/{total}")
    print(f"Proportion detected paraphrase: {proportion_detected_paraphrase*100}%")
    print(f"Detected removing: {detected_removing_count}/{total}")
    print(f"Proportion detected removing: {proportion_detected_removing*100}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stats of JSONL results.')
    parser.add_argument('--file', type=str, required=True,
                   help='Path to input JSON file ')
    args = parser.parse_args()
    
    proportionsFP(args)
