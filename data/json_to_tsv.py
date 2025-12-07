import json
import csv
import os

# Define your input files (add 500, 1000, 2000 as needed)
INPUT_FILES = [
    "/home/ezrah/CS598_DLH/project/SleepQA/data/training/sleep-train-100.json",
    "/home/ezrah/CS598_DLH/project/SleepQA/data/training/sleep-train-200.json",
    # Add others here
    "/home/ezrah/CS598_DLH/project/SleepQA/data/training/sleep-train-500.json",
    "/home/ezrah/CS598_DLH/project/SleepQA/data/training/sleep-train-1000.json",
    "/home/ezrah/CS598_DLH/project/SleepQA/data/training/sleep-train-2000.json",
]

def convert_to_dpr_tsv(input_path):
    output_path = input_path.replace(".json", ".csv")
    print(f"Converting {input_path} -> {output_path}...")
    
    with open(input_path, 'r') as f:
        # Load the JSON list
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # Fallback for JSONL if needed
            f.seek(0)
            data = [json.loads(line) for line in f if line.strip()]

    with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
        # DPR expects Tab Separation
        writer = csv.writer(f_out, delimiter='\t')
        
        for i, entry in enumerate(data):
            question = entry.get("question", "")
            answers = entry.get("answers", [])
            
            # If no ID exists, create a dummy one
            q_id = entry.get("id", str(i))
            
            # COLUMN 0: Question
            # COLUMN 1: Answers (Must be a string representation of a list like "['Ans1', 'Ans2']")
            # COLUMN 2: ID
            writer.writerow([question, str(answers), q_id])

    print("Done.")

if __name__ == "__main__":
    for f in INPUT_FILES:
        if os.path.exists(f):
            convert_to_dpr_tsv(f)
        else:
            print(f"Skipping {f} (Not Found)")