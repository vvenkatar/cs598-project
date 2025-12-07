import json
import os

# --- PATHS ---
# Adjust these to match your actual file locations
INPUT_TRAIN_200 = "/home/ezrah/CS598_DLH/project/SleepQA/data/training/sleep-train-200.json"
INPUT_TRAIN_500 = "/home/ezrah/CS598_DLH/project/SleepQA/data/training/sleep-train-500.json"
INPUT_TRAIN_1000 = "/home/ezrah/CS598_DLH/project/SleepQA/data/training/sleep-train-1000.json"
INPUT_TRAIN_2000 = "/home/ezrah/CS598_DLH/project/SleepQA/data/training/sleep-train-2000.json"

# INPUT_DEV   = "/home/ezrah/CS598_DLH/project/SleepQA/data/training/sleep-dev-clean.json" 

OUTPUT_TRAIN_200 = "/home/ezrah/CS598_DLH/project/SleepQA/data/training/oracle/sleep-train-200-reader.json"
OUTPUT_TRAIN_500 = "/home/ezrah/CS598_DLH/project/SleepQA/data/training/oracle/sleep-train-500-reader.json"
OUTPUT_TRAIN_1000 = "/home/ezrah/CS598_DLH/project/SleepQA/data/training/oracle/sleep-train-1000-reader.json"
OUTPUT_TRAIN_2000 = "/home/ezrah/CS598_DLH/project/SleepQA/data/training/oracle/sleep-train-2000-reader.json"

# OUTPUT_DEV   = "/home/ezrah/CS598_DLH/project/SleepQA/data/training/sleep-dev-reader.json"

def convert_to_reader_format(input_path, output_path):
    print(f"Converting {input_path}...")
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Skipping {input_path} (File not found)")
        return

    reader_data = []
    for entry in data:
        # The reader expects 'ctxs', not 'positive_ctxs'
        # We take the positive_ctxs and move them into 'ctxs' with a fake high score
        # This creates an "Oracle" dataset (perfect context always provided)
        
        pos_ctxs = entry.get("positive_ctxs", [])
        neg_ctxs = entry.get("negative_ctxs", []) + entry.get("hard_negative_ctxs", [])
        
        formatted_ctxs = []
        
        # Add Positive Contexts (Gold Standard)
        for pc in pos_ctxs:
            formatted_ctxs.append({
                "id": pc.get("psg_id", "0"),
                "title": pc.get("title", ""),
                "text": pc.get("text", ""),
                "score": 1000, # High score ensures Reader pays attention to this
                "has_answer": True
            })
            
        # Add Negative Contexts (Optional, but good for robustness)
        # We only add a few to keep file size manageable
        for nc in neg_ctxs[:2]: 
            formatted_ctxs.append({
                "id": nc.get("psg_id", "0"),
                "title": nc.get("title", ""),
                "text": nc.get("text", ""),
                "score": 0,
                "has_answer": False
            })

        if formatted_ctxs:
            new_entry = {
                "question": entry["question"],
                "answers": entry["answers"],
                "ctxs": formatted_ctxs
            }
            reader_data.append(new_entry)

    with open(output_path, 'w') as f:
        json.dump(reader_data, f, indent=2)
    
    print(f"Saved {len(reader_data)} items to {output_path}")

if __name__ == "__main__":
    convert_to_reader_format(INPUT_TRAIN_200, OUTPUT_TRAIN_200)
    convert_to_reader_format(INPUT_TRAIN_500, OUTPUT_TRAIN_500)
    convert_to_reader_format(INPUT_TRAIN_1000, OUTPUT_TRAIN_1000)
    convert_to_reader_format(INPUT_TRAIN_2000, OUTPUT_TRAIN_2000)
    # convert_to_reader_format(INPUT_DEV, OUTPUT_DEV)