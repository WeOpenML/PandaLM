import json

def convert_json_to_jsonl(json_file_path, jsonl_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        
        if not isinstance(data, list):
            raise ValueError("The JSON file should contain a list of objects")
        
        with open(jsonl_file_path, 'w') as jsonl_file:
            for item in data:
                jsonl_file.write(json.dumps(item) + '\n')
                
    print(f"Converted {json_file_path} to {jsonl_file_path}")

# Example usage:
convert_json_to_jsonl('train.json', 'train.jsonl')
