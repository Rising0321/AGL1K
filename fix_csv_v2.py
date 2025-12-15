import argparse
import csv
import json
import re
import io
from typing import List, Dict, Any

def split_into_records(content: str) -> List[str]:
    pattern = r'"You are an expert audio analyst'
    matches = list(re.finditer(pattern, content))
    
    records = []
    for i, match in enumerate(matches):
        start_pos = match.start()
        
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(content)
        
        record_text = content[start_pos:end_pos].strip()
        if record_text:
            records.append(record_text)
    
    return records

def parse_single_record(record_text: str) -> Dict[str, Any]:
    lines = record_text.split('\n')
    
    question_lines = []
    remaining_line = ""

    qwen25 = False
    for i, line in enumerate(lines):
        if 'You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech.' in line:
            qwen25=True
        if 'provide your JSON response now:",' in line:
            parts = line.split('provide your JSON response now:",', 1)
            question_lines.append(parts[0] + 'provide your JSON response now:')
            remaining_line = parts[1] if len(parts) > 1 else ""
            break
        else:
            question_lines.append(line)
    

    question_content = '\n'.join(question_lines)
    if question_content.startswith('"'):
        question_content = question_content[1:]
    

    if remaining_line:
        fields = []
        current_field = ""
        in_quotes = False
        
        for char in remaining_line:
            if char == '"' and (not current_field or current_field[-1] != '\\'):
                in_quotes = not in_quotes
                current_field += char
            elif char == ',' and not in_quotes:
                fields.append(current_field.strip('"'))
                current_field = ""
            else:
                current_field += char
        
        if current_field:
            fields.append(current_field.strip('"'))
    else:
        fields = []


    model_output_lines = []
    found_start = False
    model_output_end_line_idx = -1
    
    for i, line in enumerate(lines):
        if line.strip().startswith('"```json'):
            found_start = True
            model_output_lines.append(line)
        elif found_start:
            model_output_lines.append(line)
            if line.endswith('```",42'):
                model_output_end_line_idx = i
                break


    if model_output_lines:
        model_output_text = '\n'.join(model_output_lines)
        if model_output_text.startswith('"') and model_output_text.endswith('",42'):
            model_output_content = model_output_text[1:-5]  # Remove " at start and ",42 at end
        else:
            model_output_content = model_output_text
        seed = "42"
    else:
        model_output_content = ""
        seed = "42"
    
    if qwen25:
        my_remaining = remaining_line.split('provide your JSON response now:')[1]
        first_brace = my_remaining.find('{')
        last_brace = my_remaining.rfind('}')
        my_remaining = my_remaining[first_brace:last_brace+1]
        my_remaining = my_remaining.replace("\"\"", "\"")
        my_remaining = my_remaining.split("city\"")[1]
        print(my_remaining)
        names = []
        now_name = ""
        cnt = 0
        for i in range(len(my_remaining)):
            if my_remaining[i] == ':':
                now_name = ""
                cnt = 1
            elif my_remaining[i] == ',' or my_remaining[i] == '\\' or my_remaining[i] == '}':
                now_name = now_name.replace("\"", "")
                now_name = now_name.lstrip()
                if len(now_name) > 0:
                    names.append(now_name)
                now_name = ""
                cnt = 0
            else:
                now_name = now_name + my_remaining[i]

        predicted_city = names[0]
        predicted_country= names[1]
        predicted_continent = names[2]
        predicted_longitude = names[3]
        predicted_latitude = names[4]

        fields[9] = predicted_city
        fields[10] = predicted_country
        fields[11] = predicted_continent
        fields[12] = predicted_longitude
        fields[13] = predicted_latitude

    record = {
        'question': question_content,
        'task_id': fields[0] if len(fields) > 0 else '',
        'question_id': fields[1] if len(fields) > 1 else '',
        'true_city': fields[2] if len(fields) > 2 else '',
        'true_country': fields[3] if len(fields) > 3 else '',
        'true_continent': fields[4] if len(fields) > 4 else '',
        'true_longitude': fields[5] if len(fields) > 5 else '',
        'true_latitude': fields[6] if len(fields) > 6 else '',
        'description': fields[7] if len(fields) > 7 else '',
        'title': fields[8] if len(fields) > 8 else '',
        'predicted_city': fields[9] if len(fields) > 9 else '', 
        'predicted_country': fields[10] if len(fields) > 10 else '', 
        'predicted_continent': fields[11] if len(fields) > 11 else '',  
        'predicted_longitude': fields[12] if len(fields) > 12 else '',  
        'predicted_latitude': fields[13] if len(fields) > 13 else '', 
        'reasoning': fields[14] if len(fields) > 14 else '',  
        'model_output': model_output_content,
        'seed': seed
    }
    
    return record

def extract_json_from_model_output(model_output: str) -> Dict[str, Any]:
    json_content = model_output.strip()
    if json_content.startswith('```json'):
        json_content = json_content[7:]  # Remove ```json
    if json_content.endswith('```'):
        json_content = json_content[:-3]  # Remove ```
    
    json_content = json_content.strip()
    
    try:
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Content: {json_content[:200]}...")
        return {}

def main():
    parser = argparse.ArgumentParser(description='Fix CSV file with embedded newlines')
    parser.add_argument('--place', type=str, required=True,
                        help='Model place/path (e.g., gemini-2.5-pro, google/gemini-3-pro-preview)')
    parser.add_argument('--task', type=str, default='audio_localization',
                        help='Task name (default: audio_localization)')
    args = parser.parse_args()

    task = args.task
    place = args.place

    input_file = f"./results/{task}/{place}/seed_42_progress.csv"
    output_file = f"./results/{task}/{place}/seed_42_progress_fixed.json"
    
    print("Reading raw CSV content...")
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    header_line = content.split('\n')[0]
    content_without_header = content[len(header_line)+1:]
    
    print("Splitting into records...")
    records = split_into_records(content_without_header)
    print(f"Found {len(records)} records")
    
    print("Parsing records...")
    parsed_records = []
    for i, record_text in enumerate(records):
        try:
            record = parse_single_record(record_text)
            
            parsed_records.append(record)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} records...")
                
        except Exception as e:
            print(f"Error parsing record {i + 1}: {e}")
            continue
    
    print(f"Successfully parsed {len(parsed_records)} records")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(parsed_records, f, indent=4, ensure_ascii=False)

    print(f"Fixed JSON written to: {output_file}")
    print(f"Total records written: {len(parsed_records)}")

if __name__ == "__main__":
    main()




