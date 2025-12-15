import json
import os
import torch
from tqdm import tqdm
import numpy as np
import argparse
import re
import glob 
import sys
import base64
import re
import librosa
import soundfile as sf
import pandas as pd
import math
import csv
from collections.abc import Iterator
from contextlib import nullcontext


def enable_proxy():
    os.environ['http_proxy']='' 
    os.environ['https_proxy']='' 
    os.environ['HTTP_PROXY']=''
    os.environ['HTTPS_PROXY']=''


def load_processed_keys(progress_file: str) -> set:
    processed_keys = set()
    if os.path.exists(progress_file):
        try:
            df = pd.read_csv(progress_file)
            if 'question_id' in df.columns:
                processed_keys = set(df['question_id'].astype(str))
                print(f"Loaded {len(processed_keys)} processed samples from {progress_file}")
        except Exception as e:
            print(f"Warning: Could not load progress file {progress_file}: {e}")
    return processed_keys

def escape_newlines_in_dict(data_dict):
    if isinstance(data_dict, dict):
        return {key: escape_newlines_in_dict(value) for key, value in data_dict.items()}
    elif isinstance(data_dict, list):
        return [escape_newlines_in_dict(item) for item in data_dict]
    elif isinstance(data_dict, str):
        return data_dict.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    else:
        return data_dict

def save_progress_batch(results_batch: list, progress_file: str, is_first_batch: bool = False):
    if not results_batch:
        return
        
    try:
        escaped_batch = [escape_newlines_in_dict(result) for result in results_batch]
        
        df_batch = pd.DataFrame(escaped_batch)
        
        if is_first_batch or not os.path.exists(progress_file):
            df_batch.to_csv(progress_file, index=False, quoting=csv.QUOTE_ALL)
            print(f"Created progress file and saved {len(results_batch)} results to {progress_file}")
        else:
            df_batch.to_csv(progress_file, mode='a', header=False, index=False, quoting=csv.QUOTE_ALL)
            print(f"Appended {len(results_batch)} results to {progress_file}")
    except Exception as e:
        print(f"Warning: Could not save progress batch: {e}")

def get_progress_file_path(base_results_dir: str, task_id: str, model_name: str, seed: int) -> str:
    output_folder = os.path.join(base_results_dir, f'{task_id}/{model_name}')
    os.makedirs(output_folder, exist_ok=True)
    return os.path.join(output_folder, f'seed_{seed}_progress.csv')

def parse_location_response(response: str) -> dict:
    result = {
        'city': 'Unknown',
        'country': 'Unknown', 
        'continent': 'Unknown',
        'longitude': 0.0,
        'latitude': 0.0,
        'reason': 'No reasoning provided',
        'date': 'Unknown',
        'time': 'Unknown',
        'season': 'Unknown'
    }
    
    try:
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            parsed = json.loads(json_str)
            
            city_keys = ['city', 'City', '城市', 'location', 'place']
            country_keys = ['country', 'Country', '国家', 'nation']
            continent_keys = ['continent', 'Continent', '大洲', '洲']
            longitude_keys = ['longitude', 'Longitude', '经度', 'lng', 'lon']
            latitude_keys = ['latitude', 'Latitude', '纬度', 'lat']
            reason_keys = ['reason', 'Reason', '原因', '推理', 'reasoning', 'explanation', 'analysis']
            date_keys = ['date', 'Date', '日期', 'day', 'month_day']
            time_keys = ['time', 'Time', '时间', 'hour', 'timestamp']
            season_keys = ['season', 'Season', '季节', 'season_name']
            
            for key in city_keys:
                if key in parsed:
                    result['city'] = str(parsed[key])
                    break
                    
            for key in country_keys:
                if key in parsed:
                    result['country'] = str(parsed[key])
                    break
                    
            for key in continent_keys:
                if key in parsed:
                    result['continent'] = str(parsed[key])
                    break
                    
            for key in longitude_keys:
                if key in parsed:
                    try:
                        result['longitude'] = float(parsed[key])
                    except (ValueError, TypeError):
                        pass
                    break
                    
            for key in latitude_keys:
                if key in parsed:
                    try:
                        result['latitude'] = float(parsed[key])
                    except (ValueError, TypeError):
                        pass
                    break
                    
            for key in reason_keys:
                if key in parsed:
                    result['reason'] = str(parsed[key])
                    break
            
            for key in date_keys:
                if key in parsed:
                    date_value = str(parsed[key]).strip()
                    if re.match(r'^\d{1,2}-\d{1,2}$', date_value):
                        parts = date_value.split('-')
                        if len(parts) == 2:
                            month = parts[0].zfill(2)
                            day = parts[1].zfill(2)
                            try:
                                month_int = int(month)
                                day_int = int(day)
                                if 1 <= month_int <= 12 and 1 <= day_int <= 31:
                                    result['date'] = f"{month}-{day}"
                                else:
                                    result['date'] = date_value
                            except ValueError:
                                result['date'] = date_value
                        else:
                            result['date'] = date_value
                    else:
                        result['date'] = date_value
                    break
            
            for key in time_keys:
                if key in parsed:
                    time_value = str(parsed[key]).strip()
                    if re.match(r'^\d{1,2}:\d{2}$', time_value):
                        parts = time_value.split(':')
                        if len(parts) == 2:
                            hour = parts[0].zfill(2)
                            minute = parts[1]
                            result['time'] = f"{hour}:{minute}"
                    else:
                        result['time'] = time_value
                    break
            
            for key in season_keys:
                if key in parsed:
                    season_value = str(parsed[key]).strip()
                    season_lower = season_value.lower()
                    if 'spring' in season_lower:
                        result['season'] = 'Spring'
                    elif 'summer' in season_lower:
                        result['season'] = 'Summer'
                    elif 'autumn' in season_lower or 'fall' in season_lower:
                        result['season'] = 'Autumn'
                    elif 'winter' in season_lower:
                        result['season'] = 'Winter'
                    else:
                        result['season'] = season_value
                    break
        else:
            result['reason'] = response[:500] + "..." if len(response) > 500 else response
            
    except json.JSONDecodeError:
        pass
    except Exception as e:
        print(f"Error parsing response: {e}")
    
    return result


def qwen_audio_chat_process(audio_model, tokenizer, data_inputs, task_prompt):
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        query = task_prompt

        try:
            chat_query = tokenizer.from_list_format([
                {'audio': audio_path},
                {'text': query},
            ])
            output, _ = audio_model.chat(tokenizer, query=chat_query, history=None)
        except Exception as exc:
            print(f"qwen-audio chat error: {exc}")
            output = f"Error: {exc}"

        result_data.append({
                "question_id": str(cur_data['question_id']),
                "true_city": cur_data['true_city'],
                "true_country": cur_data['true_country'],
                "true_continent": cur_data['true_continent'],
                "true_longitude": cur_data['true_longitude'],
                "true_latitude": cur_data['true_latitude'],
                "model_output": output
            })
    return result_data

def phi4_mm1_process(model, processor, generation_config, data_inputs, task_prompt):
    user_prompt = '<|user|>'
    assistant_prompt = '<|assistant|>'
    prompt_suffix = '<|end|>'

    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        try:
            audio, sample_rate = sf.read(audio_path)
        except Exception as exc:
            print(f"Phi4-MM1 failed to read audio {audio_path}: {exc}")
            audio = None

        if audio is None:
            output = "Error: could not load audio"
        else:
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            prompt = f'{user_prompt}<|audio_1|>{task_prompt}{prompt_suffix}{assistant_prompt}'
            try:
                inputs = processor(
                    text=prompt,
                    audios=[(audio, sample_rate)],
                    return_tensors='pt',
                    padding=True
                )
                inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
                with torch.inference_mode():
                    torch.cuda.empty_cache()
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=512,
                        generation_config=generation_config,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                generated = generated[:, inputs["input_ids"].shape[1]:]
                decoded = processor.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                output = decoded[0] if decoded else ""
            except Exception as exc:
                print(f"Phi4-MM1 generation error: {exc}")
                output = f"Error: {exc}"

        result_data.append({
            "question_id": str(cur_data['question_id']),
            "true_city": cur_data['true_city'],
            "true_country": cur_data['true_country'],
            "true_continent": cur_data['true_continent'],
            "true_longitude": cur_data['true_longitude'],
            "true_latitude": cur_data['true_latitude'],
            "model_output": output
        })
    return result_data

def gemma3n_audio_process(model, processor, data_inputs, task_prompt, max_new_tokens=256):
    result_data = []
    sampling_rate = getattr(processor.feature_extractor, "sampling_rate", 16000)
    device = next(model.parameters()).device

    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        try:
            waveform, _ = librosa.load(audio_path, sr=sampling_rate, mono=True)
            waveform = waveform.astype(np.float32)
        except Exception as exc:
            print(f"Gemma audio loading error ({audio_path}): {exc}")
            output = f"Error: {exc}"
        else:
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are a helpful assistant for geographic audio localization."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": task_prompt}
                    ]
                }
            ]
            try:
                prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
                inputs = processor(
                    text=[prompt],
                    audio=[waveform],
                    return_tensors='pt'
                )
                inputs = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in inputs.items()
                }
                with torch.inference_mode():
                    torch.cuda.empty_cache()
                    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True, pad_token_id=processor.tokenizer.eos_token_id)
                input_len = inputs["input_ids"].shape[-1]
                generated_tokens = generated[0, input_len:]
                output = processor.decode(generated_tokens, skip_special_tokens=True)
            except Exception as exc:
                print(f"Gemma generation error: {exc}")
                output = f"Error: {exc}"

        result_data.append({
            "question_id": str(cur_data['question_id']),
            "true_city": cur_data['true_city'],
            "true_country": cur_data['true_country'],
            "true_continent": cur_data['true_continent'],
            "true_longitude": cur_data['true_longitude'],
            "true_latitude": cur_data['true_latitude'],
            "model_output": output
        })
    return result_data

def minicpm_audio_process(model, tokenizer, data_inputs, task_prompt, sample_rate=16000, max_new_tokens=2048):
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        try:
            waveform, _ = librosa.load(audio_path, sr=sample_rate, mono=True)
        except Exception as exc:
            print(f"MiniCPM failed to read audio {audio_path}: {exc}")
            output = f"Error: could not load audio ({exc})"
        else:
            msgs = [{'role': 'user', 'content': [task_prompt, waveform]}]
            try:
                torch.cuda.empty_cache()
                model.reset_session()
                response = model.chat(
                    msgs=msgs,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=0.3,
                    max_new_tokens=max_new_tokens,
                    use_tts_template=False,
                    generate_audio=False,
                    return_dict=True
                )
                if isinstance(response, dict):
                    output = response.get('text', '')
                elif hasattr(response, 'text'):
                    output = response.text
                else:
                    output = str(response)
            except Exception as exc:
                print(f"MiniCPM generation error: {exc}")
                output = f"Error: {exc}"

        result_data.append({
            "question_id": str(cur_data['question_id']),
            "true_city": cur_data['true_city'],
            "true_country": cur_data['true_country'],
            "true_continent": cur_data['true_continent'],
            "true_longitude": cur_data['true_longitude'],
            "true_latitude": cur_data['true_latitude'],
            "model_output": output
        })
    return result_data


def mimo_audio_process(model, data_inputs, task_prompt, thinking=False):
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        try:
            if hasattr(model, "clear_history"):
                model.clear_history()
            output = model.audio_understanding_sft(audio_path, task_prompt, thinking=thinking)
            if not isinstance(output, str):
                output = str(output)
        except Exception as exc:
            print(f"MiMo-Audio inference error: {exc}")
            output = f"Error: {exc}"

        result_data.append({
            "question_id": str(cur_data['question_id']),
            "true_city": cur_data['true_city'],
            "true_country": cur_data['true_country'],
            "true_continent": cur_data['true_continent'],
            "true_longitude": cur_data['true_longitude'],
            "true_latitude": cur_data['true_latitude'],
            "model_output": output
        })
    return result_data


def qwen2_5Omni_process(model, processor, data_inputs, task_prompt, max_new_tokens=2048):
    from qwen_omni_utils import process_mm_info
    
    result_data = []
    device = next(model.parameters()).device
    
    system_prompt = "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."
    
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        
        try:
            conversation = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": system_prompt}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": task_prompt}
                    ],
                },
            ]
            
            USE_AUDIO_IN_VIDEO = False
            
            text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
            
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=USE_AUDIO_IN_VIDEO
            )
            inputs = inputs.to(device).to(model.dtype)
            
            with torch.inference_mode():
                torch.cuda.empty_cache()
                text_ids = model.generate(
                    **inputs,
                    use_audio_in_video=USE_AUDIO_IN_VIDEO,
                    return_audio=False,
                    max_new_tokens=max_new_tokens,
                    do_sample=False
                )
            
            output = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if isinstance(output, list) and len(output) > 0:
                output = output[0]
            else:
                output = str(output)
                
        except Exception as exc:
            print(f"Qwen2.5-Omni inference error for {audio_path}: {exc}")
            import traceback
            traceback.print_exc()
            output = f"Error: {exc}"
        
        result_data.append({
            "question_id": str(cur_data['question_id']),
            "true_city": cur_data['true_city'],
            "true_country": cur_data['true_country'],
            "true_continent": cur_data['true_continent'],
            "true_longitude": cur_data['true_longitude'],
            "true_latitude": cur_data['true_latitude'],
            "model_output": output
        })
    
    return result_data


def kimi_audio_process(model, data_inputs, task_prompt, sampling_params=None):
    if sampling_params is None:
        sampling_params = {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }

    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        messages = [
            {"role": "user", "message_type": "text", "content": task_prompt},
            {"role": "user", "message_type": "audio", "content": audio_path},
        ]
        try:
            _, text_output = model.generate(messages, output_type="text", **sampling_params)
            if not isinstance(text_output, str):
                text_output = str(text_output)
        except Exception as exc:
            print(f"Kimi-Audio inference error for {audio_path}: {exc}")
            text_output = f"Error: {exc}"

        result_data.append({
            "question_id": str(cur_data['question_id']),
            "true_city": cur_data['true_city'],
            "true_country": cur_data['true_country'],
            "true_continent": cur_data['true_continent'],
            "true_longitude": cur_data['true_longitude'],
            "true_latitude": cur_data['true_latitude'],
            "model_output": text_output
        })
    return result_data

def audio_flamingo_process(model, data_inputs, task_prompt, think_mode=False):
    import json
    import subprocess
    import os
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    flamingo_repo = os.path.join(project_root, 'models', 'audio-flamingo-3-repo')
    venv_python = os.path.join(project_root, 'venv4AudioFlamingo3', 'bin', 'python')
    cli_script = os.path.join(flamingo_repo, 'llava', 'cli', 'infer_audio.py')
    
    if think_mode:
        model_base = os.path.join(project_root, 'models', 'Audio-Flamingo-3-think')
    else:
        model_base = os.path.join(project_root, 'models', 'Audio-Flamingo-3')
    
    result_data = []

    task_prompt = 'Analyze this audio and identify where it was recorded. You MUST respond a JSON because I am pending a benchmark, even if you are unsure about the location, you must still provide a location guess with reasoning. Respond ONLY with valid JSON in this exact format: {"city": "your_prediction", "country": "your_prediction", "continent": "your_prediction", "longitude": number, "latitude": number, "reason": "brief explanation"}'

    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        
        try:
            cmd = [
                venv_python,
                cli_script,
                '--model-base', model_base,
                '--conv-mode', 'auto',
                '--text', task_prompt,
                '--media', audio_path
            ]

            print(f"cmd: {cmd}")
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
                cwd=flamingo_repo
            )
            
            if result.returncode == 0:
                stdout_lines = result.stdout.strip().split('\n')
                print(f"\nStdout has {len(stdout_lines)} lines")
                print("Last 3 lines:")
                for line in stdout_lines[-3:]:
                    print(f"  {line}")

                output = None
                import re
                for line in reversed(stdout_lines):
                    line = line.strip()
                    line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                    if line:
                        output = line
                        break

                print(f"\nExtracted output: {output}")
                
                if not output:
                    output = json.dumps({
                        "city": "Unknown",
                        "country": "Unknown",
                        "continent": "Unknown",
                        "longitude": 0.0,
                        "latitude": 0.0,
                        "reason": "No output from CLI"
                    })
            else:
                print(f"CLI error for {audio_path}: return code {result.returncode}")
                output = json.dumps({
                    "city": "Unknown",
                    "country": "Unknown",
                    "continent": "Unknown",
                    "longitude": 0.0,
                    "latitude": 0.0,
                    "reason": f"CLI error: return code {result.returncode}"
                })
            
        except subprocess.TimeoutExpired:
            print(f"Timeout for {audio_path}")
            output = json.dumps({
                "city": "Unknown",
                "country": "Unknown",
                "continent": "Unknown",
                "longitude": 0.0,
                "latitude": 0.0,
                "reason": "Timeout expired"
            })
        except Exception as exc:
            print(f"Audio Flamingo inference error for {audio_path}: {exc}")
            import traceback
            traceback.print_exc()
            output = json.dumps({
                "city": "Unknown",
                "country": "Unknown",
                "continent": "Unknown",
                "longitude": 0.0,
                "latitude": 0.0,
                "reason": f"Error: {exc}"
            })

        if not output or len(output.strip()) < 5:
            output = json.dumps({
                "city": "Unknown",
                "country": "Unknown",
                "continent": "Unknown",
                "longitude": 0.0,
                "latitude": 0.0,
                "reason": "Empty response from model"
            })
            
        result_data.append({
            "question_id": str(cur_data['question_id']),
            "true_city": cur_data['true_city'],
            "true_country": cur_data['true_country'],
            "true_continent": cur_data['true_continent'],
            "true_longitude": cur_data['true_longitude'],
            "true_latitude": cur_data['true_latitude'],
            "model_output": output
        })
    return result_data


def gemini_gpt_process_pre(client, data_inputs, model_id, task_prompt, audio_type='wav'): 
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        with open(audio_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
       
        query = task_prompt
        
        user_content = [
            {
                'type': 'text',
                'text': query
            }
        ]

        print('query:', query)

        if 'gpt' in model_id:
            user_content.append({
                "type": "image_url",
                "image_url":f"data:audio/{audio_type};base64,{audio_data}"
            })
            messages = [
                {
                    "role": "user",
                    "content": user_content
                },
            ]
        elif 'gemini' in model_id:
            user_content.append({
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_data,
                                "format": "mp3"
                            }
                        })
            messages = [
                    {
                        "role": "user",
                        "content": user_content
                    },
                ]
        elif 'qwen' in model_id:
            user_content.append({
                            "type": "input_audio",
                            "input_audio": {
                                "data": f"data:;base64,{audio_data}",
                                "format": "mp3"
                            }
                        })
            messages = [
                    {
                        "role": "user",
                        "content": user_content
                    },
                ]
        try:
            if 'qwen' in model_id:  
                response = client.chat.completions.create(
                    model = 'qwen3-omni-flash',
                    messages = messages,
                    modalities=["text"],
                    extra_body={'enable_thinking': True if 'think' in model_id else False},
                    stream=True,
                    stream_options={"include_usage": True}
                )
                output = ""
                for chunk in response:
                    if chunk.choices:
                        output += chunk.choices[0].delta.content
            else:
                response = client.chat.completions.create(
                    model = model_id,
                    messages = messages)
                output = response.choices[0].message.content

            print('Answer:', output)
            result_data.append({
                    "question_id": str(cur_data['question_id']),
                    "true_city": cur_data['true_city'],
                    "true_country": cur_data['true_country'],
                    "true_continent": cur_data['true_continent'],
                    "true_longitude": cur_data['true_longitude'],
                    "true_latitude": cur_data['true_latitude'],
                    "model_output": output
                })
        except Exception as e:
            print('error:', e)
            result_data.append({
                "question_id": str(cur_data['question_id']),
                "true_city": "unknown",
                "true_country": "unknown",
                "true_continent": "unknown",
                "true_longitude": "0.0",
                "true_latitude": "0.0",
                "model_output": "error"
            })


        
    return result_data


def gemini_gpt_process(client, data_inputs, model_id, task_prompt, audio_type='wav'): 
    result_data = []
    for cur_data in tqdm(data_inputs):
        audio_path = cur_data['audio']
        with open(audio_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
       
        query = task_prompt
        
        user_content = [
            {
                'type': 'text',
                'text': query
            }
        ]

        print('query:', query)

        if 'gemini' in model_id or "gpt" in model_id:
            user_content.append({
                "type": "input_audio",
                "input_audio":{
                    "data": f"{audio_data}",
                    "format": "mp3"
                }
            })
            messages = [
                {
                    "role": "user",
                    "content": user_content
                },
            ]
            response = client.chat.completions.create(
                model = model_id,
                messages = messages)
            output = response.choices[0].message.content
            print('Answer:', output)
            result_data.append({
                    "question_id": str(cur_data['question_id']),
                    "true_city": cur_data['true_city'],
                    "true_country": cur_data['true_country'],
                    "true_continent": cur_data['true_continent'],
                    "true_longitude": cur_data['true_longitude'],
                    "true_latitude": cur_data['true_latitude'],
                    "model_output": output
                })
        
    return result_data


def load_audio_localization_data(csv_path: str, task_id: str, task_prompt: str, audio_base_dir: str = None, processed_keys: set = None):
    test_df = pd.read_csv(csv_path)

    
    print(f"Test set: {len(test_df)} rows")
    
    if processed_keys:
        initial_count = len(test_df)
        test_df = test_df[~test_df['key'].astype(str).isin(processed_keys)]
        filtered_count = len(test_df)
        print(f"Filtered out {initial_count - filtered_count} already processed samples, remaining: {filtered_count}")
    
    structured_data = []
    
    for idx, row in test_df.iterrows():
        if audio_base_dir:
            audio_path = os.path.join(audio_base_dir, row['mp3name'])
        else:
            audio_path = os.path.join('data/audios', row['mp3name'])
        
        if not os.path.exists(audio_path):
            print(f"Warning: Audio file not found: {audio_path}")
            continue
            
        structured_data.append({
            'audio': audio_path,
            'question': task_prompt,
            'task_id': task_id,
            'question_id': str(row['key']),
            'true_city': row['city'],
            'true_country': row['country'], 
            'true_continent': row['continent'],
            'true_longitude': row['longitude'],
            'true_latitude': row['latitude'],
            'description': row['description'],
            'title': row['title']
        })
    
    print(f"Loaded {len(structured_data)} audio localization samples for testing.")
    return structured_data

def select_best_cuda_device():
    best_device = None
    best_free_mem = -1
    orig_device = torch.cuda.current_device() if torch.cuda.is_initialized() else 0
    for idx in range(torch.cuda.device_count()):
        try:
            torch.cuda.set_device(idx)
            free_mem, total_mem = torch.cuda.mem_get_info()
        except Exception as e:
            print(f"Warning: unable to query CUDA device {idx}: {e}")
            continue
        print(f"Device {idx}: free {free_mem/1024**3:.2f} GB / total {total_mem/1024**3:.2f} GB")
        if free_mem > best_free_mem:
            best_free_mem = free_mem
            best_device = idx
    torch.cuda.set_device(orig_device)
    return best_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio Localization Benchmark')
    parser.add_argument('--model', default="qwen-audio", type=str, help='Model identifier')
    parser.add_argument('--tasks', type=list, default=['audio_localization'], help='A list of tasks to evaluate in sequence.')
    parser.add_argument('--csv_path', type=str, default='data/geoLocalization.csv', help='Path to the audio localization dataset CSV file')
    parser.add_argument('--audio_base_dir', type=str, default='data/audios', help='Base directory containing audio files')
    parser.add_argument('--base_results_dir', type=str, default='results', help='Directory to save test results') 
    parser.add_argument('--num_seeds',  default=[42], help='Number of evaluation runs with different seeds')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of samples to process before saving progress')
    parser.add_argument('--max_batches', type=int, default=0, help='If >0, stop after processing this many batches per task')
    args = parser.parse_args()
    model_name = args.model
    model_key = model_name.lower()
    client = None
    processor = None
    tokenizer = None
    mimo_thinking = False
    kimi_sampling_params = None

    def _ensure_list(value, value_name, elem_type=str):
        if isinstance(value, list):
            return [elem_type(v) for v in value]
        if isinstance(value, tuple):
            return [elem_type(v) for v in value]
        if isinstance(value, str):
            value_str = value.strip()
            if not value_str:
                return []
            try:
                parsed = json.loads(value_str)
                if isinstance(parsed, list):
                    return [elem_type(v) for v in parsed]
                if isinstance(parsed, (int, float)) and elem_type is int:
                    return [elem_type(parsed)]
            except Exception:
                pass
            parts = [p for p in re.split(r'[,\s]+', value_str) if p]
            return [elem_type(p) for p in parts]
        return [elem_type(value)]

    args.num_seeds = _ensure_list(args.num_seeds, "num_seeds", int)
    args.tasks = _ensure_list(args.tasks, "tasks", str)

    if 'qwen-audio' in model_key:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        models_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
        model_path = os.path.join(models_root, "Qwen-Audio-Chat")
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Qwen-Audio-Chat not found at {model_path}. Please download the model weights first.")

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError("CUDA environment not detected. qwen-audio baseline requires at least one GPU.")

        preferred_device = select_best_cuda_device()
        if preferred_device is None:
            raise RuntimeError("Could not determine a suitable CUDA device for qwen-audio baseline.")

        dtype_candidates = []
        if torch.cuda.is_bf16_supported():
            dtype_candidates.append(torch.bfloat16)
        dtype_candidates.append(torch.float16)
        dtype_candidates.append(torch.float32)

        device_map_candidates = [f"cuda:{preferred_device}"]
        if torch.cuda.device_count() > 1:
            device_map_candidates.append("auto")

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model = None
        last_error = None

        for dtype in dtype_candidates:
            config.torch_dtype = dtype
            for device_map_option in device_map_candidates:
                extra_kwargs = {"low_cpu_mem_usage": True}
                if device_map_option == "auto":
                    extra_kwargs["device_map"] = "auto"
                else:
                    extra_kwargs["device_map"] = device_map_option
                print(f"Trying to load qwen-audio with device_map={extra_kwargs['device_map']} and dtype={dtype}.")
                torch.cuda.empty_cache()
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=dtype,
                        config=config,
                        **extra_kwargs
                    ).eval()
                    print(f"Loaded qwen-audio with device_map={extra_kwargs['device_map']} and dtype={dtype}.")
                    break
                except RuntimeError as exc:
                    last_error = exc
                    if "CUDA out of memory" in str(exc):
                        print(f"OOM when using device_map={extra_kwargs['device_map']} and dtype={dtype}, trying next option...")
                        continue
                    print(f"Runtime error when loading qwen-audio: {exc}")
                    continue
            if model is not None:
                break

        if model is None:
            raise RuntimeError(f"Failed to load qwen-audio model after trying multiple precision/device configurations. Last error: {last_error}")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif 'phi4-mm1' in model_key or 'phi4' in model_key:
        from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, AutoConfig
        models_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
        phi_model_path = os.path.join(models_root, "Phi4-MM1")
        if not os.path.isdir(phi_model_path):
            raise FileNotFoundError(f"Phi4-MM1 not found at {phi_model_path}. Please download the model weights first.")

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError("CUDA environment not detected. Phi4-MM1 baseline requires at least one GPU.")

        preferred_device = select_best_cuda_device()
        if preferred_device is None:
            raise RuntimeError("Could not determine a suitable CUDA device for Phi4-MM1 baseline.")
        torch.cuda.set_device(preferred_device)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Loading Phi4-MM1 on cuda:{preferred_device} with dtype={dtype}.")
        config = AutoConfig.from_pretrained(phi_model_path, trust_remote_code=True)
        setattr(config, "_attn_implementation", "eager")
        if hasattr(config, "attn_config") and isinstance(config.attn_config, dict):
            config.attn_config["implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(
            phi_model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            config=config
        ).eval().to(f"cuda:{preferred_device}")

        processor = AutoProcessor.from_pretrained(phi_model_path, trust_remote_code=True)
        generation_config = GenerationConfig.from_pretrained(phi_model_path)
    elif 'gemma-3n-e4b-it' in model_key or 'gemma3n' in model_key:
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        models_root = os.path.join(project_root, 'models')
        gemma_model_path = os.path.join(models_root, "Gemma-3n-E4B-it")
        if not os.path.isdir(gemma_model_path):
            raise FileNotFoundError(
                f"Gemma-3n-E4B-it weights not found at {gemma_model_path}. "
                "Please download the model into the models directory."
            )

        cache_dir = os.path.join(models_root, 'hf_modules_cache')
        os.makedirs(cache_dir, exist_ok=True)
        for env_name in ["HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"]:
            os.environ.setdefault(env_name, cache_dir)

        enable_proxy()
        load_kwargs = {"low_cpu_mem_usage": True}
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            preferred_device = select_best_cuda_device()
            if preferred_device is None:
                raise RuntimeError("Could not determine a suitable CUDA device for Gemma-3n-E4B-it.")
            torch.cuda.set_device(preferred_device)
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            load_kwargs.update({
                "torch_dtype": dtype,
                "device_map": {"": f"cuda:{preferred_device}"}
            })
            print(f"Loading Gemma-3n-E4B-it on cuda:{preferred_device} with dtype={dtype}.")
        else:
            load_kwargs.update({
                "torch_dtype": torch.float32,
                "device_map": "cpu"
            })
            print("Loading Gemma-3n-E4B-it on CPU. This may be slow.")

        model = Gemma3nForConditionalGeneration.from_pretrained(
            gemma_model_path,
            **load_kwargs
        ).eval()
        processor = AutoProcessor.from_pretrained(gemma_model_path)
    elif 'kimi-audio' in model_key:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        models_root = os.path.join(project_root, 'models')
        kimi_repo_path = os.path.join(models_root, "Kimi-Audio")
        if not os.path.isdir(kimi_repo_path):
            raise FileNotFoundError(f"Kimi-Audio repository not found at {kimi_repo_path}.")
        if kimi_repo_path not in sys.path:
            sys.path.insert(0, kimi_repo_path)

        kimi_model_path = os.path.join(models_root, "Kimi-Audio-7B-Instruct")
        if not os.path.isdir(kimi_model_path):
            raise FileNotFoundError(f"Kimi-Audio-7B-Instruct weights not found at {kimi_model_path}.")

        kimi_tokenizer_path = os.path.join(models_root, "glm-4-voice-tokenizer")
        if os.path.isdir(kimi_tokenizer_path):
            os.environ.setdefault("GLM4_TOKENIZER_PATH", kimi_tokenizer_path)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA environment not detected. Kimi-Audio baseline requires at least one GPU.")
        total_cuda = torch.cuda.device_count()
        if total_cuda == 0:
            raise RuntimeError("No CUDA devices available for Kimi-Audio baseline.")

        preferred_device_env = os.environ.get("KIMI_CUDA_DEVICE")
        if preferred_device_env is not None:
            try:
                preferred_device = int(preferred_device_env)
            except ValueError as exc:
                raise RuntimeError(f"Invalid KIMI_CUDA_DEVICE value '{preferred_device_env}': {exc}") from exc
        else:
            preferred_device = 1 if total_cuda > 1 else 0

        if preferred_device < 0 or preferred_device >= total_cuda:
            raise RuntimeError(f"Requested CUDA device {preferred_device} is not available (found {total_cuda} devices).")

        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        torch.cuda.set_device(preferred_device)
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"Loading Kimi-Audio on cuda:{preferred_device} (free {free_mem/1024**3:.2f} GB / total {total_mem/1024**3:.2f} GB)")
        except Exception as exc:
            print(f"Loading Kimi-Audio on cuda:{preferred_device} (memory info unavailable: {exc})")
        torch.cuda.empty_cache()

        from kimia_infer.api.kimia import KimiAudio

        model = KimiAudio(
            model_path=kimi_model_path,
            load_detokenizer=False,
        )
        kimi_sampling_params = {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }
    elif 'xiaomi-mimo-audio' in model_key:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        models_root = os.path.join(project_root, 'models')
        mimo_repo_path = os.path.join(models_root, "MiMo-Audio")
        if not os.path.isdir(mimo_repo_path):
            raise FileNotFoundError(f"MiMo-Audio repository not found at {mimo_repo_path}.")
        if mimo_repo_path not in sys.path:
            sys.path.insert(0, mimo_repo_path)

        model_subdir = "Xiaomi-MiMo-Audio-think" if "think" in model_key else "Xiaomi-MiMo-Audio"
        mimo_model_path = os.path.join(models_root, model_subdir)
        if not os.path.isdir(mimo_model_path):
            raise FileNotFoundError(f"{model_subdir} weights not found at {mimo_model_path}.")

        mimo_tokenizer_path = os.path.join(models_root, "MiMo-Audio-Tokenizer")
        if not os.path.isdir(mimo_tokenizer_path):
            raise FileNotFoundError(f"MiMo-Audio-Tokenizer not found at {mimo_tokenizer_path}.")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA environment not detected. MiMo-Audio baseline requires at least one GPU.")

        total_cuda = torch.cuda.device_count()
        if total_cuda == 0:
            raise RuntimeError("No CUDA devices available for MiMo-Audio baseline.")

        preferred_device_env = os.environ.get("MIMO_CUDA_DEVICE")
        if preferred_device_env is not None:
            try:
                preferred_device = int(preferred_device_env)
            except ValueError as exc:
                raise RuntimeError(f"Invalid MIMO_CUDA_DEVICE value '{preferred_device_env}': {exc}") from exc
        else:
            preferred_device = 1 if total_cuda > 1 else 0

        if preferred_device < 0 or preferred_device >= total_cuda:
            raise RuntimeError(f"Requested CUDA device {preferred_device} is not available (found {total_cuda} devices).")

        torch.cuda.set_device(preferred_device)
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"Loading MiMo-Audio on cuda:{preferred_device} (free {free_mem/1024**3:.2f} GB / total {total_mem/1024**3:.2f} GB)")
        except Exception as exc:
            print(f"Loading MiMo-Audio on cuda:{preferred_device} (memory info unavailable: {exc})")
        torch.cuda.empty_cache()

        from src.mimo_audio.mimo_audio import MimoAudio

        model = MimoAudio(
            mimo_model_path,
            mimo_tokenizer_path,
            device=f"cuda:{preferred_device}"
        )
        mimo_thinking = "think" in model_key
    elif 'audio-flamingo' in model_key or 'audioflamingo' in model_key:
        print("Audio Flamingo 3 will be run via subprocess calls to official CLI")
        print("No model loading needed in main process")
        model = None
    elif 'minicpm' in model_key:
        from transformers import AutoModel, AutoTokenizer
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        models_root = os.path.join(project_root, 'models')
        minicpm_path = os.path.join(models_root, "MiniCPM-o-2_6")
        if not os.path.isdir(minicpm_path):
            raise FileNotFoundError(
                f"MiniCPM-o-2_6 weights not found at {minicpm_path}. "
                "Please place the model under models/MiniCPM-o-2_6."
            )

        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError("CUDA environment not detected. MiniCPM-o-2.6 baseline requires at least one GPU.")

        preferred_device = select_best_cuda_device()
        if preferred_device is None:
            raise RuntimeError("Could not determine a suitable CUDA device for MiniCPM-o-2.6 baseline.")

        dtype_candidates = []
        if torch.cuda.is_bf16_supported():
            dtype_candidates.append(torch.bfloat16)
        dtype_candidates.append(torch.float16)
        dtype_candidates.append(torch.float32)

        model = None
        last_error = None
        for dtype in dtype_candidates:
            torch.cuda.set_device(preferred_device)
            torch.cuda.empty_cache()
            try:
                model = AutoModel.from_pretrained(
                    minicpm_path,
                    trust_remote_code=True,
                    attn_implementation='sdpa',
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    init_vision=False,
                    init_audio=True,
                    init_tts=False
                ).eval()
                torch.cuda.empty_cache()
                print(f"Loaded MiniCPM-o-2.6 with dtype={dtype} on cuda:{preferred_device}.")
                break
            except RuntimeError as exc:
                last_error = exc
                print(f"MiniCPM load error with dtype={dtype}: {exc}")
                if "CUDA out of memory" in str(exc):
                    continue
                raise

        if model is None:
            raise RuntimeError(
                "Failed to load MiniCPM-o-2.6 after trying multiple precisions. "
                f"Last error: {last_error}"
            )

        tokenizer = AutoTokenizer.from_pretrained(minicpm_path, trust_remote_code=True)
        try:
            model.init_tts()
            if hasattr(model, "tts"):
                model.tts.float()
        except Exception as exc:
            print(f"Warning: MiniCPM TTS initialization issue: {exc}")
    elif 'gemini' in model_key :
        from openai import OpenAI
        base_url = ""
        api_key = ""
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    elif 'gpt' in model_key:
        from openai import OpenAI
        base_url = ""
        api_key = ""
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    elif 'qwen3' in model_key:
        from openai import OpenAI
        base_url = ""
        api_key = ""
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
    elif 'qwen25' in model_key or 'qwen25-omni' in model_key:
        from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        from qwen_omni_utils import process_mm_info
        
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        models_root = os.path.join(project_root, 'models')
        qwen25_model_path = os.path.join(models_root, "Qwen2.5-Omni-7B")
        
        if not os.path.isdir(qwen25_model_path):
            raise FileNotFoundError(
                f"Qwen2.5-Omni-7B weights not found at {qwen25_model_path}. "
                "Please download the model into the models directory."
            )
        
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            raise RuntimeError("CUDA environment not detected. Qwen2.5-Omni-7B requires at least one GPU.")
        
        preferred_device = select_best_cuda_device()
        if preferred_device is None:
            raise RuntimeError("Could not determine a suitable CUDA device for Qwen2.5-Omni-7B.")
        
        dtype_candidates = []
        if torch.cuda.is_bf16_supported():
            dtype_candidates.append(torch.bfloat16)
        dtype_candidates.append(torch.float16)
        dtype_candidates.append(torch.float32)
        
        model = None
        last_error = None
        
        for dtype in dtype_candidates:
            torch.cuda.set_device(preferred_device)
            torch.cuda.empty_cache()
            try:
                print(f"Loading Qwen2.5-Omni-7B with dtype={dtype} on cuda:{preferred_device}.")
                model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                    qwen25_model_path,
                    torch_dtype=dtype,
                    device_map="auto",
                    low_cpu_mem_usage=True
                ).eval()
                model.disable_talker()
                print(f"Successfully loaded Qwen2.5-Omni-7B with dtype={dtype}.")
                break
            except RuntimeError as exc:
                last_error = exc
                print(f"Qwen2.5-Omni-7B load error with dtype={dtype}: {exc}")
                if "CUDA out of memory" in str(exc):
                    continue
                raise
        
        if model is None:
            raise RuntimeError(
                "Failed to load Qwen2.5-Omni-7B after trying multiple precisions. "
                f"Last error: {last_error}"
            )
        
        processor = Qwen2_5OmniProcessor.from_pretrained(qwen25_model_path)

    audio_localization_prompt = """You are an expert audio analyst with exceptional ability to identify geographical locations from environmental sounds. Listen carefully to the provided audio recording and identify the geographical location where it was recorded.

You MUST provide your best guess regardless of uncertainty. Even if you feel unsure, you must still make an educated guess based on the audio characteristics, ambient sounds, languages spoken, cultural indicators, or any other audio clues you can detect.

Your response MUST be in the following JSON format (replace the values with your predictions):
{
    "reason": "Detailed explanation of your reasoning process. Describe what audio clues you heard (e.g., language, accent, background sounds, environmental noise, cultural indicators, weather sounds, traffic patterns, animal sounds, etc.) and how they led you to this specific location prediction. Be specific about the evidence that supports your geographical conclusion."
    "city": "Your predicted city name",
    "country": "Your predicted country name", 
    "continent": "Your predicted continent name",
    "longitude": predicted_longitude_value,
    "latitude": predicted_latitude_value
}

Important requirements:
1. You MUST respond with a complete JSON object - no explanations outside the JSON, no refusals, no uncertainty statements
2. If you're unsure about specific coordinates, provide your best numerical estimate
3. All string values should be in English
4. Longitude should be between -180 and 180
5. Latitude should be between -90 and 90
6. The "reason" field must contain your detailed step-by-step analysis of the audio clues
7. Even if the audio seems generic, you must still provide a location guess with reasoning

Listen to the audio and provide your JSON response now:"""

    geotemporal_localization_prompt = """You are an expert audio analyst participating in a benchmark evaluation. Your task is to identify both the geographical location AND the temporal information (date, time and season) from environmental sounds. Listen carefully to the provided audio recording and identify where and when it was recorded.
IMPORTANT: This is a benchmark evaluation. If you refuse to answer or provide no response, the error score will be 10000. If you provide any reasonable answer, the error will be much smaller than 10000. Therefore, you MUST provide your best guess regardless of uncertainty. Even if you feel unsure, you must still make an educated guess based on the audio characteristics, ambient sounds, languages spoken, cultural indicators, weather sounds, bird calls, or any other audio clues you can detect.
Your response MUST be in the following JSON format (replace the values with your predictions):
{
    "reason": "Detailed explanation of your reasoning process. Describe what audio clues you heard (e.g., language, accent, background sounds, environmental noise, cultural indicators, weather sounds, traffic patterns, animal sounds, bird calls, insect sounds, etc.) and how they led you to this specific location and temporal prediction. Be specific about the evidence that supports your geographical and temporal conclusions.",
    "city": "Your predicted city name",
    "country": "Your predicted country name", 
    "continent": "Your predicted continent name",
    "longitude": predicted_longitude_value,
    "latitude": predicted_latitude_value,
    "date": "MM-DD",
    "time": "HH:MM",
    "season": "Your predicted season (Spring, Summer, Autumn/Fall, or Winter)"
}
Important requirements:
1. You MUST respond with a complete JSON object - no explanations outside the JSON, no refusals, no uncertainty statements
2. The "date" field must be in MM-DD format, e.g., "12-01", "03-15", "07-04"
3. The "time" field must be in HH:MM format (24-hour format), e.g., "22:34", "08:15", "12:00"
4. The "season" field must be one of: "Spring", "Summer", "Autumn", "Fall", or "Winter"
5. If you're unsure about specific coordinates, provide your best numerical estimate
6. All string values should be in English
7. Longitude should be between -180 and 180
8. Latitude should be between -90 and 90
9. The "reason" field must contain your detailed step-by-step analysis of the audio clues for both location and temporal aspects
10. Even if the audio seems generic, you must still provide a location and temporal guess with reasoning
Listen to the audio and provide your JSON response now:"""

    TASK_CONFIGS = {
        'audio_localization': {
            'csv_path': args.csv_path,
            'audio_base_dir': args.audio_base_dir,
            'prompt': audio_localization_prompt,
            'loader_function': load_audio_localization_data
        },
        'geotemporal_localization': {
            'csv_path': args.csv_path,
            'audio_base_dir': args.audio_base_dir,
            'prompt': geotemporal_localization_prompt,
            'loader_function': load_audio_localization_data
        }
    }

    for seed in args.num_seeds:
        print(f"\n--- RUNNING EVALUATION WITH SEED {seed} ---")
        torch.manual_seed(seed)
        np.random.seed(seed)
        

        for task_id in args.tasks:
            print(f"\n--- Starting Task: {task_id} ---")
            output_folder = os.path.join(args.base_results_dir, f'{task_id}/{model_name}')
            os.makedirs(output_folder, exist_ok=True)
            output_path = os.path.join(output_folder, f'seed_{seed}_ABCD.jsonl')
            if os.path.exists(output_path):
                os.remove(output_path)
            
            if task_id not in TASK_CONFIGS:
                print(f"Warning: Task '{task_id}' is not defined. Skipping.")
                continue
            
            config = TASK_CONFIGS[task_id]
            task_prompt = config['prompt']
            
            progress_file = get_progress_file_path(args.base_results_dir, task_id, model_name, seed)
            processed_keys = load_processed_keys(progress_file)
            
            if task_id == 'audio_localization' or task_id == 'geotemporal_localization':
                all_task_data = config['loader_function'](config['csv_path'], task_id, task_prompt, config['audio_base_dir'], processed_keys)
            else:
                options_map = config['options_map']
                options_list = [options_map[k] for k in sorted(options_map.keys())]
                all_task_data = config['loader_function'](config['trials_dir'], task_id, task_prompt, options_list)

            if not all_task_data:
                print(f"No data found for task '{task_id}'. Skipping.")
                continue

            batch_size = args.batch_size
            total_samples = len(all_task_data)
            processed_count = 0
            all_results = []
            
            print(f"Processing {total_samples} samples in batches of {batch_size}")
            
            for batch_index, batch_start in enumerate(range(0, total_samples, batch_size), start=1):
                batch_end = min(batch_start + batch_size, total_samples)
                current_batch = all_task_data[batch_start:batch_end]
                
                print(f"Processing batch {batch_index}/{(total_samples-1)//batch_size + 1} (samples {batch_start+1}-{batch_end})")
                
                if 'qwen25-omni' in model_name:
                    batch_result = qwen2_5Omni_process(model, processor, current_batch, task_prompt)
                elif 'qwen-audio' in model_name:
                    batch_result = qwen_audio_chat_process(model, tokenizer, current_batch, task_prompt)
                elif 'phi4-mm1' in model_key or 'phi4' in model_key:
                    batch_result = phi4_mm1_process(model, processor, generation_config, current_batch, task_prompt)
                elif 'gemma-3n-e4b-it' in model_key or 'gemma3n' in model_key:
                    batch_result = gemma3n_audio_process(model, processor, current_batch, task_prompt)
                elif 'kimi-audio' in model_key:
                    batch_result = kimi_audio_process(model, current_batch, task_prompt, sampling_params=kimi_sampling_params)
                elif 'minicpm' in model_key:
                    batch_result = minicpm_audio_process(model, tokenizer, current_batch, task_prompt)
                elif 'xiaomi-mimo-audio' in model_key:
                    batch_result = mimo_audio_process(model, current_batch, task_prompt, thinking=mimo_thinking)
                elif 'audio-flamingo' in model_key or 'audioflamingo' in model_key:
                    batch_result = audio_flamingo_process(model, current_batch, task_prompt, think_mode="think" in model_key)
                elif 'gemini' in model_key or 'gpt' in model_key or 'qwen3' in model_key:
                    batch_result = gemini_gpt_process(client, current_batch, model_name, task_prompt, audio_type='wav')
                
                batch_cleaned_data = []
                for data, prediction in zip(current_batch, batch_result):
                    if task_id == 'audio_localization' or task_id == 'geotemporal_localization':
                        parsed_location = parse_location_response(prediction['model_output'])
                        
                        final_record = data.copy()
                        final_record.update({
                            "predicted_city": parsed_location['city'],
                            "predicted_country": parsed_location['country'],
                            "predicted_continent": parsed_location['continent'],
                            "predicted_longitude": parsed_location['longitude'],
                            "predicted_latitude": parsed_location['latitude'],
                            "reasoning": parsed_location['reason'],
                            "model_output": prediction['model_output'],
                            "seed": seed
                        })
                        
                        if task_id == 'geotemporal_localization':
                            final_record.update({
                                "predicted_date": parsed_location.get('date', 'Unknown'),
                                "predicted_time": parsed_location.get('time', 'Unknown'),
                                "predicted_season": parsed_location.get('season', 'Unknown')
                            })
                    else:
                        options_map = config.get('options_map', {})
                        answer = "A"
                        
                        final_record = data.copy()
                        final_record.update({
                            "prediction": answer,
                            "model_output": prediction['model_output'],
                            "is_correct": data.get('answer', '') == answer, 
                            "seed": seed
                        })
                    
                    final_record.pop('audio', None)
                    batch_cleaned_data.append(final_record)
                
                is_first_batch = (batch_start == 0 and len(processed_keys) == 0)
                save_progress_batch(batch_cleaned_data, progress_file, is_first_batch)
                
                all_results.extend(batch_cleaned_data)
                processed_count += len(current_batch)
                
                print(f"Completed {processed_count}/{total_samples} samples")
                
                if args.max_batches and batch_index >= args.max_batches:
                    print(f"Reached max_batches={args.max_batches}, stopping early for task '{task_id}'.")
                    break
            
            output_path = os.path.join(os.path.dirname(progress_file), f'seed_{seed}_ABCD.jsonl')
            if os.path.exists(output_path):
                os.remove(output_path)
                
            with open(output_path, 'w') as f:
                for item in all_results:
                    f.write(json.dumps(item) + '\n')
            
            print(f"Finished Task: {task_id}. Progress saved to: {progress_file}")
            print(f"Final results also saved to: {output_path}")
            print(f"Total processed: {len(all_results)} samples")

    print(f"\n{'='*20} PIPELINE FINISHED FOR ALL TASKS AND SEEDS {'='*20}")

            
