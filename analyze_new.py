import argparse
import pandas as pd
import numpy as np
import json
from geopy.distance import geodesic
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date, time
import random
import os

def parse_ast_label(label_str):
    if pd.isna(label_str) or label_str == '':
        return []
    
    labels = []
    if ':' in str(label_str):
        parts = str(label_str).split(';')
        for part in parts:
            if ':' in part:
                label_name = part.split(':')[0].strip()
                labels.append(label_name)
    
    return labels

def has_speech_label(labels):
    return any('Speech' in label or 'speech' in label.lower() for label in labels)



def load_and_merge_data(custom_place=None, custom_name=None):
    print("加载数据...")
    
    dataset = pd.read_csv('./data/geoLocalization_schema.csv')
    print(f"数据集大小: {len(dataset)}")
    
    task = "audio_localization"
    
    if custom_place and custom_name:
        model_list = [custom_place]
        model_name_list = [custom_name]
    else:
        place = 'gemini-2.5-pro'
        place_thinking = 'gemini-2.0-flash-thinking-exp'
        place_gpt = 'gpt-4o-audio-preview'
        place_lite = 'google/gemini-2.5-flash-lite'
        place_flash = 'google/gemini-2.5-flash'
        place_20_flash = 'google/gemini-2.0-flash-001'
        place_20_lite = 'google/gemini-2.0-flash-lite-001'
        place_3_pro = 'google/gemini-3-pro-preview'
        place_qwen3 = 'qwen3-omni'
        place_qwen2 = "qwen25-omni"

        phi4_mm1 = 'phi4'
        kimi_audio = 'kimi-audio'
        gemma_3n_e4b_it = 'gemma3n'
        mini_cpm_o_v2_6 = 'minicpm'
        xiaomi_mimo_audio = 'xiaomi-mimo-audio'
        xiaomi_mimo_audio_think = 'xiaomi-mimo-audio-think'

        model_list = [place_gpt, 
                      place_3_pro, place, place_flash, place_lite, 
                      place_thinking, place_20_flash, place_20_lite,
                      place_qwen3, place_qwen2, phi4_mm1,
                      kimi_audio, gemma_3n_e4b_it, mini_cpm_o_v2_6,
                      xiaomi_mimo_audio, xiaomi_mimo_audio_think]

        model_name_list = [
            'GPT-4o Audio Preview', 
            'Gemini 3 Pro', 'Gemini 2.5 Pro', 'Gemini 2.5 Flash', 'Gemini 2.5 Flash-Lite', 
            'Gemini 2.0 Flash Thinking', 'Gemini 2.0 Flash', 'Gemini 2.0 Flash-Lite', 
            'Qwen3-Omni', 'Qwen2.5-Omni',
            'Phi-4-MM1', 'Kimi-Audio', 'gemma-3n-E4B-it', 'MiniCPM-o-2.6', 'Mimo-audio', 'Mimo-audio-think'
        ]

    result_list = []
    
    for model, model_name in zip(model_list, model_name_list):
        results = pd.read_json(f'./results/{task}/{model}/seed_42_progress_fixed.json')
        
        merged = results.merge(dataset, left_on='question_id', right_on='key', how='left')
        

        merged = merged[(merged['human_sounds'] != 0) | (merged['animal'] != 0) | (merged['music'] != 0) | (merged['natural_sounds'] != 0) | (merged['channel'] != 0) | (merged['environment_and_background'] != 0) | (merged['sounds_of_things'] != 0) | (merged['source_ambiguous_sounds'] != 0)]

        print("after filter", len(merged))

        if 'human_sounds' in merged.columns:
            merged['human_sounds'] = pd.to_numeric(merged['human_sounds'], errors='coerce')
            merged['human_sounds'] = merged['human_sounds'].fillna(0.0)
            merged['has_speech'] = merged['human_sounds'] >= 0.1
        else:
            merged['has_speech'] = False
        
        if 'true_country' in merged.columns:
            merged['true_country'] = merged['true_country'].apply(normalize_country_name)
        if 'predicted_country' in merged.columns:
            merged['predicted_country'] = merged['predicted_country'].apply(normalize_country_name)
        
        if 'true_continent' in merged.columns:
            merged['true_continent'] = merged['true_continent'].apply(normalize_continent_name)
        if 'predicted_continent' in merged.columns:
            merged['predicted_continent'] = merged['predicted_continent'].apply(normalize_continent_name)
        
        result_list.append(merged)
    
    df_random = result_list[0].copy()
    df_random['predicted_latitude'] = np.random.uniform(-90, 90, len(df_random))
    df_random['predicted_longitude'] = np.random.uniform(-180, 180, len(df_random))
    result_list.append(df_random)
    model_name_list.append("RANDOM")

    return result_list, model_name_list

def is_refusal_response(value):
    if pd.isna(value):
        return True
    
    str_value = str(value).strip()
    
    refusal_indicators = ['0', '-1', '', 'null', 'none', 'n/a', 'na', 'unknown', 'unable', 'cannot',"Unknown", "0.0"]
    
    return str_value.lower() in refusal_indicators

def analyze_refusal_rate(df):
    partial_refusal = df.apply(lambda row: 
        is_refusal_response(row['predicted_latitude']) or 
        is_refusal_response(row['predicted_longitude']) or
        is_refusal_response(row['predicted_city']) or
        is_refusal_response(row['predicted_country']) or
        is_refusal_response(row['predicted_continent']), axis=1)
    total_samples = len(df)
    partial_refusal_rate = partial_refusal.sum() / total_samples
    
    return partial_refusal_rate

def calculate_distance(true_lat, true_lon, pred_lat, pred_lon):
    try:
        if is_refusal_response(pred_lat) or is_refusal_response(pred_lon):
            pred_lat = random.uniform(-90, 90)
            pred_lon = random.uniform(-180, 180)
            return 10000
        true_coord = (float(true_lat), float(true_lon))
        pred_coord = (float(pred_lat), float(pred_lon))
        return geodesic(true_coord, pred_coord).kilometers
    except:
        return np.nan

def calculate_distance_error(df):
    df['distance_error_km'] = df.apply(
        lambda row: calculate_distance(
            row['true_latitude'], row['true_longitude'],
            row['predicted_latitude'], row['predicted_longitude']
        ), axis=1
    )
    return df['distance_error_km'].mean()

def analyze_accuracy(df, category_col, true_col, pred_col):
    valid_data = df.dropna(subset=[true_col, pred_col])
    if len(valid_data) == 0:
        return 0.0, 0
    
    correct = valid_data[true_col].str.lower() == valid_data[pred_col].str.lower()
    accuracy = correct.sum() / len(valid_data)
    return accuracy, len(valid_data)

def analyze_location_accuracy(df):
    city_acc, city_count = analyze_accuracy(df, 'city', 'true_city', 'predicted_city')
    
    df_copy = df.copy()
    df_copy['true_country'] = df_copy['true_country'].apply(normalize_country_name)
    df_copy['predicted_country'] = df_copy['predicted_country'].apply(normalize_country_name)
    country_acc, country_count = analyze_accuracy(df_copy, 'country', 'true_country', 'predicted_country')
    
    df_copy['true_continent'] = df_copy['true_continent'].apply(normalize_continent_name)
    df_copy['predicted_continent'] = df_copy['predicted_continent'].apply(normalize_continent_name)
    continent_acc, continent_count = analyze_accuracy(df_copy, 'continent', 'true_continent', 'predicted_continent')

    return continent_acc, country_acc, city_acc


def analyze_distance_thresholds(df):
    if 'distance_error_km' not in df.columns:
        df['distance_error_km'] = df.apply(
            lambda row: calculate_distance(
                row['true_latitude'], row['true_longitude'],
                row['predicted_latitude'], row['predicted_longitude']
            ), axis=1
        )
    
    valid_distances = df['distance_error_km'].dropna()
    total_samples = len(valid_distances)
    
    if total_samples == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    less_than_1km = (valid_distances < 1).sum() / total_samples
    less_than_10km = (valid_distances < 10).sum() / total_samples
    less_than_100km = (valid_distances < 100).sum() / total_samples
    less_than_1000km = (valid_distances < 1000).sum() / total_samples
    
    return less_than_1km, less_than_10km, less_than_100km, less_than_1000km

def analyze_scene_distances(df):
    scene_cols = [
        'animal', 'music', 'natural_sounds', 'channel',
        'environment_and_background', 'sounds_of_things'
    ]

    speech_distance = df[df['human_sounds'] >= 0.1]['distance_error_km'].mean()
    non_speech_distance = df[df['human_sounds'] < 0.1]['distance_error_km'].mean()

    result_list = [speech_distance, non_speech_distance]
    for scene_col in scene_cols:
        scene_distance = df[(df[scene_col] >= 0.1) & (df['human_sounds'] < 0.1)]['distance_error_km'].mean()
        result_list.append(scene_distance)
    return result_list[0], result_list[1], result_list[2], result_list[3], result_list[4], result_list[5], result_list[6], result_list[7]

def normalize_continent_name(continent_name):
    if pd.isna(continent_name) or continent_name == '':
        return None
    continent_str = str(continent_name).strip()
    if continent_str == '':
        return None
    
    continent_lower = continent_str.lower()
    
    if continent_lower.startswith('the '):
        continent_lower = continent_lower[4:].strip()
    
    if 'arctic' in continent_lower:
        return 'Antarctica'
    
    normalized = continent_str.lower().title()
    if normalized == 'Australia':
        normalized = 'Oceania'
    return normalized

def normalize_country_name(country_name):
    if pd.isna(country_name) or country_name == '':
        return None
    country_str = str(country_name).strip()
    if country_str == '':
        return None
    
    country_lower = country_str.lower()
    
    if 'taiwan' in country_lower:
        return 'China'
    
    if country_lower.startswith('the '):
        country_lower = country_lower[4:].strip()
    
    country_variants = {
        'usa': 'United States',
        'us': 'United States',
        'united states': 'United States',
        'united states of america': 'United States',
        'america': 'United States',
        'u.s.a.': 'United States',
        'u.s.': 'United States',
        
        'united kingdom': 'United Kingdom',
        'uk': 'United Kingdom',
        'great britain': 'United Kingdom',
        'britain': 'United Kingdom',
        'england': 'United Kingdom',
        'u.k.': 'United Kingdom',
        
        'netherlands': 'Netherlands',
        'the netherlands': 'Netherlands',
        'holland': 'Netherlands',
        
        'czechia': 'Czechia',
        'czech republic': 'Czechia',
        'czech': 'Czechia',
        
        'turkey': 'Turkey',
        'türkiye': 'Turkey',
        'turkiye': 'Turkey',
        
        'russia': 'Russia',
        'russian federation': 'Russia',
        
        'south korea': 'South Korea',
        'korea': 'South Korea',
        'republic of korea': 'South Korea',
        's. korea': 'South Korea',
        
        'north korea': 'North Korea',
        'democratic people\'s republic of korea': 'North Korea',
        'dprk': 'North Korea',
        'n. korea': 'North Korea',
        
        'china': 'China',
        'people\'s republic of china': 'China',
        'prc': 'China',
        'mainland china': 'China',
        
        'japan': 'Japan',
        'nippon': 'Japan',
        
        'germany': 'Germany',
        'deutschland': 'Germany',
        'federal republic of germany': 'Germany',
        
        'france': 'France',
        'french republic': 'France',
        
        'italy': 'Italy',
        'italia': 'Italy',
        
        'spain': 'Spain',
        'españa': 'Spain',
        'espana': 'Spain',
        
        'brazil': 'Brazil',
        'brasil': 'Brazil',
        
        'mexico': 'Mexico',
        'méxico': 'Mexico',
        
        'india': 'India',
        'bharat': 'India',
        
        'australia': 'Australia',
        
        'south africa': 'South Africa',
        
        'canada': 'Canada',
        
        'argentina': 'Argentina',
        
        'chile': 'Chile',
        
        'peru': 'Peru',
        'perú': 'Peru',
        
        'colombia': 'Colombia',
        
        'greece': 'Greece',
        'hellas': 'Greece',
        
        'switzerland': 'Switzerland',
        'suisse': 'Switzerland',
        'schweiz': 'Switzerland',
        
        'austria': 'Austria',
        'österreich': 'Austria',
        'osterreich': 'Austria',
        
        'sweden': 'Sweden',
        'sverige': 'Sweden',
        
        'norway': 'Norway',
        'norge': 'Norway',
        
        'denmark': 'Denmark',
        'danmark': 'Denmark',
        
        'finland': 'Finland',
        'suomi': 'Finland',
        
        'poland': 'Poland',
        'polska': 'Poland',
        
        'hungary': 'Hungary',
        'magyarország': 'Hungary',
        'magyarorszag': 'Hungary',
        
        'romania': 'Romania',
        'românia': 'Romania',
        
        'bulgaria': 'Bulgaria',
        
        'ukraine': 'Ukraine',
        
        'belarus': 'Belarus',
        'belorussia': 'Belarus',
        
        'israel': 'Israel',
        
        'palestine': 'Palestine',
        
        'saudi arabia': 'Saudi Arabia',
        
        'iran': 'Iran',
        'islamic republic of iran': 'Iran',
        
        'iraq': 'Iraq',
        
        'egypt': 'Egypt',
        'arab republic of egypt': 'Egypt',
        
        'nigeria': 'Nigeria',
        
        'kenya': 'Kenya',
        
        'thailand': 'Thailand',
        'siam': 'Thailand',
        
        'vietnam': 'Vietnam',
        'viet nam': 'Vietnam',
        
        'indonesia': 'Indonesia',
        
        'malaysia': 'Malaysia',
        
        'philippines': 'Philippines',
        'the philippines': 'Philippines',
        
        'singapore': 'Singapore',
        
        'new zealand': 'New Zealand',
        
        'ireland': 'Ireland',
        'éire': 'Ireland',
        'eire': 'Ireland',
        
        'portugal': 'Portugal',
        
        'belgium': 'Belgium',
        
        'luxembourg': 'Luxembourg',
        'luxemburg': 'Luxembourg',
    }
    
    country_lower_clean = country_lower.replace(',', '').replace('.', '').replace(';', '').strip()
    
    if country_lower_clean in country_variants:
        return country_variants[country_lower_clean]
    
    normalized = country_str.strip()
    if normalized.lower().startswith('the '):
        normalized = 'The ' + normalized[4:].strip().title()
    else:
        normalized = normalized.title()
    
    return normalized


def main():
    parser = argparse.ArgumentParser(description='分析声音定位实验结果')
    parser.add_argument('--place', type=str, default=None,
                        help='模型路径 (e.g., gemini-2.5-pro, google/gemini-3-pro-preview)')
    parser.add_argument('--name', type=str, default=None,
                        help='模型显示名称 (e.g., "Gemini 2.5 Pro")')
    args = parser.parse_args()
    
    if (args.place is None) != (args.name is None):
        parser.error("--place 和 --name 必须同时提供或同时不提供")
    
    print("开始分析声音定位实验结果...")
    
    result_list, model_name_list = load_and_merge_data(args.place, args.name)

    error_list = []
    for result in result_list:
        error_list.append(calculate_distance_error(result))

    continent_accuracy_list = []
    country_accuracy_list = []
    city_accuracy_list = []
    for result in result_list:
        continent_rate, country_rate, city_rate = analyze_location_accuracy(result)
        continent_accuracy_list.append(continent_rate)
        country_accuracy_list.append(country_rate)
        city_accuracy_list.append(city_rate)
    
    reject_rate_list = []
    for result in result_list:
        reject_rate_list.append(analyze_refusal_rate(result))

    less_1km_list = []
    less_10km_list = []
    less_100km_list = []
    less_1000km_list = []
    for result in result_list:
        less_1km, less_10km, less_100km, less_1000km = analyze_distance_thresholds(result)
        less_1km_list.append(less_1km)
        less_10km_list.append(less_10km)
        less_100km_list.append(less_100km)
        less_1000km_list.append(less_1000km)
    
    speech_distance_list = []
    non_speech_distance_list = []
    animal_distance_list = []
    music_distance_list = []
    natural_sounds_distance_list = []
    channel_distance_list = []
    environment_and_background_distance_list = []
    sounds_of_things_distance_list = []
    for result in result_list:
        speech_distance, non_speech_distance, animal_distance, music_distance, natural_sounds_distance, channel_distance, environment_and_background_distance, sounds_of_things_distance = analyze_scene_distances(result)
        speech_distance_list.append(speech_distance)
        non_speech_distance_list.append(non_speech_distance)
        animal_distance_list.append(animal_distance)
        music_distance_list.append(music_distance)
        natural_sounds_distance_list.append(natural_sounds_distance)
        channel_distance_list.append(channel_distance)
        environment_and_background_distance_list.append(environment_and_background_distance)
        sounds_of_things_distance_list.append(sounds_of_things_distance)

    
    columns = ['model', 'distance error', 'continent accuracy', 'country accuracy', 'city accuracy', 'reject rate', 
               '<1km', '<10km', '<100km', '<1000km',
               'speech distance', 'non speech distance', 'animal distance', 'music distance', 'natural sounds distance', 'channel distance', 'environment and background distance', 'sounds of things distance']
    df = pd.DataFrame(columns=columns)
    for i in range(len(model_name_list)):
        df.loc[i] = [model_name_list[i], round(error_list[i], 2), round(continent_accuracy_list[i], 2), round(country_accuracy_list[i], 2), 
        round(city_accuracy_list[i], 2), round(reject_rate_list[i], 2), 
        round(less_1km_list[i], 2), round(less_10km_list[i], 2), round(less_100km_list[i], 2), round(less_1000km_list[i], 2),
        round(speech_distance_list[i], 2), round(non_speech_distance_list[i], 2), 
        round(animal_distance_list[i], 2), round(music_distance_list[i], 2), round(natural_sounds_distance_list[i], 2), round(channel_distance_list[i], 2), 
        round(environment_and_background_distance_list[i], 2), round(sounds_of_things_distance_list[i], 2)]
    df.to_csv('results.csv', index=False)
    print("\n分析完成！")

if __name__ == "__main__":
    main()