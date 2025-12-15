# 🌍 AGL1K: Audio Geo-Localization 1K Benchmark


**🎯 The First World-Class Audio Language Model Geo-Localization Benchmark**

*Can AI hear where a sound comes from? Let's find out!*

**🎮 [Try it Online on Hugging Face Spaces](https://huggingface.co/spaces/RisingZhang/AudioGeoLocalization)** - Experience audio geo-localization interactively!

---

## 🌟 What is AGL1K?

**AGL1K** is a pioneering benchmark designed to evaluate the **geographical localization** capabilities of Audio Language Models (ALMs). Given only an audio recording, can an AI determine where on Earth it was recorded?


```
🎧 Audio Input → 🤖 ALM → 📍 Location Prediction (City, Country, Continent, Coordinates)
```

---

## 🗺️ Dataset Overview


| Statistic        | Value                               |
| ---------------- | ----------------------------------- |
| Total Samples    | 1,444                               |
| Countries        | 74                                  |
| Continents       | 6                                   |
| Audio Duration   | 10s - 180s                          |
| Sound Categories | Human, Animal, Music, Nature, Urban |

---

## 📊 Model Leaderboard

We evaluated **16 state-of-the-art Audio Language Models** on AGL1K:

### Closed-Source Models

| Model                     | Distance Error ↓ | Cont. Acc. ↑ | Country Acc. ↑ | City Acc. ↑ |
| ------------------------- | :--------------: | :----------: | :------------: | :---------: |
| 🥇 Gemini 3 Pro            |   **2180.57**    |   **0.82**   |    **0.51**    |  **0.11**   |
| 🥈 Gemini 2.5 Pro          |     2521.97      |     0.78     |      0.49      |  **0.11**   |
| 🥉 Gemini 2.0 Flash        |     2906.31      |     0.73     |      0.40      |    0.08     |
| Gemini 2.0 Flash-Thinking |     2991.51      |     0.73     |      0.39      |    0.07     |
| Gemini 2.0 Flash-Lite     |     3223.85      |     0.71     |      0.38      |    0.06     |
| Gemini 2.5 Flash          |     3558.37      |     0.65     |      0.39      |    0.07     |
| GPT-4o Audio Preview      |     4067.87      |     0.61     |      0.37      |    0.05     |
| Gemini 2.5 Flash-Lite     |     4373.89      |     0.55     |      0.28      |    0.03     |

### Open-Source Models

| Model            | Distance Error ↓ | Cont. Acc. ↑ | Country Acc. ↑ | City Acc. ↑ |
| ---------------- | :--------------: | :----------: | :------------: | :---------: |
| Mimo-audio       |     4853.25      |     0.54     |      0.20      |    0.03     |
| Mimo-audio-think |     5008.01      |     0.51     |      0.20      |    0.03     |
| Qwen3-Omni       |     5174.36      |     0.47     |      0.25      |    0.02     |
| Qwen2.5-Omni     |     5476.83      |     0.45     |      0.26      |    0.02     |
| Kimi-Audio       |     5590.20      |     0.43     |      0.22      |    0.02     |
| Gemma-3n-E4B-it  |     5815.46      |     0.41     |      0.17      |    0.01     |
| Phi-4-MM1        |     6462.43      |     0.33     |      0.08      |    0.01     |
| MiniCPM-o-2.6    |     6600.83      |     0.44     |      0.22      |    0.02     |

### Baseline

| Model  | Distance Error |
| ------ | :------------: |
| RANDOM |    9869.01     |


- **Distance Error**: Average geodesic distance (in km) between predicted and ground-truth coordinates. Lower is better.
- **Cont. Acc.**: Continent-level accuracy. Higher is better.
- **Country Acc.**: Country-level accuracy. Higher is better.
- **City Acc.**: City-level accuracy. Higher is better.

> 📌 **Key Finding:** Audio geo-localization is tractable! Gemini 3 Pro achieves 51% country-level accuracy with a 2180 km average distance error, demonstrating that modern ALMs can extract meaningful geographical information from audio alone.

---

## 🎧 Audio Examples

Experience the challenge of audio geo-localization with these representative cases:

### Case 1: Belgium - Student Discussions

|                   AI-Generated Image                   |                 Audio Recording                 |
| :----------------------------------------------------: | :---------------------------------------------: |
| <img src="assests/sampleImage/case1.png" width="300"/> | 🔊 [Listen](assests/sampleAudio/case1_LLN05.mp3) |

**Scene:** Students discussing after class at the Institut des Langues Vivantes  
**Audio Cues:** French conversations, university ambiance, European urban soundscape
**Gemini 3 Pro Reasoning:** The audio recording captures the ambience of a busy indoor public space, characterized by high reverberation, the sound of heavy doors opening and slamming shut, and footsteps on hard flooring, which strongly suggests a school hallway, university corridor, or a similar educational institution during a break. The dominant language spoken is distinctly French. The demographic sounds like young adults or students engaged in casual chatter. Crucially, the accent and intonation of the speakers provide the geographical key. The French spoken lacks the specific 'sing-song' melody of Swiss French or the distinct nasal twang of Quebecois. It also differs slightly from standard Parisian French, exhibiting a flatter intonation and specific vowel qualities (a slightly more guttural or 'back of the throat' resonance) that are characteristic of Belgian French speakers. Given the density of the crowd and the accent markers, Brussels, as a major francophone education hub in Belgium, is the most probable location.


---

### Case 2: Morocco - Ancient Medina


|                   AI-Generated Image                   |                       Audio Recording                        |
| :----------------------------------------------------: | :----------------------------------------------------------: |
| <img src="assests/sampleImage/case2.png" width="300"/> | 🔊 [Listen](assests/sampleAudio/case2_20020207EssaouiraRempartsMuezzin.mp3) |

**Scene:** Muezzin call echoing from the ancient battlements of the old Medina  
**Audio Cues:** Islamic call to prayer, North African atmosphere, ocean breeze
**Gemini 3 Pro Reasoning: **The audio recording provides a distinct sonic fingerprint pointing to the Maghreb region of North Africa. The most prominent initial clue is the Adhan (Islamic call to prayer) performed in a style typical of the region, immediately placing the location in a Muslim-majority country. This is overlaid with the loud, persistent squawking of seagulls and significant wind noise, which strongly suggests a coastal city on a major body of water. The traffic noise includes the distinctive high-pitched whine of 2-stroke mopeds or scooters navigating narrow streets, a common soundscape in Moroccan medinas. In the second half of the recording, the sound shifts to powerful, rhythmic waves crashing against a shore, sounding more like the heavy swells of the Atlantic Ocean than the typically calmer Mediterranean. Linguistically, shouts of 'Yallah' (Arabic) and what appears to be French phrases are audible, reflecting the bilingual heritage of Morocco. The specific combination of intense wind, overwhelming seagull presence, the Adhan, and heavy Atlantic surf is the hallmark of Essaouira, Morocco, famously known as the 'Wind City of Africa' with its historic fortified port.

---

### Case 3: Sweden - Urban Transit Hub

|                   AI-Generated Image                   |                       Audio Recording                        |
| :----------------------------------------------------: | :----------------------------------------------------------: |
| <img src="assests/sampleImage/case3.png" width="300"/> | 🔊 [Listen](assests/sampleAudio/case3_190221093904stockcityterminalen.mp3) |

**Scene:** Main train station (Cityterminalen) with travelers and announcements  
**Audio Cues:** Swedish PA announcements, luggage wheels, Nordic transit ambiance
**Gemini 3 Pro Reasoning:** The audio environment is a large, reverberant hall with the distinct sounds of rolling suitcases, closing doors, and murmuring crowds, indicating a busy train station. At 0:48, faint conversation in Swedish can be heard. The definitive geographical evidence begins at 1:08 with a standard electronic chime followed by a female automated station announcement in Swedish. The voice announces: 'Till tåg [number]... mot Uppsala... via Arlanda C, Knivsta... Var god stig ombord, spår 6' (To train... towards Uppsala... via Arlanda Central, Knivsta... Please board, track 6). This specific route—traveling northbound to the city of Uppsala with stops at Arlanda Airport (Arlanda C) and Knivsta—originates from the main railway hub in the region. Therefore, this is recorded at Stockholm Central Station, specifically on the platform level.

---

### Case 4: Germany - Urban Nature

|                   AI-Generated Image                   |                       Audio Recording                        |
| :----------------------------------------------------: | :----------------------------------------------------------: |
| <img src="assests/sampleImage/case4.png" width="300"/> | 🔊 [Listen](assests/sampleAudio/case4_AmselanDietersFensterJuni2003Edit.mp3) |

**Scene:** European blackbird (*Amsel*) singing from a window in June  
**Audio Cues:** Distinctive blackbird song, urban garden atmosphere, Central European bird species
**Gemini 3 Pro Reasoning:** The most prominent audio clue is the distinct, fluty, and melodious song of a Common Blackbird (Turdus merula). This bird is widespread across Europe and is famous for its adaptability to urban environments. The acoustic characteristics of the recording are equally telling: there is a noticeable reverberation or echo to the bird's song and the background noises, which strongly suggests the recording was taken in a 'Hinterhof'—a large, enclosed inner courtyard formed by block-style apartment buildings. This architectural style is iconic to Central European cities, particularly Berlin. In the background, there is a steady 'city hum' of distant traffic without aggressive honking, and the mechanical sounds at the beginning (a heavy rumble and metallic squeal) resemble a tram (Straßenbahn) or urban train passing nearby. The combination of the specific 'courtyard acoustics,' the pervasive blackbird song, and the sounds of European public transport infrastructure creates a sonic signature that is widely recognized as the soundscape of a residential neighborhood in Berlin, Germany.


---

## 🚀 Quick Start

**🎮 [Try the Interactive Demo](https://huggingface.co/spaces/RisingZhang/AudioGeoLocalization)** - Experience audio geo-localization without installation!

### Installation

```bash
git clone https://github.com/your-username/AGL1K.git
cd AGL1K
```

### Download Audio Data

**⚠️ Important:** Before running the benchmark, you need to download the audio files:

1. Download the audio data from: https://bhpan.buaa.edu.cn/link/AA0667E919A78342A09800447948D5175E
2. Extract all contents to the `data/audios/` folder

```bash
# After downloading and extracting, your directory structure should look like:
# data/
#   ├── audios/
#   │   ├── audio_file_1.mp3
#   │   ├── audio_file_2.mp3
#   │   └── ...
#   └── geoLocalization_schema.csv
```

**Note:** The `data/audios/` folder is excluded from the git repository due to file size. You must download it separately to run the benchmark.

### Running Evaluation

```bash
# Evaluate a model on audio localization task
python openllm.py --model gemini-2.5-flash \
                  --csv_path data/geoLocalization_schema.csv \
                  --audio_base_dir data/audios \
                  --tasks audio_localization
```

### Run Experiment

**Step 1: Structure the raw CSV data**

The evaluation script outputs raw CSV files with embedded newlines. Use `fix_csv_v2.py` to convert them into structured JSON format:

```bash
# Convert raw CSV to structured JSON
python fix_csv_v2.py --place gemini-2.5-pro
```

**Step 2: Generate evaluation results**

Use `analyze_new.py` to compute metrics and output results to `results.csv`:

```bash
# Analyze all predefined models
python analyze_new.py

# Or analyze a specific model
python analyze_new.py --place google/gemini-3-pro-preview --name "Gemini 3 Pro"
```

The output `results.csv` contains the following metrics for each model:
- **Distance Error**: Average geodesic distance (km) between predicted and ground-truth coordinates
- **Continent/Country/City Accuracy**: Location prediction accuracy at different granularities
- **Reject Rate**: Proportion of samples where the model refused to predict
- **Distance Thresholds**: Proportion of predictions within 1km, 10km, 100km, 1000km

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

- 🐛 Report bugs and issues
- 💡 Suggest new features or models to evaluate  
- 📊 Share your evaluation results
- 🎵 Contribute new audio samples

---

## 📜 License

This benchmark is released for **research purposes only**. Please refer to our paper for detailed terms of use.

---

## 🙏 Acknowledgments

- Audio samples sourced from [Aporee Sound Maps](https://aporee.org/maps/)

- Thanks to all model providers for API access

---

**🌍 Where in the world does this sound come from? Let's find out together!**