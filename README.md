# kokoro-tts-swift

text-to-speech in Swift. give it text, get 24kHz audio back.

Kokoro-82M on the Apple Neural Engine via CoreML. 6-16x real-time on M-series depending on input length. handles any length text -- automatic chunking, streaming, voice selection, speed control.

## install

```swift
// Package.swift
dependencies: [
    .package(url: "https://github.com/Jud/kokoro-tts-swift.git", from: "0.1.0"),
]
```

models (~640MB) download automatically on first use. or grab them manually:

```bash
./scripts/download-models.sh
```

## usage

basic synthesis -- give it text, get samples back:

```swift
import KokoroTTS

let engine = try KokoroEngine(modelDirectory: modelPath)
engine.warmUp()

let result = try engine.synthesize(text: "hello world", voice: "af_heart")
// result.samples → 24kHz mono PCM float array
// result.duration → audio length in seconds
// result.realTimeFactor → how much faster than real-time
```

### streaming

for long text, `speak()` streams audio as it's synthesized. each buffer is playback-ready -- consumers just schedule and go:

```swift
let audioEngine = AVAudioEngine()
let player = AVAudioPlayerNode()
audioEngine.attach(player)
audioEngine.connect(player, to: audioEngine.mainMixerNode, format: KokoroEngine.audioFormat)
try audioEngine.start()
player.play()

for await buffer in try engine.speak("any length text...", voice: "af_heart") {
    player.scheduleBuffer(buffer)
}
```

no manual chunking. no PCM conversion. text goes in, playback-ready `AVAudioPCMBuffer` comes out.

### speed control

```swift
let slow = try engine.synthesize(text: "take your time", voice: "af_heart", speed: 0.7)
let fast = try engine.synthesize(text: "let's go", voice: "af_heart", speed: 1.5)
```

## CLI

```bash
make install
```

```bash
kokoro say "hello from the terminal"
kokoro say -v am_adam -s 1.2 "speed it up"
kokoro say -o output.wav "save to file"
kokoro say --stream "start hearing audio before synthesis finishes"
echo "long article" | kokoro say --stream
kokoro say --list-voices
kokoro daemon start   # keep models loaded for fast repeat synthesis
kokoro daemon stop
```

`--stream` starts playback as soon as the first chunk is ready. reports time-to-first-audio and per-chunk stats.

models download on first run. `--model-dir <path>` to override.

## how it works

```mermaid
graph LR
    A[text] --> B[G2P]
    B --> C[phonemes]
    C --> D[tokenizer]
    D --> E[token IDs]
    E --> F1[CoreML frontend<br/>predictor + SineGen, CPU]
    G[voice embedding] --> F1
    F1 --> F2[CoreML backend<br/>decoder + iSTFTNet, ANE]
    F2 --> H[24kHz audio]
```

text goes through an english G2P pipeline -- lexicon lookup, morphological stemming, number expansion. unknown words hit a fallback chain:

1. **CamelCase splitting** -- "AVAudioPlayer" → "AV" + "Audio" + "Player", each part resolved independently
2. **BART neural G2P** -- per-part, so "AVKubernetesPlayer" → AV(spelled) + Kubernetes(BART) + Player(lexicon)
3. **letter spelling** -- last resort for acronyms

phonemes get tokenized and fed to the CoreML models with a voice style embedding.

the engine picks the smallest model bucket that fits:

| bucket | max tokens | audio length |
|--------|-----------|-------------|
| small  | 124       | ~5s         |
| medium | 242       | ~10s        |

longer text gets chunked at sentence boundaries, merged where possible, and stitched with silence gaps. `synthesize()` returns the full result. `speak()` streams chunks as `AVAudioPCMBuffer` via `AsyncStream`.

## architecture

```mermaid
graph TD
    subgraph "text pipeline"
        T1[text] --> T2[paragraph split]
        T2 --> T3[G2P phonemization]
        T3 --> T4[punctuation-aware chunking]
        T4 --> T5[tokenization + merging]
    end

    subgraph "inference"
        T5 --> I1[bucket selection]
        I1 --> I2[frontend: predictor + SineGen<br/>CPU]
        V[voice store] --> I2
        I2 --> I3[backend: decoder + iSTFTNet<br/>Neural Engine]
        I3 --> I4[PCM samples]
    end

    subgraph "output"
        I4 --> P1[fades + post-processing]
        P1 -->|synthesize| O1[full audio array]
        P1 -->|speak| O2[AsyncStream&lt;AVAudioPCMBuffer&gt;]
    end
```

## model

- **architecture**: Kokoro-82M -- StyleTTS2 encoder + iSTFTNet vocoder, split frontend (CPU) + backend (ANE)
- **sample rate**: 24kHz mono
- **voices**: style embedding vectors, one JSON per voice preset
- **runtime**: CoreML -- frontend on CPU, backend on Apple Neural Engine
- **platform**: macOS 15+, iOS 18+

## license

Apache 2.0
