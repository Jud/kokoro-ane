import Testing

@testable import KokoroCoreML

@Suite("KokoroEngine BOS trim")
struct KokoroEngineBOSTrimTests {
    @Test("Fallback trims short BOS spans instead of keeping the whole allocation")
    func fallbackTrimsShortBOSSpans() {
        let onsetMarginSamples = Int(Float(KokoroEngine.sampleRate) * 0.005)

        for leadingFrames in 1...3 {
            let leadSamples = leadingFrames * KokoroEngine.hopSize
            let samples = [Float](repeating: 0.01, count: leadSamples + KokoroEngine.hopSize)
            let silentSamples = [Float](repeating: 0, count: leadSamples + KokoroEngine.hopSize)

            let trimSamples = KokoroEngine.adaptiveLeadingBOSTrimSamples(
                in: samples, leadSamples: leadSamples, speed: 1.0)
            let silentTrimSamples = KokoroEngine.adaptiveLeadingBOSTrimSamples(
                in: silentSamples, leadSamples: leadSamples, speed: 1.0)

            #expect(trimSamples == leadSamples - onsetMarginSamples)
            #expect(silentTrimSamples == leadSamples - onsetMarginSamples)
        }
    }

    @Test("Detected onset keeps adaptive preroll before the first phoneme")
    func detectedOnsetKeepsAdaptivePreroll() {
        let leadSamples = 4 * KokoroEngine.hopSize
        let onsetOffset = leadSamples - 960
        var samples = [Float](repeating: 0, count: leadSamples + KokoroEngine.hopSize)
        for index in onsetOffset..<leadSamples {
            samples[index] = 0.1
        }

        let trimSamples = KokoroEngine.adaptiveLeadingBOSTrimSamples(
            in: samples, leadSamples: leadSamples, speed: 1.0)

        #expect(trimSamples == onsetOffset - 120)
    }

    @Test("Boundary-near onset can use post-BOS samples for sustain")
    func boundaryNearOnsetUsesPostBOSSustain() {
        let leadSamples = 4 * KokoroEngine.hopSize
        let onsetOffset = leadSamples - 120
        var samples = [Float](repeating: 0, count: leadSamples + KokoroEngine.hopSize)
        for index in onsetOffset..<samples.count {
            samples[index] = 0.1
        }

        let trimSamples = KokoroEngine.adaptiveLeadingBOSTrimSamples(
            in: samples, leadSamples: leadSamples, speed: 1.0)

        #expect(trimSamples == leadSamples - 960)
    }

    @Test("Post-BOS peaks do not raise the onset threshold")
    func postBOSPeaksDoNotRaiseOnsetThreshold() {
        let leadSamples = 4 * KokoroEngine.hopSize
        let onsetOffset = leadSamples - 240
        var samples = [Float](repeating: 0, count: leadSamples + KokoroEngine.hopSize)
        for index in onsetOffset..<leadSamples {
            samples[index] = 0.02
        }
        for index in leadSamples..<samples.count {
            samples[index] = 0.5
        }

        let trimSamples = KokoroEngine.adaptiveLeadingBOSTrimSamples(
            in: samples, leadSamples: leadSamples, speed: 1.0)

        #expect(trimSamples == leadSamples - 960)
    }

    @Test("EOS fallback preserves minPostroll when no drop is detected")
    func eosFallbackPreservesMinPostroll() {
        let trailSamples = 4 * KokoroEngine.hopSize
        let totalSamples = trailSamples + KokoroEngine.hopSize
        let uniform = [Float](repeating: 0.01, count: totalSamples)
        let silent = [Float](repeating: 0, count: totalSamples)

        let uniformTrim = KokoroEngine.adaptiveTrailingEOSTrimSamples(
            in: uniform, trailSamples: trailSamples, speed: 1.0)
        let silentTrim = KokoroEngine.adaptiveTrailingEOSTrimSamples(
            in: silent, trailSamples: trailSamples, speed: 1.0)

        // No drop → fallback minPostroll = 40ms = 960 samples.
        #expect(uniformTrim == trailSamples - 960)
        #expect(silentTrim == trailSamples - 960)
    }

    @Test("EOS detected drop expands postroll up to maxPostroll")
    func eosDetectedDropAdaptsPostroll() {
        let trailSamples = 8 * KokoroEngine.hopSize  // 4800 samples (200ms)
        let totalSamples = trailSamples + KokoroEngine.hopSize
        let eosStart = totalSamples - trailSamples
        // Drop boundary at +5 windows (600 samples = 25ms) into EOS.
        let dropAt = eosStart + 5 * 120
        var samples = [Float](repeating: 0.01, count: totalSamples)
        for index in 0..<dropAt {
            samples[index] = 0.1
        }

        let trimSamples = KokoroEngine.adaptiveTrailingEOSTrimSamples(
            in: samples, trailSamples: trailSamples, speed: 1.0)

        // detectedPostroll = (dropAt - eosStart) + 120 margin = 600 + 120 = 720;
        // max(minPostroll=960, 720) wins → postroll = 960.
        #expect(trimSamples == trailSamples - 960)
    }

    @Test("EOS deep drop selects detected postroll over minimum")
    func eosDeepDropUsesDetectedPostroll() {
        let trailSamples = 12 * KokoroEngine.hopSize  // 7200 samples (300ms)
        let totalSamples = trailSamples + KokoroEngine.hopSize
        let eosStart = totalSamples - trailSamples
        // Drop boundary at +10 windows (1200 samples = 50ms) into EOS.
        let dropAt = eosStart + 10 * 120
        var samples = [Float](repeating: 0.01, count: totalSamples)
        for index in 0..<dropAt {
            samples[index] = 0.1
        }

        let trimSamples = KokoroEngine.adaptiveTrailingEOSTrimSamples(
            in: samples, trailSamples: trailSamples, speed: 1.0)

        // detectedPostroll = 1200 + 120 = 1320, between min(960) and max(2040) → 1320.
        #expect(trimSamples == trailSamples - 1320)
    }

    @Test("Pre-EOS drops do not trigger boundary detection")
    func preEOSDropIgnored() {
        let trailSamples = 4 * KokoroEngine.hopSize
        let totalSamples = trailSamples + 4 * KokoroEngine.hopSize  // extra speech tail
        let eosStart = totalSamples - trailSamples
        // Drop happens entirely BEFORE EOS — should fall back to minPostroll.
        let dropAt = eosStart - 240
        var samples = [Float](repeating: 0.01, count: totalSamples)
        for index in 0..<dropAt {
            samples[index] = 0.1
        }

        let trimSamples = KokoroEngine.adaptiveTrailingEOSTrimSamples(
            in: samples, trailSamples: trailSamples, speed: 1.0)

        #expect(trimSamples == trailSamples - 960)
    }
}
