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
}
