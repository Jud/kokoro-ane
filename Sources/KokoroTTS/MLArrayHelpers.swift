import CoreML

/// Helpers for MLMultiArray operations used by the synthesis pipeline.
enum MLArrayHelpers {
    /// Extract float samples from an MLMultiArray, handling both float16 and float32 data types.
    static func extractFloats(from array: MLMultiArray, maxCount: Int? = nil) -> [Float] {
        let count = min(array.count, max(0, maxCount ?? array.count))
        guard count > 0 else { return [] }
        var samples = [Float](repeating: 0, count: count)
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count { samples[i] = Float(ptr[i]) }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            samples.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: ptr, count: count)
            }
        }
        return samples
    }

    /// Fill a pre-allocated style embedding MLMultiArray from a float vector.
    ///
    /// Copies up to `dim` elements from `styleVector` into `array`, zeroing
    /// any trailing slots to prevent stale data in reused buffers.
    static func fillStyleArray(from styleVector: [Float], into array: MLMultiArray, dim: Int? = nil) {
        let dim = dim ?? VoiceStore.styleDim
        let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
        let n = min(styleVector.count, dim)
        styleVector.withUnsafeBufferPointer { src in
            ptr.update(from: src.baseAddress!, count: n)
        }
        if n < dim {
            ptr.advanced(by: n).update(repeating: 0, count: dim - n)
        }
    }

    /// Fill pre-allocated input_ids and attention_mask MLMultiArrays from token IDs.
    static func fillTokenInputs(
        from tokenIds: [Int], into inputIds: MLMultiArray, mask: MLMultiArray, maxLength: Int
    ) {
        let n = min(tokenIds.count, maxLength)
        let inputPtr = inputIds.dataPointer.assumingMemoryBound(to: Int32.self)
        let maskPtr = mask.dataPointer.assumingMemoryBound(to: Int32.self)
        for i in 0..<maxLength {
            if i < n {
                inputPtr[i] = Int32(tokenIds[i])
                maskPtr[i] = 1
            } else {
                inputPtr[i] = 0
                maskPtr[i] = 0
            }
        }
    }
}
