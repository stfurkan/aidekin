// Minimal ambient types for fft.js@4 (ships no types).
declare module 'fft.js' {
  export default class FFT {
    constructor(size: number)
    readonly size: number
    createComplexArray(): number[]
    /** Real → complex spectrum; fills the first half (use completeSpectrum for the rest). */
    realTransform(output: ArrayLike<number>, input: ArrayLike<number>): void
    /** Mirror the conjugate-symmetric upper half into output. */
    completeSpectrum(output: ArrayLike<number>): void
    transform(output: ArrayLike<number>, input: ArrayLike<number>): void
    inverseTransform(output: ArrayLike<number>, input: ArrayLike<number>): void
  }
}
