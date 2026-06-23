// Minimal ONNX Runtime surface shared by the ASR engine across runtimes
// (onnxruntime-web in the browser worker, onnxruntime-node in the headless tests).
// Kept tiny and runtime-agnostic so the streaming engine never imports a concrete ORT.

export type TensorData =
  | Float32Array
  | Float64Array
  | Int32Array
  | BigInt64Array
  | Uint16Array
  | Int8Array
  | Uint8Array

export interface OrtTensor {
  readonly type: string
  readonly data: TensorData
  readonly dims: readonly number[]
}

export type TensorCtor = (type: string, data: TensorData, dims: number[]) => OrtTensor

export interface OrtSession {
  run(feeds: Record<string, OrtTensor>): Promise<Record<string, OrtTensor>>
}
