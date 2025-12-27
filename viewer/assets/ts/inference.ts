import * as ort from 'onnxruntime-web';

export class ModelRunner {
    private modelUrl: string;
    private session: ort.InferenceSession | null;

    constructor(modelUrl: string) {
        this.modelUrl = modelUrl;
        this.session = null;
    }

    async init(): Promise<void> {
        // Configure WASM paths to be served from the root
        ort.env.wasm.wasmPaths = '/';
        
        this.session = await ort.InferenceSession.create(this.modelUrl, {
            executionProviders: ['wasm', 'webgl']
        });
    }

    async run(input: Float32Array): Promise<ort.Tensor> {
        if (!this.session) await this.init();
        if (!this.session) throw new Error("Failed to initialize session");
        
        // Assuming input is a Float32Array of the correct shape (e.g. 1x3x640x640)
        // Adjust shape based on your specific YOLO model export
        const tensor = new ort.Tensor('float32', input, [1, 3, 640, 640]);
        
        const feeds: Record<string, ort.Tensor> = { [this.session.inputNames[0]]: tensor };
        const results = await this.session.run(feeds);
        
        return results[this.session.outputNames[0]] as ort.Tensor;
    }
}