import java.util.Random;

public class LayerBias {
    private double[] weights;
    private double[] biases;
    private double[] lastInputs;
    private double[] outputs;
    private double[] weightGrads;
    private double[] biasGrads;
    private double[] inputGrads;
    private int batchSize;
    private int inputSize;
    private int outputSize;

    public LayerBias(int batchSize, int inputSize, int outputSize) {
        this.batchSize = batchSize;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weights = new double[inputSize * outputSize];
        this.biases = new double[outputSize];
        this.outputs = new double[outputSize * batchSize];
        this.weightGrads = new double[inputSize * outputSize];
        this.biasGrads = new double[outputSize];
        this.inputGrads = new double[inputSize * batchSize];
        Random prng = new Random(123);

        for (int i = 0; i < inputSize * outputSize; i++) {
            weights[i] = prng.nextGaussian() * 0.2;
        }

        for (int i = 0; i < outputSize; i++) {
            biases[i] = prng.nextGaussian() * 0.2;
        }
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public void setBiases(double[] biases) {
        this.biases = biases;
    }

    public void readParams(double[] params) {
        System.arraycopy(params, 0, this.weights, 0, this.weights.length);
        System.arraycopy(params, this.weights.length, this.biases, 0, this.biases.length);
    }

    public void writeParams(double[] params) {
        System.arraycopy(this.weights, 0, params, 0, this.weights.length);
        System.arraycopy(this.biases, 0, params, this.weights.length, this.biases.length);
    }

    public void deinitBackwards() {
        // No need to free memory in Java
    }

    public void forward(double[] inputs) {
        if (inputs.length != this.inputSize * this.batchSize) {
            System.out.printf("size mismatch %d, vs expected %d * %d = %d", inputs.length, this.inputSize, this.batchSize, this.inputSize * this.batchSize);
        }
        assert inputs.length == this.inputSize * this.batchSize;

        for (int b = 0; b < this.batchSize; b++) {
            for (int o = 0; o < this.outputSize; o++) {
                double sum = 0;
                for (int i = 0; i < this.inputSize; i++) {
                    sum += inputs[b * this.inputSize + i] * this.weights[this.outputSize * i + o];
                }
                this.outputs[b * this.outputSize + o] = sum + this.biases[o];
            }
        }
        this.lastInputs = inputs;
    }

    public void backwards(double[] grads) {
        assert this.lastInputs.length == this.inputSize * this.batchSize;

        for (int i = 0; i < this.inputGrads.length; i++) {
            this.inputGrads[i] = 0;
        }
        for (int i = 0; i < this.weightGrads.length; i++) {
            this.weightGrads[i] = 0;
        }
        for (int i = 0; i < this.biasGrads.length; i++) {
            this.biasGrads[i] = 0;
        }

        for (int b = 0; b < this.batchSize; b++) {
            for (int o = 0; o < this.outputSize; o++) {
                this.biasGrads[o] += grads[b * this.outputSize + o] / (double) this.batchSize;
                for (int i = 0; i < this.inputSize; i++) {
                    this.weightGrads[i * this.outputSize + o] += (grads[b * this.outputSize + o] * this.lastInputs[b * this.inputSize + i]) / (double) this.batchSize;
                    this.inputGrads[b * this.inputSize + i] += grads[b * this.outputSize + o] * this.weights[i * this.outputSize + o];
                }
            }
        }
    }

    public void applyGradients() {
        for (int i = 0; i < this.inputSize * this.outputSize; i++) {
            this.weights[i] -= 0.01 * this.weightGrads[i];
        }

        for (int o = 0; o < this.outputSize; o++) {
            this.biases[o] -= 0.01 * this.biasGrads[o];
        }
    }
}
