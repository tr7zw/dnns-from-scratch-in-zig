import java.util.Random;

public class Layer {
    private double[] weights;
    private double[] lastInputs;
    public double[] outputs;
    private double[] weightGrads;
    public double[] input_grads;
    private int batchSize;
    private int inputSize;
    private int outputSize;

    public Layer(int batchSize, int inputSize, int outputSize) {
        this.batchSize = batchSize;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weights = new double[inputSize * outputSize];
        this.outputs = new double[outputSize * batchSize];
        this.weightGrads = new double[inputSize * outputSize];
        this.input_grads = new double[inputSize * batchSize];
        Random prng = new Random(123);
        for (int i = 0; i < inputSize * outputSize; i++) {
            weights[i] = prng.nextGaussian() * 0.2;
        }
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public double[] getWeights() {
        return this.weights;
    }

    public void readParams(java.io.DataInputStream params) throws java.io.IOException {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = params.readDouble();
        }
    }

    public void writeParams(java.io.DataOutputStream params) throws java.io.IOException {
        for (double weight : weights) {
            params.writeDouble(weight);
        }
    }

    public void deinitBackwards() {
        // No equivalent in Java, handled by garbage collector
    }

    public void forward(double[] inputs) {
        assert inputs.length == inputSize * batchSize;
        for (int b = 0; b < batchSize; b++) {
            for (int o = 0; o < outputSize; o++) {
                double sum = 0;
                for (int i = 0; i < inputSize; i++) {
                    sum += inputs[b * inputSize + i] * weights[outputSize * i + o];
                }
                outputs[b * outputSize + o] = sum;
            }
        }
        lastInputs = inputs;
    }

    public void backwards(double[] grads) {
        assert lastInputs.length == inputSize * batchSize;
        java.util.Arrays.fill(input_grads, 0);
        java.util.Arrays.fill(weightGrads, 0);
        for (int b = 0; b < batchSize; b++) {
            for (int i = 0; i < inputSize; i++) {
                for (int o = 0; o < outputSize; o++) {
                    weightGrads[i * outputSize + o] += (grads[b * outputSize + o] * lastInputs[b * inputSize + i]) / batchSize;
                    input_grads[b * inputSize + i] += grads[b * outputSize + o] * weights[i * outputSize + o];
                }
            }
        }
    }

    public void applyGradients() {
        for (int i = 0; i < inputSize * outputSize; i++) {
            weights[i] -= 0.01 * weightGrads[i];
        }
    }

    public int getSize() {
        return this.outputSize;
    }

    public void reinit(double percent) {
        Random prng = new Random(123);
        for (int i = 0; i < inputSize * outputSize; i++) {
            weights[i] = prng.nextGaussian() * 0.2 * percent;
        }
    }
}
