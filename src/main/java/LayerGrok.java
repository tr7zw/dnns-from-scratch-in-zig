import java.util.Random;

public class LayerGrok {
    private boolean[] dropOut;
    private double[] weights;
    private double[] biases;
    private double[] lastInputs;
    private double[] outputs;
    private double[] weightGrads;
    private double[] averageWeights;
    private double[] biasGrads;
    private double[] inputGrads;
    private int batchSize;
    private int inputSize;
    private int outputSize;

    private double normMulti = 1;
    private double normBias = 0;

    private double nodrop = 1.0;
    private double rounds = batchdropskip + 1;

    private static final double batchdropskip = 0.5;
    private static final double dropOutRate = 0.00;

    private static final double scale = 1.0 / (1.0 - dropOutRate);
    private static final boolean usedrop = false;

    private Random prng = new Random(123);

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public void setBiases(double[] biases) {
        this.biases = biases;
    }

    public void readParams(java.io.DataInputStream params) throws java.io.IOException {
        params.readFully(this.weights);
        params.readFully(this.biases);
        params.readFully(this.averageWeights);
        this.normMulti = params.readDouble();
        this.normBias = params.readDouble();
    }

    public void writeParams(java.io.DataOutputStream params) throws java.io.IOException {
        params.write(this.weights);
        params.write(this.biases);
        params.write(this.averageWeights);
        params.writeDouble(this.normMulti);
        params.writeDouble(this.normBias);
    }

    public void reinit(double percent) {
        Stat wa = stats(this.weights);
        for (int w = 0; w < this.inputSize * this.outputSize; w++) {
            double sqrI = Math.sqrt(2.0 / this.inputSize);
            double dev = wa.range * sqrI * percent;
            this.weights[w] = (this.averageWeights[w]) + prng.nextGaussian() * dev;
        }

        for (int b = 0; b < this.outputSize; b++) {
            this.biases[b] = prng.nextGaussian() * 0.2;
        }
    }

    public static LayerGrok init(int batchSize, int inputSize, int outputSize) {
        LayerGrok layer = new LayerGrok();
        layer.batchSize = batchSize;
        layer.inputSize = inputSize;
        layer.outputSize = outputSize;

        layer.weights = new double[inputSize * outputSize];
        layer.biases = new double[outputSize];
        layer.outputs = new double[outputSize * batchSize];
        layer.weightGrads = new double[inputSize * outputSize];
        layer.averageWeights = new double[inputSize * outputSize];
        layer.biasGrads = new double[outputSize];
        layer.inputGrads = new double[inputSize * batchSize];
        layer.dropOut = new boolean[inputSize * outputSize];

        for (int w = 0; w < inputSize * outputSize; w++) {
            double dev = inputSize;
            layer.weights[w] = layer.prng.nextGaussian() * Math.sqrt(2.0 / dev);
        }

        for (int b = 0; b < outputSize; b++) {
            layer.biases[b] = layer.prng.nextGaussian() * 0.2;
        }

        System.arraycopy(layer.weights, 0, layer.averageWeights, 0, inputSize * outputSize);

        return layer;
    }

    public void deinitBackwards() {
        this.nodrop = scale;
    }

    public void forward(double[] inputs) {
        if (inputs.length != this.inputSize * this.batchSize) {
            throw new IllegalArgumentException("size mismatch " + inputs.length + ", vs expected " + this.inputSize + " * " + this.batchSize + " = " + (this.inputSize * this.batchSize));
        }

        for (int b = 0; b < this.batchSize; b++) {
            for (int o = 0; o < this.outputSize; o++) {
                double sum = 0;
                for (int i = 0; i < this.inputSize; i++) {
                    double d = 1.0;
                    sum += d * inputs[b * this.inputSize + i] * this.weights[i + this.inputSize * o];
                }
                this.outputs[b * this.outputSize + o] = sum + this.biases[o];
            }
        }
        this.lastInputs = inputs;
    }

    public void backwards(double[] grads) {
        if (this.lastInputs.length != this.inputSize * this.batchSize) {
            throw new IllegalArgumentException("size mismatch " + this.lastInputs.length + ", vs expected " + this.inputSize + " * " + this.batchSize + " = " + (this.inputSize * this.batchSize));
        }

        java.util.Arrays.fill(this.inputGrads, 0);
        java.util.Arrays.fill(this.weightGrads, 0);
        java.util.Arrays.fill(this.biasGrads, 0);

        for (int b = 0; b < this.batchSize; b++) {
            for (int o = 0; o < this.outputSize; o++) {
                this.biasGrads[o] += grads[b * this.outputSize + o] / this.batchSize;

                for (int i = 0; i < this.inputSize; i++) {
                    double w = grads[b * this.outputSize + o] * this.lastInputs[b * this.inputSize + i];
                    this.weightGrads[i + this.inputSize * o] += w / this.batchSize;
                    this.inputGrads[b * this.inputSize + i] += grads[b * this.outputSize + o] * this.weights[i + this.inputSize * o];
                }
            }
        }
    }

    private static class Stat {
        double range;
        double avg;
        double avgabs;

        Stat(double range, double avg, double avgabs) {
            this.range = range;
            this.avg = avg;
            this.avgabs = avgabs;
        }
    }

    private static Stat stats(double[] arr) {
        double min = Double.MAX_VALUE;
        double max = -min;
        double sum = 0.000000001;
        double absum = 0.000000001;
        for (double elem : arr) {
            if (min > elem) min = elem;
            if (max < elem) max = elem;
            sum += elem;
            absum += Math.abs(elem);
        }
        return new Stat(Math.max(0.000000001, Math.abs(max - min)), sum / arr.length, absum / arr.length);
    }

    private static double[] normalize(double[] arr, double multi, double bias, double alpha) {
        Stat gv = stats(arr);
        for (int i = 0; i < arr.length; i++) {
            arr[i] -= alpha * (arr[i] - (((arr[i] - gv.avg) / gv.range * multi) + bias));
        }
        return arr;
    }

    private static final double roundsPerEp = 60000 / 100;
    private static final double smoothing = 0.00001;
    private static final double lr = 0.00001;
    private static final double normlr = lr / 10.0;
    private static final double lambda = 0.0075;
    private static final double elasticAlpha = 0.0;

    public void applyGradients() {
        Stat wavg = stats(this.weightGrads);
        Stat bavg = stats(this.biasGrads);
        this.normMulti -= wavg.avg * normlr;
        this.normBias -= bavg.avg * normlr;

        this.rounds += 1.0;

        for (int i = 0; i < this.inputSize * this.outputSize; i++) {
            double l2 = lambda * this.weights[i];
            double l1 = lambda * Math.signum(this.weights[i]);

            double EN = Math.lerp(l2, l1, elasticAlpha);
            double g = this.weightGrads[i] + EN;

            double awdiff = this.averageWeights[i] - this.weights[i];
            double gdiff = 1.0 / (0.5 + Math.abs(g - awdiff));
            this.weights[i] -= lr * g * gdiff;

            double aw = this.averageWeights[i];
            this.averageWeights[i] = aw + (smoothing * (this.weights[i] - aw));
        }

        for (int o = 0; o < this.outputSize; o++) {
            this.biases[o] -= lr * this.biasGrads[o];
        }

        if (this.rounds >= roundsPerEp * 100) {
            this.rounds = 0.0;
            this.reinit(0.001);
        }

        this.biases = normalize(this.biases, this.normMulti, this.normBias, 0.01);
    }
}
