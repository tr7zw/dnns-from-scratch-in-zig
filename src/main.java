import java.io.*;
import java.nio.file.*;
import java.util.*;

public class Main {
    static final boolean timer = false;
    static final boolean readfile = true;
    static final boolean writeFile = true;
    static final String typesignature = "G25RRRR_G10R.f64";
    static final boolean graphfuncs = false;
    static final boolean reinit = true;
    static final int epoch = 1000;

    public static void main(String[] args) throws IOException {
        final int batchSize = 100;
        final int inputSize = 784;
        final int outputSize = 10;
        final int testImageCount = 10000;

        if (graphfuncs) {
            double[] inputs = new double[200];
            Gaussian pyr = new Gaussian(1, 200);
            for (int i = 0; i < inputs.length; i++) {
                inputs[i] = (-100 + (double) i) / 20;
            }

            pyr.forward(inputs);
            pyr.backwards(inputs);
            for (int i = 0; i < inputs.length; i++) {
                System.out.printf("%.4f,%.4f,%.4f%n", inputs[i], Gaussian.leaky_gaussian(inputs[i]), Gaussian.leaky_gaussian_derivative(inputs[i]));
            }
        }

        Path filePath = Paths.get("data/Params_" + typesignature);
        File file = filePath.toFile();

        LayerDescriptor[] layers = {
            new LayerDescriptor(new LayerG(25), ActivationType.RELU),
            new LayerDescriptor(new LayerG(25), ActivationType.RELU),
            new LayerDescriptor(new LayerG(25), ActivationType.RELU),
            new LayerDescriptor(new LayerG(25), ActivationType.RELU),
            new LayerDescriptor(new LayerG(10), ActivationType.RELU)
        };

        int previousLayerSize = inputSize;
        LayerStorage[] storage = new LayerStorage[layers.length];
        LayerStorage[] validationStorage = new LayerStorage[layers.length];

        try (BufferedReader reader = Files.newBufferedReader(filePath)) {
            for (int i = 0; i < layers.length; i++) {
                int size = layers[i].layer.getSize();
                storage[i] = layerFromDescriptor(layers[i], batchSize, previousLayerSize);
                validationStorage[i] = layerFromDescriptor(layers[i], testImageCount, previousLayerSize);
                validationStorage[i].layer.deinitBackwards();

                if (readfile) {
                    storage[i].layer.readParams(reader);
                    if (reinit) {
                        storage[i].layer.reinit(0.000);
                    }
                }
                previousLayerSize = size;
            }
        }

        Neuralnet(validationStorage, storage, inputSize, outputSize, batchSize, epoch);

        if (writeFile) {
            try (BufferedWriter writer = Files.newBufferedWriter(filePath)) {
                for (LayerStorage layerStorage : storage) {
                    layerStorage.layer.writeParams(writer);
                }
            }
        }
    }

    static LayerStorage layerFromDescriptor(LayerDescriptor desc, int batchSize, int inputSize) {
        Layer layer = desc.layer.init(batchSize, inputSize, desc.layer.getSize());
        Activation activation = switch (desc.activation) {
            case RELU -> new Relu(batchSize, desc.layer.getSize());
            case PYRAMID -> new Pyramid(batchSize, desc.layer.getSize());
            case GAUSSIAN -> new Gaussian(batchSize, desc.layer.getSize());
            case NONE -> null;
        };
        return new LayerStorage(layer, activation);
    }

    static LayerStorage[] Neuralnet(LayerStorage[] validationStorage, LayerStorage[] storage, int inputSize, int outputSize, int batchSize, int epochs) throws IOException {
        NLL loss = new NLL(outputSize, batchSize);

        Data mnist_data = Data.readMnist();

        long t = System.currentTimeMillis();
        System.out.println("Training...");

        for (int e = 0; e < epochs; e++) {
            for (int i = 0; i < 60000 / batchSize; i++) {
                double[] inputs = Arrays.copyOfRange(mnist_data.train_images, i * inputSize * batchSize, (i + 1) * inputSize * batchSize);
                byte[] targets = Arrays.copyOfRange(mnist_data.train_labels, i * batchSize, (i + 1) * batchSize);

                double[] previousLayerOut = inputs;
                for (LayerStorage current : storage) {
                    current.layer.forward(previousLayerOut);
                    previousLayerOut = current.layer.outputs;
                    if (current.activation != null) {
                        current.activation.forward(previousLayerOut);
                        previousLayerOut = current.activation.fwd_out;
                    }
                }

                loss.nll(previousLayerOut, targets);

                double[] previousGradient = loss.input_grads;
                for (int ni = 0; ni < storage.length; ni++) {
                    int index = storage.length - ni - 1;
                    if (storage[index].activation != null) {
                        storage[index].activation.backwards(previousGradient);
                        previousGradient = storage[index].activation.bkw_out;
                    }
                    storage[index].layer.backwards(previousGradient);
                    previousGradient = storage[index].layer.input_grads;
                }

                for (LayerStorage current : storage) {
                    current.layer.applyGradients();
                }
            }

            double correct = 0;
            double[] inputs = mnist_data.test_images;

            for (int cur = 0; cur < validationStorage.length; cur++) {
                validationStorage[cur].layer.setWeights(storage[cur].layer.getWeights());
                if (validationStorage[cur].layer instanceof LayerB) {
                    ((LayerB) validationStorage[cur].layer).setBiases(((LayerB) storage[cur].layer).getBiases());
                } else if (validationStorage[cur].layer instanceof LayerG) {
                    ((LayerG) validationStorage[cur].layer).setBiases(((LayerG) storage[cur].layer).getBiases());
                }
            }

            double[] previousLayerOut = inputs;
            for (LayerStorage current : validationStorage) {
                current.layer.forward(previousLayerOut);
                previousLayerOut = current.layer.outputs;
                if (current.activation != null) {
                    current.activation.forward(previousLayerOut);
                    previousLayerOut = current.activation.fwd_out;
                }
            }

            for (int b = 0; b < 10000; b++) {
                double max_guess = Double.NEGATIVE_INFINITY;
                int guess_index = 0;
                for (int oi = 0; oi < outputSize; oi++) {
                    if (previousLayerOut[b * outputSize + oi] > max_guess) {
                        max_guess = previousLayerOut[b * outputSize + oi];
                        guess_index = oi;
                    }
                }
                if (guess_index == mnist_data.test_labels[b]) {
                    correct++;
                }
            }
            correct /= 10000;
            if (timer) {
                System.out.printf("time total: %dms%n", System.currentTimeMillis() - t);
            }
            System.out.println(correct);
        }
        System.out.printf("time total: %dms%n", System.currentTimeMillis() - t);
        return storage;
    }

    static double averageArray(double[] arr) {
        double sum = 0;
        for (double elem : arr) {
            sum += elem;
        }
        return sum / arr.length;
    }
}

class LayerDescriptor {
    Layer layer;
    ActivationType activation;

    LayerDescriptor(Layer layer, ActivationType activation) {
        this.layer = layer;
        this.activation = activation;
    }
}

enum ActivationType {
    NONE, RELU, PYRAMID, GAUSSIAN
}

class LayerStorage {
    Layer layer;
    Activation activation;

    LayerStorage(Layer layer, Activation activation) {
        this.layer = layer;
        this.activation = activation;
    }
}

interface Layer {
    void forward(double[] inputs);
    void backwards(double[] grads);
    void applyGradients();
    void readParams(BufferedReader reader) throws IOException;
    void writeParams(BufferedWriter writer) throws IOException;
    void deinitBackwards();
    void setWeights(double[] weights);
    double[] getWeights();
    int getSize();
    void reinit(double percent);
    double[] outputs = null;
    double[] input_grads = null;
}

class LayerG implements Layer {
    double[] weights;
    double[] biases;
    double[] last_inputs;
    double[] outputs;
    double[] weight_grads;
    double[] averageWeights;
    double[] bias_grads;
    double[] input_grads;
    int batchSize;
    int inputSize;
    int outputSize;
    double normMulti = 1;
    double normBias = 0;
    double nodrop = 1.0;
    double rounds = batchdropskip + 1;
    static final double batchdropskip = 0.5;
    static final double dropOutRate = 0.00;
    static final double scale = 1.0 / (1.0 - dropOutRate);
    static final boolean usedrop = false;
    Random prng = new Random(123);

    LayerG(int batchSize, int inputSize, int outputSize) {
        this.batchSize = batchSize;
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.weights = new double[inputSize * outputSize];
        this.biases = new double[outputSize];
        this.outputs = new double[outputSize * batchSize];
        this.weight_grads = new double[inputSize * outputSize];
        this.averageWeights = new double[inputSize * outputSize];
        this.bias_grads = new double[outputSize];
        this.input_grads = new double[inputSize * batchSize];
        for (int i = 0; i < inputSize * outputSize; i++) {
            this.weights[i] = prng.nextGaussian() * Math.sqrt(2.0 / inputSize);
        }
        for (int i = 0; i < outputSize; i++) {
            this.biases[i] = prng.nextGaussian() * 0.2;
        }
        System.arraycopy(this.weights, 0, this.averageWeights, 0, inputSize * outputSize);
    }

    @Override
    public void forward(double[] inputs) {
        if (inputs.length != this.inputSize * this.batchSize) {
            throw new IllegalArgumentException("size mismatch");
        }
        for (int b = 0; b < this.batchSize; b++) {
            for (int o = 0; o < this.outputSize; o++) {
                double sum = 0;
                for (int i = 0; i < this.inputSize; i++) {
                    sum += inputs[b * this.inputSize + i] * this.weights[i + this.inputSize * o];
                }
                this.outputs[b * this.outputSize + o] = sum + this.biases[o];
            }
        }
        this.last_inputs = inputs;
    }

    @Override
    public void backwards(double[] grads) {
        if (this.last_inputs.length != this.inputSize * this.batchSize) {
            throw new IllegalArgumentException("size mismatch");
        }
        Arrays.fill(this.input_grads, 0);
        Arrays.fill(this.weight_grads, 0);
        Arrays.fill(this.bias_grads, 0);
        for (int b = 0; b < this.batchSize; b++) {
            for (int o = 0; o < this.outputSize; o++) {
                this.bias_grads[o] += grads[b * this.outputSize + o] / this.batchSize;
                for (int i = 0; i < this.inputSize; i++) {
                    this.weight_grads[i + this.inputSize * o] += (grads[b * this.outputSize + o] * this.last_inputs[b * this.inputSize + i]) / this.batchSize;
                    this.input_grads[b * this.inputSize + i] += grads[b * this.outputSize + o] * this.weights[i + this.inputSize * o];
                }
            }
        }
    }

    @Override
    public void applyGradients() {
        for (int i = 0; i < this.inputSize * this.outputSize; i++) {
            this.weights[i] -= 0.01 * this.weight_grads[i];
        }
        for (int o = 0; o < this.outputSize; o++) {
            this.biases[o] -= 0.01 * this.bias_grads[o];
        }
    }

    @Override
    public void readParams(BufferedReader reader) throws IOException {
        for (int i = 0; i < this.weights.length; i++) {
            this.weights[i] = Double.parseDouble(reader.readLine());
        }
        for (int i = 0; i < this.biases.length; i++) {
            this.biases[i] = Double.parseDouble(reader.readLine());
        }
        for (int i = 0; i < this.averageWeights.length; i++) {
            this.averageWeights[i] = Double.parseDouble(reader.readLine());
        }
        this.normMulti = Double.parseDouble(reader.readLine());
        this.normBias = Double.parseDouble(reader.readLine());
    }

    @Override
    public void writeParams(BufferedWriter writer) throws IOException {
        for (double weight : this.weights) {
            writer.write(Double.toString(weight));
            writer.newLine();
        }
        for (double bias : this.biases) {
            writer.write(Double.toString(bias));
            writer.newLine();
        }
        for (double averageWeight : this.averageWeights) {
            writer.write(Double.toString(averageWeight));
            writer.newLine();
        }
        writer.write(Double.toString(this.normMulti));
        writer.newLine();
        writer.write(Double.toString(this.normBias));
        writer.newLine();
    }

    @Override
    public void deinitBackwards() {
        this.nodrop = scale;
    }

    @Override
    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public void setBiases(double[] biases) {
        this.biases = biases;
    }

    @Override
    public double[] getWeights() {
        return this.weights;
    }

    @Override
    public int getSize() {
        return this.outputSize;
    }

    @Override
    public void reinit(double percent) {
        Stat wa = stats(this.weights);
        for (int i = 0; i < this.inputSize * this.outputSize; i++) {
            double sqrI = Math.sqrt(2.0 / this.inputSize);
            double dev = wa.range * sqrI * percent;
            this.weights[i] = (this.averageWeights[i]) + prng.nextGaussian() * dev;
        }
        for (int i = 0; i < this.outputSize; i++) {
            this.biases[i] = prng.nextGaussian() * 0.2;
        }
    }

    static class Stat {
        double range;
        double avg;
        double avgabs;

        Stat(double range, double avg, double avgabs) {
            this.range = range;
            this.avg = avg;
            this.avgabs = avgabs;
        }
    }

    static Stat stats(double[] arr) {
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

    static double[] normalize(double[] arr, double multi, double bias, double alpha) {
        Stat gv = stats(arr);
        for (int i = 0; i < arr.length; i++) {
            arr[i] -= alpha * (arr[i] - (((arr[i] - gv.avg) / gv.range * multi) + bias));
        }
        return arr;
    }
}

class Relu implements Activation {
    double[] last_inputs;
    double[] fwd_out;
    double[] bkw_out;
    int batchSize;
    int size;

    Relu(int batchSize, int size) {
        this.batchSize = batchSize;
        this.size = size;
        this.fwd_out = new double[size * batchSize];
        this.bkw_out = new double[size * batchSize];
    }

    @Override
    public void forward(double[] inputs) {
        if (inputs.length != this.size * this.batchSize) {
            throw new IllegalArgumentException("size mismatch");
        }
        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i] < 0) {
                this.fwd_out[i] = 0.01 * inputs[i];
            } else {
                this.fwd_out[i] = inputs[i];
            }
        }
        this.last_inputs = inputs;
    }

    @Override
    public void backwards(double[] grads) {
        if (grads.length != this.size * this.batchSize) {
            throw new IllegalArgumentException("size mismatch");
        }
        for (int i = 0; i < this.last_inputs.length; i++) {
            if (this.last_inputs[i] < 0) {
                this.bkw_out[i] = 0.01 * grads[i];
            } else {
                this.bkw_out[i] = grads[i];
            }
        }
    }
}

class Pyramid implements Activation {
    double[] last_inputs;
    double[] fwd_out;
    double[] bkw_out;
    int batchSize;
    int size;

    static final double threshold = 1.0;
    static final double leak_slope = 0.01;

    Pyramid(int batchSize, int size) {
        this.batchSize = batchSize;
        this.size = size;
        this.fwd_out = new double[size * batchSize];
        this.bkw_out = new double[size * batchSize];
    }

    @Override
    public void forward(double[] inputs) {
        if (inputs.length != this.size * this.batchSize) {
            throw new IllegalArgumentException("size mismatch");
        }
        for (int i = 0; i < inputs.length; i++) {
            double x = inputs[i];
            if (x < 0) {
                this.fwd_out[i] = leak_slope * x;
            } else if (x < threshold) {
                this.fwd_out[i] = x;
            } else if (x < 2 * threshold) {
                this.fwd_out[i] = 2 * threshold - x;
            } else {
                this.fwd_out[i] = leak_slope * (2 * threshold - x);
            }
        }
        this.last_inputs = inputs;
    }

    @Override
    public void backwards(double[] grads) {
        if (grads.length != this.size * this.batchSize) {
            throw new IllegalArgumentException("size mismatch");
        }
        for (int i = 0; i < this.last_inputs.length; i++) {
            double x = this.last_inputs[i];
            if (x < 0) {
                this.bkw_out[i] = leak_slope * grads[i];
            } else if (x < threshold) {
                this.bkw_out[i] = grads[i];
            } else if (x < 2 * threshold) {
                this.bkw_out[i] = -grads[i];
            } else {
                this.bkw_out[i] = -leak_slope * grads[i];
            }
        }
    }
}

class Gaussian implements Activation {
    double[] last_inputs;
    double[] fwd_out;
    double[] bkw_out;
    int batchSize;
    int size;

    static final double mu = 1.0;
    static final double sigma = 1.0;
    static final double sigma2 = sigma * sigma;
    static final double continuepoint = 3;
    static final double epsilon = gaussian_derivative(continuepoint);

    Gaussian(int batchSize, int size) {
        this.batchSize = batchSize;
        this.size = size;
        this.fwd_out = new double[size * batchSize];
        this.bkw_out = new double[size * batchSize];
    }

    @Override
    public void forward(double[] inputs) {
        if (inputs.length != this.size * this.batchSize) {
            throw new IllegalArgumentException("size mismatch");
        }
        for (int i = 0; i < inputs.length; i++) {
            this.fwd_out[i] = leaky_gaussian(inputs[i] - mu);
        }
        this.last_inputs = inputs;
    }

    @Override
    public void backwards(double[] grads) {
        if (grads.length != this.size * this.batchSize) {
            throw new IllegalArgumentException("size mismatch");
        }
        for (int i = 0; i < this.last_inputs.length; i++) {
            this.bkw_out[i] = grads[i] * leaky_gaussian_derivative(this.last_inputs[i] - mu);
        }
    }

    static double gaussian(double x) {
        double val = 1.0 / Math.sqrt(2 * Math.PI * sigma2);
        return val * Math.exp(-Math.pow(x, 2) / (2 * sigma2));
    }

    static double gaussian_derivative(double x) {
        return -(x / sigma2) * gaussian(x);
    }

    static double leaky_gaussian(double x) {
        double gauss = gaussian(x);
        if (Math.abs(x) < continuepoint) {
            return gauss;
        } else {
            return gaussian(continuepoint) + epsilon * (Math.abs(x) - continuepoint);
        }
    }

    static double leaky_gaussian_derivative(double x) {
        double gd = gaussian_derivative(x);
        if (Math.abs(x) < continuepoint) {
            return gd;
        } else {
            return epsilon * Math.signum(x);
        }
    }
}

interface Activation {
    void forward(double[] inputs);
    void backwards(double[] grads);
    double[] fwd_out = null;
    double[] bkw_out = null;
}

class NLL {
    double[] loss;
    double[] input_grads;
    int batchSize;
    int inputSize;

    NLL(int inputSize, int batchSize) {
        this.inputSize = inputSize;
        this.batchSize = batchSize;
        this.loss = new double[batchSize];
        this.input_grads = new double[batchSize * inputSize];
    }

    void nll(double[] inputs, byte[] targets) {
        for (int b = 0; b < this.batchSize; b++) {
            double sum = 0;
            double maxInput = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < this.inputSize; i++) {
                maxInput = Math.max(maxInput, inputs[b * this.inputSize + i]);
            }
            for (int i = 0; i < this.inputSize; i++) {
                sum += Math.exp(inputs[b * this.inputSize + i] - maxInput);
                if (Double.isInfinite(sum)) {
                    throw new ArithmeticException("output with inf");
                }
            }
            sum = Math.signum(sum) * Math.max(0.0000001, Math.abs(sum));
            if (Double.isInfinite(sum)) {
                throw new ArithmeticException("division by inf");
            }
            if (Math.signum(sum) == 0) {
                throw new ArithmeticException("sum is NaN");
            }
            if (sum == 0) {
                throw new ArithmeticException("division by zero");
            }

            this.loss[b] = -Math.log(Math.exp(inputs[b * this.inputSize + targets[b]]) / sum);
            for (int i = 0; i < this.inputSize; i++) {
                this.input_grads[b * this.inputSize + i] = Math.exp(inputs[b * this.inputSize + i] - maxInput) / sum;
                if (i == targets[b]) {
                    this.input_grads[b * this.inputSize + i] -= 1;
                }
            }
        }
    }
}

class Data {
    double[] train_images;
    byte[] train_labels;
    double[] test_images;
    byte[] test_labels;

    static Data readMnist() throws IOException {
        Data data = new Data();
        data.train_images = readIdxFile("data/train-images.idx3-ubyte", 16);
        data.train_labels = readIdxFile("data/train-labels.idx1-ubyte", 8);
        data.test_images = readIdxFile("data/t10k-images.idx3-ubyte", 16);
        data.test_labels = readIdxFile("data/t10k-labels.idx1-ubyte", 8);
        return data;
    }

    static double[] readIdxFile(String path, int skipBytes) throws IOException {
        try (InputStream is = new FileInputStream(path)) {
            is.skip(skipBytes);
            byte[] buffer = is.readAllBytes();
            double[] data = new double[buffer.length];
            for (int i = 0; i < buffer.length; i++) {
                data[i] = (buffer[i] & 0xFF) / 255.0;
            }
            return data;
        }
    }
}
