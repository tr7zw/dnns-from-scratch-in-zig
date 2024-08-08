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
            new LayerDescriptor(new LayerGrok(), ActivationType.RELU),
            new LayerDescriptor(new LayerGrok(), ActivationType.RELU),
            new LayerDescriptor(new LayerGrok(), ActivationType.RELU),
            new LayerDescriptor(new LayerGrok(), ActivationType.RELU),
            new LayerDescriptor(new LayerGrok(), ActivationType.RELU)
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
            case RELU -> new ReLU(batchSize, desc.layer.getSize());
            case PYRAMID -> new Pyramid(batchSize, desc.layer.getSize());
            case GAUSSIAN -> new Gaussian(batchSize, desc.layer.getSize());
            case NONE -> null;
        };
        return new LayerStorage(layer, activation);
    }

    static LayerStorage[] Neuralnet(LayerStorage[] validationStorage, LayerStorage[] storage, int inputSize, int outputSize, int batchSize, int epochs) throws IOException {
        NLL loss = new NLL(outputSize, batchSize);

        Mnist.Data mnist_data = Mnist.readMnist();

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
                        previousLayerOut = current.activation.getFwdOut();
                    }
                }

                loss.nll(previousLayerOut, targets, outputSize);

                double[] previousGradient = loss.getInputGrads();
                for (int ni = 0; ni < storage.length; ni++) {
                    int index = storage.length - ni - 1;
                    if (storage[index].activation != null) {
                        storage[index].activation.backwards(previousGradient);
                        previousGradient = storage[index].activation.getBkwOut();
                    }
                    storage[index].layer.backwards(previousGradient);
                    previousGradient = storage[index].layer.getInputGrads();
                }

                for (LayerStorage current : storage) {
                    current.layer.applyGradients();
                }
            }

            double correct = 0;
            double[] inputs = mnist_data.test_images;

            for (int cur = 0; cur < validationStorage.length; cur++) {
                validationStorage[cur].layer.setWeights(storage[cur].layer.getWeights());
                if (validationStorage[cur].layer instanceof LayerBias) {
                    ((LayerBias) validationStorage[cur].layer).setBiases(((LayerBias) storage[cur].layer).getBiases());
                } else if (validationStorage[cur].layer instanceof LayerGrok) {
                    ((LayerGrok) validationStorage[cur].layer).setBiases(((LayerGrok) storage[cur].layer).getBiases());
                }
            }

            double[] previousLayerOut = inputs;
            for (LayerStorage current : validationStorage) {
                current.layer.forward(previousLayerOut);
                previousLayerOut = current.layer.outputs;
                if (current.activation != null) {
                    current.activation.forward(previousLayerOut);
                    previousLayerOut = current.activation.getFwdOut();
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
