import java.util.Arrays;

public class Gaussian {
    private double[] lastInputs;
    private double[] fwdOut;
    private double[] bkwOut;
    private int batchSize;
    private int size;

    public static final double MU = 1.0;
    public static final double SIGMA = 1.0;
    private static final double SIGMA2 = SIGMA * SIGMA;
    private static final double CONTINUE_POINT = 3;
    private static final double EPSILON = gaussianDerivative(CONTINUE_POINT);

    public Gaussian(int batchSize, int size) {
        this.batchSize = batchSize;
        this.size = size;
        this.lastInputs = new double[size * batchSize];
        this.fwdOut = new double[size * batchSize];
        this.bkwOut = new double[size * batchSize];
    }

    public static double gaussian(double x) {
        double val = 1.0 / Math.sqrt(2 * Math.PI * SIGMA2);
        return val * Math.exp(-Math.pow(x, 2) / (2 * SIGMA2));
    }

    public static double gaussianDerivative(double x) {
        return -(x / SIGMA2) * gaussian(x);
    }

    public static double leakyGaussian(double x) {
        double gauss = gaussian(x);
        if (Math.abs(x) < CONTINUE_POINT) {
            return gauss;
        } else {
            return gaussian(CONTINUE_POINT) + EPSILON * (Math.abs(x) - CONTINUE_POINT);
        }
    }

    public static double leakyGaussianDerivative(double x) {
        double gd = gaussianDerivative(x);
        if (Math.abs(x) < CONTINUE_POINT) {
            return gd;
        } else {
            return EPSILON * Math.signum(x);
        }
    }

    public void forward(double[] inputs) {
        assert inputs.length == this.size * this.batchSize;

        for (int i = 0; i < inputs.length; i++) {
            this.fwdOut[i] = leakyGaussian(inputs[i] - MU);
        }
        this.lastInputs = Arrays.copyOf(inputs, inputs.length);
    }

    public void backwards(double[] grads) {
        assert grads.length == this.size * this.batchSize;

        for (int i = 0; i < this.lastInputs.length; i++) {
            this.bkwOut[i] = grads[i] * leakyGaussianDerivative(this.lastInputs[i] - MU);
        }
    }
}
