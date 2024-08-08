import java.util.Arrays;

public class Pyramid {
    private double[] lastInputs;
    private double[] fwdOut;
    private double[] bkwOut;
    private int batchSize;
    private int size;

    public static final double THRESHOLD = 1.0;
    public static final double LEAK_SLOPE = 0.01;

    public Pyramid(int batchSize, int size) {
        this.batchSize = batchSize;
        this.size = size;
        this.lastInputs = new double[size * batchSize];
        this.fwdOut = new double[size * batchSize];
        this.bkwOut = new double[size * batchSize];
    }

    public void forward(double[] inputs) {
        if (inputs.length != this.size * this.batchSize) {
            throw new IllegalArgumentException("Input size mismatch");
        }

        for (int i = 0; i < inputs.length; i++) {
            double x = inputs[i];
            if (x < 0) {
                this.fwdOut[i] = LEAK_SLOPE * x;
            } else if (x < THRESHOLD) {
                this.fwdOut[i] = x;
            } else if (x < 2 * THRESHOLD) {
                this.fwdOut[i] = 2 * THRESHOLD - x;
            } else {
                this.fwdOut[i] = LEAK_SLOPE * (2 * THRESHOLD - x);
            }
        }
        this.lastInputs = Arrays.copyOf(inputs, inputs.length);
    }

    public void backwards(double[] grads) {
        if (grads.length != this.size * this.batchSize) {
            throw new IllegalArgumentException("Gradient size mismatch");
        }

        for (int i = 0; i < this.lastInputs.length; i++) {
            double x = this.lastInputs[i];
            if (x < 0) {
                this.bkwOut[i] = LEAK_SLOPE * grads[i];
            } else if (x < THRESHOLD) {
                this.bkwOut[i] = grads[i];
            } else if (x < 2 * THRESHOLD) {
                this.bkwOut[i] = -grads[i];
            } else {
                this.bkwOut[i] = -LEAK_SLOPE * grads[i];
            }
        }
    }

    public double[] getFwdOut() {
        return fwdOut;
    }

    public double[] getBkwOut() {
        return bkwOut;
    }
}
