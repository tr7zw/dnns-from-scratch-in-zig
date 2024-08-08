import java.util.Arrays;

public class ReLU {
    private double[] lastInputs;
    private double[] fwdOut;
    private double[] bkwOut;
    private int batchSize;
    private int size;

    public ReLU(int batchSize, int size) {
        this.batchSize = batchSize;
        this.size = size;
        this.lastInputs = new double[size * batchSize];
        this.fwdOut = new double[size * batchSize];
        this.bkwOut = new double[size * batchSize];
    }

    public void forward(double[] inputs) {
        assert inputs.length == this.size * this.batchSize;

        for (int i = 0; i < inputs.length; i++) {
            if (inputs[i] < 0) {
                this.fwdOut[i] = 0.01 * inputs[i];
            } else {
                this.fwdOut[i] = inputs[i];
            }
        }
        this.lastInputs = Arrays.copyOf(inputs, inputs.length);
    }

    public void backwards(double[] grads) {
        assert grads.length == this.size * this.batchSize;

        for (int i = 0; i < this.lastInputs.length; i++) {
            if (this.lastInputs[i] < 0) {
                this.bkwOut[i] = 0.01 * grads[i];
            } else {
                this.bkwOut[i] = grads[i];
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
