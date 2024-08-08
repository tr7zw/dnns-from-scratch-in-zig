import java.util.Arrays;

public class NLL {
    private double[] loss;
    private double[] inputGrads;
    private int batchSize;
    private static final boolean GIVE_LOSS = true;
    private static final double INF = Double.POSITIVE_INFINITY;

    public NLL(int inputSize, int batchSize) {
        this.loss = new double[batchSize];
        this.inputGrads = new double[batchSize * inputSize];
        this.batchSize = batchSize;
    }

    public void nll(double[] inputs, byte[] targets, int inputSize) throws Exception {
        for (int b = 0; b < batchSize; b++) {
            double sum = 0;
            double maxInput = -INF;

            // Find the maximum value in the inputs for the current batch
            for (int i = 0; i < inputSize; i++) {
                maxInput = Math.max(maxInput, inputs[b * inputSize + i]);
            }

            for (int i = 0; i < inputSize; i++) {
                sum += Math.exp(inputs[b * inputSize + i] - maxInput);
                if (sum == INF) {
                    System.out.println("output with inf:\n" + inputs[b * inputSize + i]);
                    throw new Exception("division by inf");
                }
            }

            double s = sum;
            sum = (Math.signum(sum) + 0.000000001) * Math.max(0.0000001, Math.abs(sum));
            if (sum >= INF) throw new Exception("division by inf");
            if (Math.signum(sum) == 0) {
                System.out.println("sum:" + s);
                throw new Exception("sum is NaN");
            }
            if (sum == 0) {
                throw new Exception("division by zero");
            }

            if (GIVE_LOSS) {
                loss[b] = -1 * Math.log(Math.exp(inputs[b * inputSize + targets[b]]) / sum);
            }

            for (int i = 0; i < inputSize; i++) {
                inputGrads[b * inputSize + i] = Math.exp(inputs[b * inputSize + i] - maxInput) / sum;
                if (i == targets[b]) {
                    inputGrads[b * inputSize + i] -= 1;
                }
            }
        }
    }

    public double[] getLoss() {
        return loss;
    }

    public double[] getInputGrads() {
        return inputGrads;
    }
}
