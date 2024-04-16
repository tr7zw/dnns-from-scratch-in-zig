const std = @import("std");

pub fn NLL(
    comptime inputSize: usize,
    comptime batchSize: usize,
) type {
    const NLLOuput = struct {
        loss: []f64,
        input_grads: []f64,
        const Self = @This();
    };

    return struct {
        var sum_e: [batchSize]f64 = undefined;
        var loss: [batchSize]f64 = undefined;
        var input_grads: [batchSize * inputSize]f64 = undefined;

        pub fn nll(inputs: []f64, targets: []u8) NLLOuput {
            var b: usize = 0;
            while (b < batchSize) : (b += 1) {
                var sum: f64 = 0;
                var i: usize = 0;
                while (i < inputSize) : (i += 1) {
                    sum += std.math.exp(inputs[b * inputSize + i]);
                }
                sum_e[b] = sum;
            }

            b = 0;
            while (b < batchSize) : (b += 1) {
                loss[b] = -1 * @log(std.math.exp(inputs[b * inputSize + targets[b]]) / sum_e[b]);
            }

            b = 0;
            while (b < batchSize) : (b += 1) {
                var i: usize = 0;
                while (i < inputSize) : (i += 1) {
                    input_grads[b * inputSize + i] = std.math.exp(inputs[b * inputSize + i]) / sum_e[b];
                    if (i == targets[b]) {
                        input_grads[b * inputSize + i] -= 1;
                    }
                }
            }

            return NLLOuput{ .loss = &loss, .input_grads = &input_grads };
        }
    };
}
