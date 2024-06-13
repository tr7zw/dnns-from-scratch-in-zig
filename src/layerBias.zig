const std = @import("std");

pub fn Layer(
    comptime inputSize: usize,
    comptime outputSize: usize,
) type {
    std.debug.assert(inputSize != 0);
    std.debug.assert(outputSize != 0);
    //std.debug.assert(batchSize != 0);

    return struct {
        weights: []f64,
        biases: []f64,
        last_inputs: []const f64,
        outputs: []f64,
        weight_grads: []f64,
        bias_grads: []f64,
        input_grads: []f64,
        batchSize: usize,

        const Self = @This();

        pub fn setWeights(self: *Self, weights: []f64) void {
            self.weights = weights;
        }

        pub fn setBiases(self: *Self, biases: []f64) void {
            self.biases = biases;
        }

        pub fn init(
            alloc: std.mem.Allocator,
            batchSize: usize,
        ) !Self {
            var weights: []f64 = try alloc.alloc(f64, inputSize * outputSize);
            var biases: []f64 = try alloc.alloc(f64, outputSize);
            var prng = std.rand.DefaultPrng.init(123);

            var w: usize = 0;
            while (w < inputSize * outputSize) : (w += 1) {
                weights[w] = prng.random().floatNorm(f64) * 0.2;
            }

            var b: usize = 0;
            while (b < outputSize) : (b += 1) {
                biases[b] = prng.random().floatNorm(f64) * 0.2;
            }

            return Self{
                .weights = weights,
                .biases = biases,
                .last_inputs = undefined,
                .outputs = try alloc.alloc(f64, outputSize * batchSize),
                .weight_grads = try alloc.alloc(f64, inputSize * outputSize),
                .bias_grads = try alloc.alloc(f64, outputSize),
                .input_grads = try alloc.alloc(f64, inputSize * batchSize),
                .batchSize = batchSize,
            };
        }

        pub fn forward(
            self: *Self,
            inputs: []const f64,
        ) void {
            std.debug.assert(inputs.len == inputSize * self.batchSize);

            var b: usize = 0;
            while (b < self.batchSize) : (b += 1) {
                var o: usize = 0;
                while (o < outputSize) : (o += 1) {
                    var sum: f64 = 0;
                    var i: usize = 0;
                    while (i < inputSize) : (i += 1) {
                        sum += inputs[b * inputSize + i] * self.weights[outputSize * i + o];
                    }
                    self.outputs[b * outputSize + o] = sum + self.biases[o];
                }
            }
            self.last_inputs = inputs;
        }

        pub fn backwards(
            self: *Self,
            grads: []f64,
        ) void {
            std.debug.assert(self.last_inputs.len == inputSize * self.batchSize);

            @memset(self.input_grads, 0);
            @memset(self.weight_grads, 0);
            @memset(self.bias_grads, 0);

            var b: usize = 0;
            while (b < self.batchSize) : (b += 1) {
                var o: usize = 0;
                while (o < outputSize) : (o += 1) {
                    self.bias_grads[o] += grads[b * outputSize + o] / @as(f64, @floatFromInt(self.batchSize));
                    var i: usize = 0;
                    while (i < inputSize) : (i += 1) {
                        self.weight_grads[i * outputSize + o] +=
                            (grads[b * outputSize + o] * self.last_inputs[b * inputSize + i]) / @as(f64, @floatFromInt(self.batchSize));
                        self.input_grads[b * inputSize + i] +=
                            grads[b * outputSize + o] * self.weights[i * outputSize + o];
                    }
                }
            }
        }

        pub fn applyGradients(self: *Self, weight_grads: []f64, bias_grads: []f64) void {
            var i: usize = 0;
            while (i < inputSize * outputSize) : (i += 1) {
                self.weights[i] -= 0.01 * weight_grads[i];
            }

            var o: usize = 0;
            while (o < outputSize) : (o += 1) {
                self.biases[o] -= 0.01 * bias_grads[o];
            }
        }
    };
}
