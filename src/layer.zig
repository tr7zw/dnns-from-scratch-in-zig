const std = @import("std");

pub fn Layer(
    comptime inputSize: usize,
    comptime outputSize: usize,
    comptime batchSize: usize,
) type {
    if (batchSize == 0) {
        @compileError("0 is an invalid batch size, try 1 or larger");
    }
    return struct {
        weights: [inputSize * outputSize]f64,
        last_inputs: []const f64,
        outputs: [batchSize * outputSize]f64,
        weight_grads: [inputSize * outputSize]f64, // = [1]f64{0} ** (inputSize * outputSize);
        input_grads: [batchSize * inputSize]f64, //= [1]f64{0} ** (batchSize * inputSize);

        const Self = @This();
        //var outputs: [batchSize * outputSize]f64 = [1]f64{0} ** (batchSize * outputSize);

        pub fn forward(
            self: *Self,
            inputs: []const f64,
        ) void {
            std.debug.assert(inputs.len == inputSize * batchSize);
            var b: usize = 0;
            while (b < batchSize) : (b += 1) {
                var o: usize = 0;
                while (o < outputSize) : (o += 1) {
                    var sum: f64 = 0;
                    var i: usize = 0;
                    while (i < inputSize) : (i += 1) {
                        sum += inputs[b * inputSize + i] * self.weights[outputSize * i + o];
                    }
                    self.outputs[b * outputSize + o] = sum;
                }
            }
            self.last_inputs = inputs;
        }

        pub fn backwards(
            self: *Self,
            grads: []f64,
        ) void {
            std.debug.assert(self.last_inputs.len == inputSize * batchSize);

            self.input_grads = [1]f64{0} ** (inputSize * batchSize);
            self.weight_grads = [1]f64{0} ** (inputSize * outputSize);

            var b: usize = 0;
            while (b < batchSize) : (b += 1) {
                var i: usize = 0;
                while (i < inputSize) : (i += 1) {
                    var o: usize = 0;
                    while (o < outputSize) : (o += 1) {
                        self.weight_grads[i * outputSize + o] +=
                            (grads[b * outputSize + o] * self.last_inputs[b * inputSize + i]) / @as(f64, @floatFromInt(batchSize));
                        self.input_grads[b * inputSize + i] +=
                            grads[b * outputSize + o] * self.weights[i * outputSize + o];
                    }
                }
            }
        }

        pub fn applyGradients(self: *Self, grads: []f64) void {
            var i: usize = 0;
            while (i < inputSize * outputSize) : (i += 1) {
                self.weights[i] -= 0.01 * grads[i];
            }
        }

        pub fn init() Self {
            var weights: [inputSize * outputSize]f64 = [1]f64{0} ** (inputSize * outputSize);
            var prng = std.rand.DefaultPrng.init(123);
            var w: usize = 0;
            while (w < inputSize * outputSize) : (w += 1) {
                weights[w] = prng.random().floatNorm(f64) * 0.2;
            }
            return Self{
                .weights = weights,
                .last_inputs = undefined, //[1]f64{0} ** (batchSize * inputSize),
                .outputs = [1]f64{0} ** (batchSize * outputSize),
                .weight_grads = [1]f64{0} ** (inputSize * outputSize),
                .input_grads = [1]f64{0} ** (batchSize * inputSize),
            };
        }
    };
}
