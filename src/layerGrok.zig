const std = @import("std");

weights: []f64,
biases: []f64,
last_inputs: []const f64,
outputs: []f64,
weight_grads: []f64,
bias_grads: []f64,
input_grads: []f64,
batchSize: usize,
inputSize: usize,
outputSize: usize,

rounds: f64,
const Self = @This();

pub fn setWeights(self: *Self, weights: []f64) void {
    self.weights = weights;
}

pub fn setBiases(self: *Self, biases: []f64) void {
    self.biases = biases;
}
pub fn readBiases(self: *Self, weights: anytype) !void {
    _ = try weights.read(std.mem.sliceAsBytes(self.biases));
}
pub fn readWeights(self: *Self, weights: anytype) !void {
    //std.debug.print("deez: {any};", .{self.weights});
    _ = try weights.read(std.mem.sliceAsBytes(self.weights));
    //std.debug.print("deez: {any};", .{self.weights});
}
pub fn init(
    alloc: std.mem.Allocator,
    batchSize: usize,
    inputSize: usize,
    outputSize: usize,
) !Self {
    std.debug.assert(inputSize != 0);
    std.debug.assert(outputSize != 0);
    std.debug.assert(batchSize != 0);
    var weights: []f64 = try alloc.alloc(f64, inputSize * outputSize);
    var biases: []f64 = try alloc.alloc(f64, outputSize);
    var prng = std.Random.DefaultPrng.init(123);

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
        .outputSize = outputSize,
        .inputSize = inputSize,
        .rounds = 0,
    };
}
pub fn deinitBackwards(self: *Self, alloc: std.mem.Allocator) void {

    //alloc.free(self.last_inputs);
    //alloc.free(self.outputs);
    alloc.free(self.weight_grads);
    alloc.free(self.bias_grads);
    alloc.free(self.input_grads);
}

pub fn forward(
    self: *Self,
    inputs: []const f64,
) void {
    if (inputs.len != self.inputSize * self.batchSize) {
        std.debug.print("size mismatch {any}, vs expected {any} * {any} = {any}", .{ inputs.len, self.inputSize, self.batchSize, self.inputSize * self.batchSize });
    }
    std.debug.assert(inputs.len == self.inputSize * self.batchSize);

    var b: usize = 0;
    while (b < self.batchSize) : (b += 1) {
        var o: usize = 0;
        while (o < self.outputSize) : (o += 1) {
            var sum: f64 = 0;
            var i: usize = 0;
            while (i < self.inputSize) : (i += 1) {
                sum += inputs[b * self.inputSize + i] * self.weights[self.outputSize * i + o];
            }
            self.outputs[b * self.outputSize + o] = sum + self.biases[o];
        }
    }
    self.last_inputs = inputs;
}

pub fn backwards(
    self: *Self,
    grads: []f64,
) void {
    std.debug.assert(self.last_inputs.len == self.inputSize * self.batchSize);

    @memset(self.input_grads, 0);
    //const lgrad: []f64 = undefined;
    //@memcpy(lgrad, self.weight_grads);

    @memset(self.weight_grads, 0);
    @memset(self.bias_grads, 0);
    self.rounds += 1;
    var b: usize = 0;
    while (b < self.batchSize) : (b += 1) {
        var o: usize = 0;
        while (o < self.outputSize) : (o += 1) {
            self.bias_grads[o] -= std.math.pow(f64, (self.bias_grads[o] - grads[b * self.outputSize + o] / @as(f64, @floatFromInt(self.batchSize))), 1);
            var i: usize = 0;
            while (i < self.inputSize) : (i += 1) {
                self.weight_grads[i * self.outputSize + o] +=
                    std.math.pow(f64, self.weight_grads[i * self.outputSize + o] - (grads[b * self.outputSize + o] * self.last_inputs[b * self.inputSize + i]) / @as(f64, @floatFromInt(self.batchSize)), 1);
                self.input_grads[b * self.inputSize + i] +=
                    grads[b * self.outputSize + o] * self.weights[i * self.outputSize + o];
            }
        }
    }
}

//todo try minmax range with sin/cosine

pub fn applyGradients(self: *Self) void {
    var i: usize = 0;
    while (i < self.inputSize * self.outputSize) : (i += 1) {
        self.weights[i] -= 0.01 * self.weight_grads[i];
    }

    var o: usize = 0;
    while (o < self.outputSize) : (o += 1) {
        self.biases[o] -= 0.01 * self.bias_grads[o];
    }
}
