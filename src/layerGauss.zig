const std = @import("std");

const Self = @This();
mus: []f64,
sigmas: []f64,
last_inputs: []const f64,
outputs: []f64,
mu_grads: []f64,
sigma_grads: []f64,
input_grads: []f64,
batchSize: usize,
inputSize: usize,
outputSize: usize,

pub fn setParameters(self: *Self, mus: []f64, sigmas: []f64) void {
    self.mus = mus;
    self.sigmas = sigmas;
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

    const mus: []f64 = try alloc.alloc(f64, outputSize);
    const sigmas: []f64 = try alloc.alloc(f64, outputSize);
    var prng = std.rand.DefaultPrng.init(123);
    for (mus) |*mu| {
        mu = prng.random().floatNorm(f64);
    }
    for (sigmas) |*sigma| {
        sigma = 1.0; // initialize sigmas to 1
    }

    return Self{
        .mus = mus,
        .sigmas = sigmas,
        .last_inputs = undefined,
        .outputs = try alloc.alloc(f64, outputSize * batchSize),
        .mu_grads = try alloc.alloc(f64, outputSize),
        .sigma_grads = try alloc.alloc(f64, outputSize),
        .input_grads = try alloc.alloc(f64, inputSize * batchSize),
        .batchSize = batchSize,
        .inputSize = inputSize,
        .outputSize = outputSize,
    };
}

fn gaussian_bump(self: *Self, x: f64, index: usize) f64 {
    const exponent = -((x - self.mus[index]) * (x - self.mus[index])) / (2.0 * self.sigmas[index] * self.sigmas[index]);
    return @exp(exponent);
}

fn gaussian_bump_derivative(self: *Self, x: f64, index: usize) f64 {
    const exponent = -((x - self.mus[index]) * (x - self.mus[index])) / (2.0 * self.sigmas[index] * self.sigmas[index]);
    return -((x - self.mus[index]) / (self.sigmas[index] * self.sigmas[index])) * @exp(exponent);
}

pub fn forward(self: *Self, inputs: []const f64) void {
    std.debug.assert(inputs.len == self.inputSize * self.batchSize);

    var batch_start: usize = 0;
    while (batch_start < self.batchSize) : (batch_start += 1) {
        var sum: f64 = 0.0;
        var i: usize = 0;
        while (i < self.inputSize) : (i += 1) {
            sum += inputs[batch_start * self.inputSize + i];
        }
        var j: usize = 0;
        while (j < self.outputSize) : (j += 1) {
            self.outputs[batch_start * self.outputSize + j] = self.gaussian_bump(sum, j);
        }
    }
    self.last_inputs = inputs;
}

pub fn backwards(self: *Self, grads: []f64) void {
    std.debug.assert(grads.len == self.outputSize * self.batchSize);
    std.debug.assert(self.last_inputs.len == self.inputSize * self.batchSize);

    @memset(self.input_grads, 0);
    @memset(self.mu_grads, 0);
    @memset(self.sigma_grads, 0);

    var batch_start: usize = 0;
    while (batch_start < self.batchSize) : (batch_start += 1) {
        var sum: f64 = 0.0;
        var i: usize = 0;
        while (i < self.inputSize) : (i += 1) {
            sum += self.last_inputs[batch_start * self.inputSize + i];
        }
        var j: usize = 0;
        while (j < self.outputSize) : (j += 1) {
            const grad_index = batch_start * self.outputSize + j;
            const grad = grads[grad_index];
            const gaussian_derivative = self.gaussian_bump_derivative(sum, j);

            var k: usize = 0;
            while (k < self.inputSize) : (k += 1) {
                self.input_grads[batch_start * self.inputSize + k] += grad * gaussian_derivative;
            }

            self.mu_grads[j] += grad * ((sum - self.mus[j]) / (self.sigmas[j] * self.sigmas[j])) * self.gaussian_bump(sum, j);
            self.sigma_grads[j] += grad * (((sum - self.mus[j]) * (sum - self.mus[j])) / (self.sigmas[j] * self.sigmas[j] * self.sigmas[j])) * self.gaussian_bump(sum, j);
        }
    }
}

pub fn applyGradients(self: *Self) void {
    var i: usize = 0;
    while (i < self.outputSize) : (i += 1) {
        self.mus[i] -= 0.01 * self.mu_grads[i];
        self.sigmas[i] -= 0.01 * self.sigma_grads[i];
    }
}
