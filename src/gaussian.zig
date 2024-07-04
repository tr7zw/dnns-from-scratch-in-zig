const std = @import("std");

last_inputs: []const f64 = undefined,
fwd_out: []f64,
bkw_out: []f64,
batchSize: usize,
size: usize,

const Self = @This();
pub const mu = 1.0; // center of the bump
pub const sigma = 1.0; // width of the bump
const epsilon = 0.01;

pub fn init(
    alloc: std.mem.Allocator,
    batchSize: usize,
    size: usize,
) !Self {
    return Self{
        .last_inputs = try alloc.alloc(f64, size * batchSize),
        .fwd_out = try alloc.alloc(f64, size * batchSize),
        .bkw_out = try alloc.alloc(f64, size * batchSize),
        .batchSize = batchSize,
        .size = size,
    };
}
fn gaussian_bump(x: f64) f64 {
    const exponent = -((x - mu) * (x - mu)) / (2.0 * sigma * sigma);
    const gaussian = @exp(exponent);
    if (gaussian > epsilon) {
        return gaussian;
    } else {
        return epsilon * (x - 2 * mu);
    }
}

fn gaussian_bump_derivative(x: f64) f64 {
    const exponent = -((x - mu) * (x - mu)) / (2.0 * sigma * sigma);
    const gaussian = @exp(exponent);
    const gaussian_derivative = -((x - mu) / (sigma * sigma)) * gaussian;
    if (@abs(gaussian_derivative) > epsilon) {
        return gaussian_derivative;
    } else {
        return epsilon * std.math.sign(mu - x);
    }
}

pub fn forward(self: *Self, inputs: []f64) void {
    std.debug.assert(inputs.len == self.size * self.batchSize);

    var i: usize = 0;
    while (i < inputs.len) : (i += 1) {
        self.fwd_out[i] = gaussian_bump(inputs[i]);
    }
    self.last_inputs = inputs;
}

pub fn backwards(self: *Self, grads: []f64) void {
    std.debug.assert(grads.len == self.size * self.batchSize);
    var i: usize = 0;
    while (i < self.last_inputs.len) : (i += 1) {
        self.bkw_out[i] = grads[i] * gaussian_bump_derivative(self.last_inputs[i]);
    }
}
