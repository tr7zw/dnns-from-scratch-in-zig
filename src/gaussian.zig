const std = @import("std");

pub fn Activation(comptime size: usize) type {
    return struct {
        last_inputs: []const f64 = undefined,
        fwd_out: []f64,
        bkw_out: []f64,
        batchSize: usize,

        const Self = @This();
        pub const mu: f64 = 1.0; // center of the bump
        pub const sigma: f64 = 2.0; // width of the bump

        pub fn init(
            alloc: std.mem.Allocator,
            batchSize: usize,
        ) !Self {
            return Self{
                .last_inputs = try alloc.alloc(f64, size * batchSize),
                .fwd_out = try alloc.alloc(f64, size * batchSize),
                .bkw_out = try alloc.alloc(f64, size * batchSize),
                .batchSize = batchSize,
            };
        }

        fn gaussian_bump(x: f64) f64 {
            const exponent = -((x - mu) * (x - mu)) / (2.0 * sigma * sigma);
            return @exp(exponent);
        }

        fn gaussian_bump_derivative(x: f64) f64 {
            const exponent = -((x - mu) * (x - mu)) / (2.0 * sigma * sigma);
            return -((x - mu) / (sigma * sigma)) * @exp(exponent);
        }

        pub fn forward(self: *Self, inputs: []f64) void {
            std.debug.assert(inputs.len == size * self.batchSize);

            var i: usize = 0;
            while (i < inputs.len) : (i += 1) {
                self.fwd_out[i] = gaussian_bump(inputs[i]);
            }
            self.last_inputs = inputs;
        }

        pub fn backwards(self: *Self, grads: []f64) void {
            std.debug.assert(grads.len == size * self.batchSize);
            var i: usize = 0;
            while (i < self.last_inputs.len) : (i += 1) {
                self.bkw_out[i] = grads[i] * gaussian_bump_derivative(self.last_inputs[i]);
            }
        }
    };
}
