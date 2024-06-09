const std = @import("std");

pub fn Pyramid(comptime size: usize) type {
    return struct {
        last_inputs: []const f64 = undefined,
        fwd_out: []f64,
        bkw_out: []f64,

        const Self = @This();

        pub const threshold: f64 = 1.0;
        pub const leak_slope: f64 = 0.01;

        pub fn init(alloc: std.mem.Allocator) !Self {
            return Self{
                .last_inputs = try alloc.alloc(f64, size),
                .fwd_out = try alloc.alloc(f64, size),
                .bkw_out = try alloc.alloc(f64, size),
            };
        }

        pub fn forward(self: *Self, inputs: []f64) void {
            std.debug.assert(inputs.len == size);

            var i: usize = 0;
            while (i < inputs.len) : (i += 1) {
                const x = inputs[i];
                if (x < 0) {
                    self.fwd_out[i] = leak_slope * x;
                } else if (x < threshold) {
                    self.fwd_out[i] = x;
                } else if (x < 2 * threshold) {
                    self.fwd_out[i] = 2 * threshold - x;
                } else {
                    self.fwd_out[i] = leak_slope * (x - 2 * threshold);
                }
            }
            self.last_inputs = inputs;
        }

        pub fn backwards(self: *Self, grads: []f64) void {
            std.debug.assert(grads.len == size);
            var i: usize = 0;
            while (i < self.last_inputs.len) : (i += 1) {
                const x = self.last_inputs[i];
                if (x < 0) {
                    self.bkw_out[i] = leak_slope * grads[i];
                } else if (x < threshold) {
                    self.bkw_out[i] = grads[i];
                } else if (x < 2 * threshold) {
                    self.bkw_out[i] = -grads[i];
                } else {
                    self.bkw_out[i] = leak_slope * grads[i];
                }
            }
        }
    };
}
