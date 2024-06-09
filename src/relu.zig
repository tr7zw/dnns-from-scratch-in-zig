const std = @import("std");

pub fn Relu(comptime size: usize) type {
    return struct {
        last_inputs: []const f64 = undefined,
        fwd_out: []f64,
        bkw_out: []f64,

        const Self = @This();

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
                if (inputs[i] < 0) {
                    self.fwd_out[i] = 0.01 * inputs[i];
                } else {
                    self.fwd_out[i] = inputs[i];
                }
            }
            self.last_inputs = inputs;
        }

        pub fn backwards(self: *Self, grads: []f64) void {
            std.debug.assert(grads.len == size);
            var i: usize = 0;
            while (i < self.last_inputs.len) : (i += 1) {
                if (self.last_inputs[i] < 0) {
                    self.bkw_out[i] = 0.01 * grads[i];
                } else {
                    self.bkw_out[i] = grads[i];
                }
            }
        }
    };
}
