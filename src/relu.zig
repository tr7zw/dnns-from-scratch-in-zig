const std = @import("std");

pub fn Relu(size: usize) type {
    return struct {
        last_inputs: []f64 = undefined,

        const Self = @This();

        var fwd_out: [size]f64 = [_]f64{0} ** size;
        pub fn forward(self: *Self, inputs: []f64) []f64 {
            std.debug.assert(inputs.len == size);

            var i: usize = 0;
            while (i < inputs.len) : (i += 1) {
                if (inputs[i] < 0) {
                    fwd_out[i] = 0.01 * inputs[i];
                } else {
                    fwd_out[i] = inputs[i];
                }
            }
            self.last_inputs = inputs;
            return &fwd_out;
        }

            var bkw_out: [size]f64 = [_]f64{0} ** size;
        pub fn backwards(self: *Self, grads: []f64) []f64 {
            std.debug.assert(grads.len == size);
            var i: usize = 0;
            while (i < self.last_inputs.len) : (i += 1) {
                if (self.last_inputs[i] < 0) {
                    bkw_out[i] = 0.01 * grads[i];
                } else {
                    bkw_out[i] = grads[i];
                }
            }
            return &bkw_out;
        }
    };
}
