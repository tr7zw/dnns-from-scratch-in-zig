const std = @import("std");

pub fn Relu(size: usize) type {
    return struct {
        last_inputs: [size]f64 = undefined,
        fwd_out: [size]f64,
        bkw_out: [size]f64,
        const Self = @This();

        pub fn init() Self {
            return Self{
                .last_inputs = undefined,
                .fwd_out = [_]f64{0} ** size,
                .bkw_out = [_]f64{0} ** size,
            };
        }

        pub fn forward(self: *Self, inputs: [size]f64) *Self {
            //std.debug.assert(inputs.len == size);

            var i: usize = 0;
            while (i < inputs.len) : (i += 1) {
                if (inputs[i] < 0) {
                    self.fwd_out[i] = 0.01 * inputs[i];
                } else {
                    self.fwd_out[i] = inputs[i];
                }
            }
            self.last_inputs = inputs;
            return self;
        }

        pub fn backwards(self: *Self, grads: []f64) *Self {
            std.debug.assert(grads.len == size);
            var i: usize = 0;
            while (i < self.last_inputs.len) : (i += 1) {
                if (self.last_inputs[i] < 0) {
                    self.bkw_out[i] = 0.01 * grads[i];
                } else {
                    self.bkw_out[i] = grads[i];
                }
            }
            return self;
        }
    };
}
